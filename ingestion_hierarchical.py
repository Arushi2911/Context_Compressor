from dataclasses import dataclass, field
from typing import List, Dict, Optional
import uuid
import re
import math
import pdfplumber
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import time

# Import token configuration
try:
    from token_config import (
        SECTION_SUMMARY_MAX_TOKENS,
        CHAPTER_SUMMARY_MAX_TOKENS,
        DOCUMENT_SUMMARY_MAX_TOKENS,
        DELAY_BETWEEN_CALLS,
        MAX_RETRIES,
        RETRY_DELAY,
        LINES_PER_SECTION,
        SECTIONS_PER_CHAPTER
    )
except ImportError:
    # Fallback defaults if token_config.py not found
    print("⚠️  Warning: token_config.py not found, using default values")
    SECTION_SUMMARY_MAX_TOKENS = 200
    CHAPTER_SUMMARY_MAX_TOKENS = 300
    DOCUMENT_SUMMARY_MAX_TOKENS = 400
    DELAY_BETWEEN_CALLS = 2.0
    MAX_RETRIES = 5
    RETRY_DELAY = 90
    LINES_PER_SECTION = 10
    SECTIONS_PER_CHAPTER = 5

# ---------- Token Budget Configuration ----------

@dataclass
class TokenBudget:
    """Token allocation for different parts of the system."""
    # Summarization budgets (for ingestion)
    section_summary_max: int = 300      # Max tokens per section summary
    chapter_summary_max: int = 400      # Max tokens per chapter summary  
    document_summary_max: int = 500     # Max tokens per document summary
    
    # Query budgets (for model.py - not used in ingestion)
    system_prompt_max: int = 500        # Max tokens for system prompt
    user_query_max: int = 200           # Max tokens for user query
    context_max: int = 3000             # Max tokens for retrieved context
    response_max: int = 1000            # Max tokens for LLM response
    
    # Rate limiting
    delay_between_calls: float = 1.0    # Seconds to wait between API calls
    max_retries: int = 3                # Max retries on rate limit
    retry_delay: int = 60               # Seconds to wait on rate limit error


class TokenBudgetManager:
    """Manages token budgets and rate limiting."""
    
    def __init__(self, budget: TokenBudget):
        self.budget = budget
        self.last_call_time = 0
        
    def wait_for_rate_limit(self):
        """Ensure minimum delay between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.budget.delay_between_calls:
            sleep_time = self.budget.delay_between_calls - time_since_last
            print(f"   ⏳ Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def call_with_retry(self, func, *args, **kwargs):
        """Call a function with retry logic for rate limits."""
        for attempt in range(self.budget.max_retries):
            try:
                self.wait_for_rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < self.budget.max_retries - 1:
                        wait_time = self.budget.retry_delay * (attempt + 1)
                        print(f"   ⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{self.budget.max_retries}...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ Max retries reached. Error: {e}")
                        raise
                else:
                    raise


# Initialize global token budget from config
TOKEN_BUDGET = TokenBudget(
    section_summary_max=SECTION_SUMMARY_MAX_TOKENS,
    chapter_summary_max=CHAPTER_SUMMARY_MAX_TOKENS,
    document_summary_max=DOCUMENT_SUMMARY_MAX_TOKENS,
    delay_between_calls=DELAY_BETWEEN_CALLS,
    max_retries=MAX_RETRIES,
    retry_delay=RETRY_DELAY,
)

# ---------- Data Structures ----------

@dataclass
class SourceRef:
    doc_id: str
    page: int
    line_number: int
    char_range: tuple
    level: str  # 'line', 'section', 'chapter'
    parent_id: Optional[str] = None  # ID of parent summary

@dataclass
class SummaryUnit:
    unit_id: str
    level: str  # 'line', 'section', 'chapter', 'document'
    content: str  # Original text for lines, summary for higher levels
    summary: Optional[str] = None  # Summary of this unit
    source: SourceRef = None
    children_ids: List[str] = field(default_factory=list)  # IDs of child units
    metadata: Dict = field(default_factory=dict)


# ---------- PDF Text Extraction ----------

def extract_text_with_structure(pdf_path: str) -> List[Dict]:
    """Extract text from PDF with page and line structure."""
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                pages_data.append({
                    'page_num': page_num,
                    'lines': lines
                })
    
    return pages_data


# ---------- Hierarchical Summarization ----------

class HierarchicalSummarizer:
    def __init__(self, budget: TokenBudget):
        self.model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        self.budget_manager = TokenBudgetManager(budget)
        self.budget = budget
        
    def _call_model(self, prompt: str, max_tokens: int) -> str:
        """Call model with retry logic."""
        def _invoke():
            response = self.model.invoke(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.content.strip()
        
        return self.budget_manager.call_with_retry(_invoke)
        
    def summarize_lines_to_section(self, lines: List[str], max_lines_per_section: int = 10) -> str:
        """Summarize a group of lines into a section summary."""
        combined_text = "\n".join(lines)
        
        # Truncate input if too long (rough estimate: 1 token ≈ 4 chars)
        max_input_chars = 2000  # ~500 tokens
        if len(combined_text) > max_input_chars:
            combined_text = combined_text[:max_input_chars] + "..."
        
        prompt = f"""Summarize the following text into 2-3 concise sentences that capture the key information:

{combined_text}

Provide only the summary, no preamble."""

        return self._call_model(prompt, self.budget.section_summary_max)
    
    def summarize_sections_to_chapter(self, section_summaries: List[str]) -> str:
        """Summarize multiple section summaries into a chapter summary."""
        combined = "\n\n".join([f"Section {i+1}: {s}" for i, s in enumerate(section_summaries)])
        
        # Truncate if too long
        max_input_chars = 3000  # ~750 tokens
        if len(combined) > max_input_chars:
            combined = combined[:max_input_chars] + "..."
        
        prompt = f"""Synthesize the following section summaries into a cohesive 3-4 sentence chapter summary:

{combined}

Provide only the summary, no preamble."""

        return self._call_model(prompt, self.budget.chapter_summary_max)
    
    def summarize_chapters_to_document(self, chapter_summaries: List[str]) -> str:
        """Summarize multiple chapters into a document-level summary."""
        combined = "\n\n".join([f"Chapter {i+1}: {s}" for i, s in enumerate(chapter_summaries)])
        
        # Truncate if too long
        max_input_chars = 4000  # ~1000 tokens
        if len(combined) > max_input_chars:
            combined = combined[:max_input_chars] + "..."
        
        prompt = f"""Create a comprehensive document summary (4-5 sentences) from these chapter summaries:

{combined}

Provide only the summary, no preamble."""

        return self._call_model(prompt, self.budget.document_summary_max)


# ---------- Main Processing ----------

def build_hierarchical_index(pdf_path: Path, 
                             lines_per_section: int = 10,
                             sections_per_chapter: int = 5,
                             token_budget: TokenBudget = None) -> Dict:
    """
    Build hierarchical summary structure:
    Lines → Sections → Chapters → Document
    """
    if token_budget is None:
        token_budget = TOKEN_BUDGET
        
    doc_id = pdf_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")
    print(f"📊 Token Budget Configuration:")
    print(f"   Section summaries: max {token_budget.section_summary_max} tokens")
    print(f"   Chapter summaries: max {token_budget.chapter_summary_max} tokens")
    print(f"   Document summary: max {token_budget.document_summary_max} tokens")
    print(f"   Rate limit delay: {token_budget.delay_between_calls}s between calls")
    
    # Extract text with structure
    pages_data = extract_text_with_structure(str(pdf_path))
    print(f"✅ Extracted {len(pages_data)} pages")
    
    # Initialize summarizer with token budget
    summarizer = HierarchicalSummarizer(token_budget)
    
    # Storage for all units
    all_units = {}
    
    # Level 1: Process Lines
    print("\n📝 Level 1: Processing lines...")
    line_units = []
    char_offset = 0
    
    for page_data in pages_data:
        page_num = page_data['page_num']
        for line_num, line_text in enumerate(page_data['lines'], 1):
            line_id = str(uuid.uuid4())
            char_start = char_offset
            char_end = char_offset + len(line_text)
            
            unit = SummaryUnit(
                unit_id=line_id,
                level='line',
                content=line_text,
                summary=line_text,  # Lines summarize to themselves
                source=SourceRef(
                    doc_id=doc_id,
                    page=page_num,
                    line_number=line_num,
                    char_range=(char_start, char_end),
                    level='line'
                )
            )
            
            line_units.append(unit)
            all_units[line_id] = unit
            char_offset = char_end + 1
    
    print(f"   Created {len(line_units)} line units")
    
    # Level 2: Group lines into sections and summarize
    print("\n📑 Level 2: Creating section summaries...")
    section_units = []
    
    for i in range(0, len(line_units), lines_per_section):
        section_lines = line_units[i:i + lines_per_section]
        section_id = str(uuid.uuid4())
        
        # Get line texts
        line_texts = [unit.content for unit in section_lines]
        
        # Summarize section
        print(f"   Summarizing section {len(section_units) + 1} ({len(section_lines)} lines)...")
        section_summary = summarizer.summarize_lines_to_section(line_texts)
        
        # Create section unit
        first_line = section_lines[0]
        last_line = section_lines[-1]
        
        unit = SummaryUnit(
            unit_id=section_id,
            level='section',
            content="\n".join(line_texts),  # Store original content
            summary=section_summary,
            source=SourceRef(
                doc_id=doc_id,
                page=first_line.source.page,
                line_number=first_line.source.line_number,
                char_range=(
                    first_line.source.char_range[0],
                    last_line.source.char_range[1]
                ),
                level='section'
            ),
            children_ids=[unit.unit_id for unit in section_lines],
            metadata={'num_lines': len(section_lines)}
        )
        
        # Update parent references in line units
        for line_unit in section_lines:
            line_unit.source.parent_id = section_id
        
        section_units.append(unit)
        all_units[section_id] = unit
    
    print(f"   Created {len(section_units)} section summaries")
    
    # Level 3: Group sections into chapters and summarize
    print("\n📚 Level 3: Creating chapter summaries...")
    chapter_units = []
    
    for i in range(0, len(section_units), sections_per_chapter):
        chapter_sections = section_units[i:i + sections_per_chapter]
        chapter_id = str(uuid.uuid4())
        
        # Get section summaries
        section_summaries = [unit.summary for unit in chapter_sections]
        
        # Summarize chapter
        print(f"   Summarizing chapter {len(chapter_units) + 1} ({len(chapter_sections)} sections)...")
        chapter_summary = summarizer.summarize_sections_to_chapter(section_summaries)
        
        # Create chapter unit
        first_section = chapter_sections[0]
        last_section = chapter_sections[-1]
        
        unit = SummaryUnit(
            unit_id=chapter_id,
            level='chapter',
            content="\n\n".join([s.content for s in chapter_sections]),
            summary=chapter_summary,
            source=SourceRef(
                doc_id=doc_id,
                page=first_section.source.page,
                line_number=first_section.source.line_number,
                char_range=(
                    first_section.source.char_range[0],
                    last_section.source.char_range[1]
                ),
                level='chapter'
            ),
            children_ids=[unit.unit_id for unit in chapter_sections],
            metadata={'num_sections': len(chapter_sections)}
        )
        
        # Update parent references in section units
        for section_unit in chapter_sections:
            section_unit.source.parent_id = chapter_id
        
        chapter_units.append(unit)
        all_units[chapter_id] = unit
    
    print(f"   Created {len(chapter_units)} chapter summaries")
    
    # Level 4: Create document-level summary
    print("\n📖 Level 4: Creating document summary...")
    chapter_summaries = [unit.summary for unit in chapter_units]
    document_summary = summarizer.summarize_chapters_to_document(chapter_summaries)
    
    document_id = str(uuid.uuid4())
    document_unit = SummaryUnit(
        unit_id=document_id,
        level='document',
        content="",  # Full document text not stored at this level
        summary=document_summary,
        source=SourceRef(
            doc_id=doc_id,
            page=1,
            line_number=1,
            char_range=(0, char_offset),
            level='document'
        ),
        children_ids=[unit.unit_id for unit in chapter_units],
        metadata={
            'num_chapters': len(chapter_units),
            'num_sections': len(section_units),
            'num_lines': len(line_units)
        }
    )
    
    # Update parent references in chapter units
    for chapter_unit in chapter_units:
        chapter_unit.source.parent_id = document_id
    
    all_units[document_id] = document_unit
    
    print(f"   Document summary created")
    print(f"\n✅ Hierarchical structure complete!")
    print(f"   Document → {len(chapter_units)} Chapters → {len(section_units)} Sections → {len(line_units)} Lines")
    
    return {
        'doc_id': doc_id,
        'document_unit': document_unit,
        'all_units': all_units,
        'hierarchy': {
            'document': document_unit,
            'chapters': chapter_units,
            'sections': section_units,
            'lines': line_units
        }
    }


def save_hierarchical_index(result: Dict, output_dir: str):
    """Save the hierarchical index to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    doc_id = result['doc_id']
    
    # Convert units to serializable format
    def unit_to_dict(unit: SummaryUnit):
        return {
            'unit_id': unit.unit_id,
            'level': unit.level,
            'content': unit.content if unit.level == 'line' else unit.content[:200] + '...',  # Truncate for storage
            'summary': unit.summary,
            'source': {
                'doc_id': unit.source.doc_id,
                'page': unit.source.page,
                'line_number': unit.source.line_number,
                'char_range': unit.source.char_range,
                'level': unit.source.level,
                'parent_id': unit.source.parent_id
            },
            'children_ids': unit.children_ids,
            'metadata': unit.metadata
        }
    
    # Save complete index
    index_data = {
        'doc_id': doc_id,
        'document_summary': result['document_unit'].summary,
        'hierarchy_stats': result['document_unit'].metadata,
        'all_units': {
            unit_id: unit_to_dict(unit) 
            for unit_id, unit in result['all_units'].items()
        }
    }
    
    output_path = Path(output_dir) / f"{doc_id}_hierarchical.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"\n💾 Saved hierarchical index to: {output_path}")


# ---------- Entry Point ----------

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Verify Groq API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment")
        print("   Please add it to your .env file")
        exit(1)
    
    collection_dir = "collection"
    output_dir = "output"
    
    collection_path = Path(collection_dir)
    
    if not collection_path.exists():
        print(f"❌ ERROR: Directory '{collection_dir}' does not exist!")
        exit(1)
    
    pdf_files = list(collection_path.glob("*.pdf")) + list(collection_path.glob("*.PDF"))
    
    if len(pdf_files) == 0:
        print(f"❌ ERROR: No PDF files found in '{collection_dir}'")
        exit(1)
    
    print(f"\n📚 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    for pdf_file in pdf_files:
        result = build_hierarchical_index(
            pdf_file,
            lines_per_section=LINES_PER_SECTION,
            sections_per_chapter=SECTIONS_PER_CHAPTER
        )
        save_hierarchical_index(result, output_dir)
    
    print(f"\n✅ All documents processed!")

