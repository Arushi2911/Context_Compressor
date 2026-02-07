import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

load_dotenv()

# Import token configuration
try:
    from token_config import (
        SYSTEM_PROMPT_MAX_TOKENS,
        CONTEXT_MAX_TOKENS,
        RESPONSE_MAX_TOKENS
    )
except ImportError:
    print("⚠️  Warning: token_config.py not found, using default values")
    SYSTEM_PROMPT_MAX_TOKENS = 500
    CONTEXT_MAX_TOKENS = 3000
    RESPONSE_MAX_TOKENS = 1000

# Setup embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load all hierarchy levels
document_store = Chroma(
    collection_name="document_summaries",
    embedding_function=embeddings,
    persist_directory="./DocIndex/document"
)

chapter_store = Chroma(
    collection_name="chapter_summaries",
    embedding_function=embeddings,
    persist_directory="./DocIndex/chapter"
)

section_store = Chroma(
    collection_name="section_summaries",
    embedding_function=embeddings,
    persist_directory="./DocIndex/section"
)

line_store = Chroma(
    collection_name="line_content",
    embedding_function=embeddings,
    persist_directory="./DocIndex/line"
)

# Initialize LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=RESPONSE_MAX_TOKENS
)

print(f"\n📊 Query Token Budget:")
print(f"   Max response tokens: {RESPONSE_MAX_TOKENS}")
print(f"   Max context tokens: {CONTEXT_MAX_TOKENS}")

# Load the full hierarchical data for tracing
def load_hierarchy_data():
    """Load all hierarchical JSON files for metadata lookup."""
    output_dir = "./output"
    hierarchy_map = {}
    
    files = [f for f in os.listdir(output_dir) if f.endswith("_hierarchical.json")]
    
    for file in files:
        path = os.path.join(output_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Index all units by their ID
            for unit_id, unit in data["all_units"].items():
                hierarchy_map[unit_id] = unit
    
    return hierarchy_map

print("📚 Loading hierarchy data...")
hierarchy_map = load_hierarchy_data()
print(f"✅ Loaded {len(hierarchy_map)} units from hierarchy")


def get_children(unit_id: str, hierarchy_map: dict) -> list:
    """Get all children of a unit."""
    unit = hierarchy_map.get(unit_id)
    if not unit:
        return []
    
    children_ids = unit.get("children_ids", [])
    return [hierarchy_map[cid] for cid in children_ids if cid in hierarchy_map]


def truncate_context(context: str, max_tokens: int) -> str:
    """
    Truncate context to approximately max_tokens.
    Uses rough estimate: 1 token ≈ 4 characters.
    """
    max_chars = max_tokens * 4
    if len(context) <= max_chars:
        return context
    
    # Truncate and add indicator
    truncated = context[:max_chars]
    # Try to cut at last newline to avoid mid-sentence cuts
    last_newline = truncated.rfind('\n')
    if last_newline > max_chars * 0.8:  # If we're not losing too much
        truncated = truncated[:last_newline]
    
    return truncated + "\n\n[... context truncated to fit token budget ...]"
    if not unit:
        return []
    
    children_ids = unit.get("children_ids", [])
    return [hierarchy_map[cid] for cid in children_ids if cid in hierarchy_map]


def format_unit_citation(unit: dict) -> str:
    """Format a unit with its citation."""
    source = unit["source"]
    level = unit["level"]
    
    if level == "document":
        return f"📖 Document: {source['doc_id']}"
    elif level == "chapter":
        return f"📚 Chapter in {source['doc_id']} (Page {source['page']})"
    elif level == "section":
        return f"📑 Section in {source['doc_id']} (Page {source['page']}, Line {source['line_number']})"
    elif level == "line":
        return f"📝 Line in {source['doc_id']} (Page {source['page']}, Line {source['line_number']}, Chars {source['char_range'][0]}-{source['char_range'][1]})"
    return ""


def answer_query(query: str, detail_level: str = "auto"):
    """
    Answer a query using hierarchical retrieval.
    
    detail_level: 'auto', 'document', 'chapter', 'section', 'line'
    """
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Step 1: Search at document level first
    print("\n🔍 Searching document-level summaries...")
    doc_results = document_store.similarity_search(query, k=3)
    
    if not doc_results:
        return "❌ No relevant documents found."
    
    # Get the most relevant document
    top_doc = doc_results[0]
    doc_unit_id = top_doc.metadata["unit_id"]
    
    print(f"✅ Found relevant document: {top_doc.metadata['doc_id']}")
    print(f"   Document Summary: {top_doc.page_content[:200]}...")
    
    # If auto mode, determine if we need more detail based on query specificity
    if detail_level == "auto":
        # Check if query is asking for specific details
        specific_keywords = ["specific", "detail", "exact", "precisely", "how", "what exactly", "line"]
        if any(kw in query.lower() for kw in specific_keywords):
            detail_level = "section"  # Go deeper for specific queries
        else:
            detail_level = "chapter"  # Stay at chapter for general queries
    
    # Step 2: Drill down to chapters
    if detail_level in ["chapter", "section", "line"]:
        print(f"\n🔍 Drilling down to chapter level...")
        doc_unit = hierarchy_map[doc_unit_id]
        chapter_ids = doc_unit["children_ids"]
        
        # Search chapters
        chapter_results = chapter_store.similarity_search(query, k=5)
        # Filter to only chapters from the relevant document
        relevant_chapters = [c for c in chapter_results if c.metadata["parent_id"] == doc_unit_id][:3]
        
        if not relevant_chapters:
            # Fallback to document summary
            return format_answer(query, [(top_doc, doc_unit)], "document")
        
        print(f"✅ Found {len(relevant_chapters)} relevant chapters")
        
        # Step 3: Drill down to sections if needed
        if detail_level in ["section", "line"]:
            print(f"\n🔍 Drilling down to section level...")
            relevant_chapter_ids = [c.metadata["unit_id"] for c in relevant_chapters]
            
            section_results = section_store.similarity_search(query, k=10)
            # Filter to sections from relevant chapters
            relevant_sections = [
                s for s in section_results 
                if s.metadata.get("parent_id") in relevant_chapter_ids
            ][:5]
            
            if not relevant_sections:
                # Fallback to chapters
                chapter_units = [hierarchy_map[c.metadata["unit_id"]] for c in relevant_chapters]
                return format_answer(query, list(zip(relevant_chapters, chapter_units)), "chapter")
            
            print(f"✅ Found {len(relevant_sections)} relevant sections")
            
            # Step 4: Drill down to lines if needed
            if detail_level == "line":
                print(f"\n🔍 Drilling down to line level...")
                relevant_section_ids = [s.metadata["unit_id"] for s in relevant_sections]
                
                line_results = line_store.similarity_search(query, k=15)
                # Filter to lines from relevant sections
                relevant_lines = [
                    l for l in line_results 
                    if l.metadata.get("parent_id") in relevant_section_ids
                ][:10]
                
                if not relevant_lines:
                    # Fallback to sections
                    section_units = [hierarchy_map[s.metadata["unit_id"]] for s in relevant_sections]
                    return format_answer(query, list(zip(relevant_sections, section_units)), "section")
                
                print(f"✅ Found {len(relevant_lines)} relevant lines")
                line_units = [hierarchy_map[l.metadata["unit_id"]] for l in relevant_lines]
                return format_answer(query, list(zip(relevant_lines, line_units)), "line")
            
            # Return section-level answer
            section_units = [hierarchy_map[s.metadata["unit_id"]] for s in relevant_sections]
            return format_answer(query, list(zip(relevant_sections, section_units)), "section")
        
        # Return chapter-level answer
        chapter_units = [hierarchy_map[c.metadata["unit_id"]] for c in relevant_chapters]
        return format_answer(query, list(zip(relevant_chapters, chapter_units)), "chapter")
    
    # Return document-level answer
    return format_answer(query, [(top_doc, doc_unit)], "document")


def format_answer(query: str, results_with_units: list, level: str) -> str:
    """Format the answer using LLM with retrieved content."""
    
    context = f"Level: {level.upper()}\n\n"
    
    for i, (result, unit) in enumerate(results_with_units, 1):
        context += f"[Source {i}]\n"
        context += f"{result.page_content}\n"
        context += f"{format_unit_citation(unit)}\n\n"
    
    # Truncate context to fit token budget
    context = truncate_context(context, CONTEXT_MAX_TOKENS)
    
    prompt = f"""You are answering questions using hierarchical document summaries.

The information is organized in levels: Document → Chapter → Section → Line
You are currently viewing {level.upper()}-level summaries.

INSTRUCTIONS:
1. Answer the question using ONLY the provided context
2. Synthesize information from the sources naturally
3. Cite sources using [Source N] notation
4. If the current level doesn't have enough detail, mention that more detailed information may be available at lower levels
5. Keep the answer concise and focused

Context:
{context}

Question: {query}

Answer:"""

    response = model.invoke(prompt)
    answer = response.content
    
    # Add metadata about the hierarchy level
    footer = f"\n\n{'─'*60}\n"
    footer += f"📊 Answer generated from {level.upper()}-level summaries\n"
    footer += f"💡 Type 'drill' to see more detailed {get_next_level(level)} information"
    
    return answer + footer


def get_next_level(current_level: str) -> str:
    """Get the next level down in hierarchy."""
    levels = ["document", "chapter", "section", "line"]
    try:
        idx = levels.index(current_level)
        return levels[idx + 1] if idx + 1 < len(levels) else "line"
    except ValueError:
        return "section"


# Main query loop
print("\n" + "="*60)
print("Hierarchical Document QA System")
print("="*60)
print("\nCommands:")
print("  - Ask any question (answered from document summaries)")
print("  - Type 'drill' to get more detailed answers")
print("  - Type 'detail:chapter|section|line' to specify level")
print("  - Type 'exit' to quit")
print("="*60)

current_query = None
current_level = "auto"

while True:
    if not current_query:
        user_input = input("\n💬 Ask a question: ").strip()
    else:
        user_input = input("\n💬 Continue (or ask new question): ").strip()
    
    if user_input.lower() == "exit":
        break
    
    # Check for detail level command
    if user_input.lower().startswith("detail:"):
        current_level = user_input.split(":")[1]
        if current_query:
            answer = answer_query(current_query, detail_level=current_level)
            print(f"\n{answer}")
        continue
    
    # Check for drill command
    if user_input.lower() == "drill":
        if not current_query:
            print("❌ No active query to drill down on")
            continue
        
        # Go one level deeper
        next_level = get_next_level(current_level if current_level != "auto" else "document")
        current_level = next_level
        answer = answer_query(current_query, detail_level=current_level)
        print(f"\n{answer}")
        continue
    
    # New query
    current_query = user_input
    current_level = "auto"
    answer = answer_query(current_query, detail_level=current_level)
    print(f"\n{answer}")
