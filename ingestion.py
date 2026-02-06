from dataclasses import dataclass, field
from typing import List, Dict, Optional
import uuid
import re
import math
import pdfplumber
from pathlib import Path
import json
import os
import spacy

# ---------- Data Structures ----------

@dataclass
class SourceRef:
    doc_id: str
    section: str
    paragraph_id: str
    char_range: tuple

@dataclass
class Atom:
    atom_id: str
    text: str
    source: SourceRef
    metadata: Dict
    salience_score: float = 0.0
    risk_flags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[tuple] = field(default_factory=list)
    dropped_reason: Optional[str] = None


# ---------- PDF Text Extraction ----------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


# ---------- Text → Atoms ----------

def extract_atoms_from_text(text: str, doc_id: str, section: str, metadata: Dict) -> List[Atom]:
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    print(f"  📄 Found {len(paragraphs)} paragraphs (>50 chars)")
    atoms = []
    cursor = 0

    for p in paragraphs:
        start = text.find(p, cursor)
        end = start + len(p)
        cursor = end

        atoms.append(
            Atom(
                atom_id=str(uuid.uuid4()),
                text=p,
                source=SourceRef(
                    doc_id=doc_id,
                    section=section,
                    paragraph_id=str(uuid.uuid4()),
                    char_range=(start, end)
                ),
                metadata=metadata
            )
        )
    return atoms


# ---------- Scoring & NLP ----------

IMPORTANT_PATTERNS = [
    r"\bshall\b", r"\bmust\b", r"\bonly if\b", r"\bunless\b",
    r"\bnotwithstanding\b", r"\brisk\b", r"\bpenalty\b",
    r"\bthreshold\b", r"\bmaximum\b", r"\bminimum\b",
    r"\b\d+(\.\d+)?\b"
]


def salience_score(atom: Atom) -> float:
    score = 0.0
    text = atom.text.lower()

    for pattern in IMPORTANT_PATTERNS:
        if re.search(pattern, text):
            score += 1.5

    if len(text) > 300:
        score += math.log(len(text))

    return score


# Load NLP model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")


def extract_entities_and_relations(atom: Atom):
    doc = nlp(atom.text)
    atom.entities = [ent.text for ent in doc.ents]
    relations = []

    for token in doc:
        if token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ in ("dobj", "pobj"):
                    relations.append((token.lemma_, child.text))

    atom.relations = relations


def detect_contradictions(atoms: List[Atom]) -> List[tuple]:
    contradictions = []

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            a, b = atoms[i], atoms[j]
            shared_entities = set(a.entities) & set(b.entities)
            if not shared_entities:
                continue
            if "shall" in a.text.lower() and "shall not" in b.text.lower():
                contradictions.append((a.atom_id, b.atom_id))

    return contradictions


# ---------- Compression ----------

def compress_atoms(atoms: List[Atom], salience_threshold: float) -> Dict:
    kept, dropped = [], []

    for atom in atoms:
        atom.salience_score = salience_score(atom)
        extract_entities_and_relations(atom)

        if atom.salience_score >= salience_threshold:
            kept.append(atom)
        else:
            atom.dropped_reason = "Low salience score"
            dropped.append(atom)
    
    print(f"  ✅ Kept: {len(kept)} atoms (score >= {salience_threshold})")
    print(f"  ❌ Dropped: {len(dropped)} atoms (score < {salience_threshold})")

    return {"kept": kept, "dropped": dropped}


def summarize_atoms(atoms: List[Atom]) -> Dict:
    """Keep full statements instead of truncating."""
    return {
        "key_facts": [
            {
                "statement": atom.text,
                "source": atom.source.__dict__,
                "salience": atom.salience_score
            }
            for atom in atoms
        ]
    }


# ---------- Memory Class ----------

class ContextMemory:
    def __init__(self):
        self.levels = {}

    def store(self, level: str, summary: Dict):
        self.levels[level] = summary

    def compress_and_merge(self, new_summary: Dict):
        self.levels["global"] = {"merged": True, "content": new_summary}


# ---------- Main Processing ----------

def process_pdf(pdf_path: Path, salience_threshold: float = 0.5) -> Dict:
    doc_id = pdf_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")
    
    raw_text = extract_text_from_pdf(str(pdf_path))
    print(f"✅ Extracted {len(raw_text)} characters")

    atoms = extract_atoms_from_text(
        text=raw_text,
        doc_id=doc_id,
        section="FULL_DOCUMENT",
        metadata={"filename": pdf_path.name}
    )

    compressed = compress_atoms(atoms, salience_threshold)
    contradictions = detect_contradictions(compressed["kept"])
    summary = summarize_atoms(compressed["kept"])
    atom_index = {a.atom_id: a for a in atoms}

    print(f"  📊 Summary contains {len(summary['key_facts'])} key facts")

    return {
        "doc_id": doc_id,
        "summary": summary,
        "kept_atoms": compressed["kept"],
        "dropped_atoms": compressed["dropped"],
        "contradictions": contradictions,
        "atom_index": atom_index
    }


def run_pdf_collection(collection_dir: str, output_dir: str, salience_threshold: float = 0.5):
    os.makedirs(output_dir, exist_ok=True)

    memory = ContextMemory()
    collection_path = Path(collection_dir)

    # Check if collection directory exists
    if not collection_path.exists():
        print(f"❌ ERROR: Directory '{collection_dir}' does not exist!")
        print(f"   Please create it and add PDF files.")
        return

    # case-insensitive glob (PDF/Pdf/pdf)
    pdf_files = list(collection_path.glob("*.pdf")) + list(collection_path.glob("*.PDF"))
    
    if len(pdf_files) == 0:
        print(f"❌ ERROR: No PDF files found in '{collection_dir}'")
        print(f"   Please add PDF files to this directory.")
        return
    
    print(f"\n📚 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")

    global_atoms = []

    for pdf_file in pdf_files:
        result = process_pdf(pdf_file, salience_threshold=salience_threshold)

        # store per-doc summary in memory
        memory.store(result["doc_id"], result["summary"])

        # accumulate kept atoms for cross-doc reasoning
        global_atoms.extend(result["kept_atoms"])

        # write per-document output
        out_path = Path(output_dir) / f"{result['doc_id']}_compressed.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "doc_id": result["doc_id"],
                "summary": result["summary"],
                "contradictions": result["contradictions"],
                "kept_atoms": [
                    {
                        "atom_id": a.atom_id,
                        "text": a.text,
                        "source": a.source.__dict__,
                        "salience": a.salience_score,
                        "entities": a.entities,
                        "relations": a.relations
                    }
                    for a in result["kept_atoms"]
                ],
                "dropped_atoms": [
                    {
                        "atom_id": a.atom_id,
                        "reason": a.dropped_reason,
                        "source": a.source.__dict__
                    }
                    for a in result["dropped_atoms"]
                ]
            }, f, indent=2)
        
        print(f"  💾 Saved to: {out_path}")

    # ---- cross-document layer ----
    cross_doc_contradictions = detect_contradictions(global_atoms)

    memory.compress_and_merge({
        "total_documents": len(pdf_files),
        "global_contradictions": cross_doc_contradictions
    })

    # write global memory
    global_path = Path(output_dir) / "GLOBAL_CONTEXT.json"
    with open(global_path, "w", encoding="utf-8") as f:
        json.dump(memory.levels, f, indent=2)
    
    print(f"\n💾 Saved global context to: {global_path}")
    print(f"\n✅ Collection processing complete!")
    print(f"   Total atoms kept across all docs: {len(global_atoms)}")


# ---------- Entry Point ----------

if __name__ == "__main__":
    run_pdf_collection(
        collection_dir="collection",
        output_dir="output",
        salience_threshold=0.5  # LOWERED threshold
    )