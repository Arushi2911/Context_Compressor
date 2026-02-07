import os, json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

# Setup embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create separate collections for each level
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

# Load hierarchical data
output_dir = "./output"
if not os.path.exists(output_dir):
    print(f"❌ Error: {output_dir} directory not found!")
    exit(1)

files = [f for f in os.listdir(output_dir) if f.endswith("_hierarchical.json")]
print(f"📄 Found {len(files)} hierarchical JSON files")

if len(files) == 0:
    print("❌ No hierarchical files found! Run ingestion_hierarchical.py first.")
    exit(1)

# Storage for texts and metadata at each level
doc_texts, doc_metas = [], []
chap_texts, chap_metas = [], []
sect_texts, sect_metas = [], []
line_texts, line_metas = [], []

for file in files:
    path = os.path.join(output_dir, file)
    print(f"\n📂 Processing: {file}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        all_units = data["all_units"]
        
        for unit_id, unit in all_units.items():
            level = unit["level"]
            summary = unit["summary"]
            source = unit["source"]
            
            if level == "document":
                doc_texts.append(summary)
                doc_metas.append({
                    "unit_id": unit_id,
                    "doc_id": source["doc_id"],
                    "level": level,
                    "num_chapters": unit["metadata"].get("num_chapters", 0),
                    "num_sections": unit["metadata"].get("num_sections", 0),
                    "num_lines": unit["metadata"].get("num_lines", 0),
                    "children_ids": ",".join(unit["children_ids"])
                })
                print(f"   ✓ Document summary")
            
            elif level == "chapter":
                chap_texts.append(summary)
                chap_metas.append({
                    "unit_id": unit_id,
                    "doc_id": source["doc_id"],
                    "level": level,
                    "page": source["page"],
                    "parent_id": source.get("parent_id", ""),
                    "children_ids": ",".join(unit["children_ids"]),
                    "num_sections": unit["metadata"].get("num_sections", 0)
                })
            
            elif level == "section":
                sect_texts.append(summary)
                sect_metas.append({
                    "unit_id": unit_id,
                    "doc_id": source["doc_id"],
                    "level": level,
                    "page": source["page"],
                    "line_number": source["line_number"],
                    "parent_id": source.get("parent_id", ""),
                    "children_ids": ",".join(unit["children_ids"]),
                    "num_lines": unit["metadata"].get("num_lines", 0)
                })
            
            elif level == "line":
                line_texts.append(summary)  # For lines, summary is the original text
                line_metas.append({
                    "unit_id": unit_id,
                    "doc_id": source["doc_id"],
                    "level": level,
                    "page": source["page"],
                    "line_number": source["line_number"],
                    "char_range_start": source["char_range"][0],
                    "char_range_end": source["char_range"][1],
                    "parent_id": source.get("parent_id", "")
                })

print(f"\n📊 Indexing Statistics:")
print(f"   Documents: {len(doc_texts)}")
print(f"   Chapters: {len(chap_texts)}")
print(f"   Sections: {len(sect_texts)}")
print(f"   Lines: {len(line_texts)}")

# Index each level
if len(doc_texts) > 0:
    print("\n🔍 Indexing document summaries...")
    document_store.add_texts(texts=doc_texts, metadatas=doc_metas)
    print(f"   ✅ Indexed {len(doc_texts)} documents")

if len(chap_texts) > 0:
    print("\n🔍 Indexing chapter summaries...")
    chapter_store.add_texts(texts=chap_texts, metadatas=chap_metas)
    print(f"   ✅ Indexed {len(chap_texts)} chapters")

if len(sect_texts) > 0:
    print("\n🔍 Indexing section summaries...")
    section_store.add_texts(texts=sect_texts, metadatas=sect_metas)
    print(f"   ✅ Indexed {len(sect_texts)} sections")

if len(line_texts) > 0:
    print("\n🔍 Indexing line content...")
    line_store.add_texts(texts=line_texts, metadatas=line_metas)
    print(f"   ✅ Indexed {len(line_texts)} lines")

print(f"\n✅ Hierarchical indexing complete!")
print(f"   Users will query document summaries first")
print(f"   Then can drill down: document → chapter → section → line")
