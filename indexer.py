import os, json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Setup embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(
    collection_name="compressed_docs",
    embedding_function=embeddings,
    persist_directory="./DocIndex"
)

# Load all compressed summaries
output_dir = "./output"
texts, metadatas = [], []

if not os.path.exists(output_dir):
    print(f"❌ Error: {output_dir} directory not found!")
    exit(1)

files = [f for f in os.listdir(output_dir) if f.endswith("_compressed.json")]
print(f"📄 Found {len(files)} compressed JSON files")

for file in files:
    path = os.path.join(output_dir, file)
    print(f"  Processing: {file}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        facts = data["summary"]["key_facts"]
        print(f"    → {len(facts)} facts found")
        
        for fact in facts:
            texts.append(fact["statement"])
            
            # ✅ FIX: Convert char_range tuple to string
            source = fact["source"]
            char_range = source.get("char_range", [0, 0])
            
            metadatas.append({
                "doc_id": data["doc_id"],
                "section": source.get("section", ""),
                "paragraph_id": source.get("paragraph_id", ""),
                "char_range_start": char_range[0] if isinstance(char_range, (list, tuple)) else 0,
                "char_range_end": char_range[1] if isinstance(char_range, (list, tuple)) else 0,
                "salience": fact.get("salience", 0.0)
            })

print(f"\n📝 Total facts to index: {len(texts)}")

if len(texts) == 0:
    print("❌ No facts to index! Check your salience threshold in ingestion.py")
    exit(1)

# Add them to Chroma
print("Adding to ChromaDB...")
vector_store.add_texts(texts=texts, metadatas=metadatas)
print(f"✅ Successfully indexed {len(texts)} facts into ChromaDB")

# Verify
count = vector_store._collection.count()
print(f"✅ Verified: ChromaDB now contains {count} documents")