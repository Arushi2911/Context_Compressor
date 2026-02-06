import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# 1️⃣ Define embeddings first
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2️⃣ Then create Chroma vector store
vector_store = Chroma(
    collection_name="compressed_docs",
    embedding_function=embeddings,
    persist_directory="./DocIndex"
)

# 3️⃣ Then create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4️⃣ Initialize your LLM
model = ChatGroq(model="llama-3.3-70b-versatile")

SYSTEM_PROMPT = """
You are answering questions using compressed, traceable document summaries.
Rules:
- Use ONLY the retrieved context.
- Cite sources: (Document: <doc_id>, Section: <section>, Paragraph: <paragraph_id>).
- If not found, say "Information not available."
"""


# 3️⃣ Query loop
while True:
    query = input("\nAsk (or type 'exit'): ")
    if query.lower() == "exit":
        break

    results = retriever.invoke(query)
    if not results:
        print("\nAnswer:\nInformation not available.")
        continue

    context = ""
    for r in results:
        src = r.metadata
        context += (
            f"{r.page_content}\n"
            f"(Document: {src.get('doc_id')}, Section: {src.get('section')}, Paragraph: {src.get('paragraph_id')})\n\n"
        )

    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{query}"

    response = model.invoke(prompt)
    print("\nAnswer:\n", response.content)
