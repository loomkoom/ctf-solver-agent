import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

# local vector DB
client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5"
)
collection = client.get_or_create_collection(
    name="ctf_knowledge",
    embedding_function=embedding_function,
)

def poc_ingest():
    # Example: Indexing angstrom/2021/pwn/pawn/readme.md
    # We store 'angstrom/2021/pwn/pawn' as the source_path metadata
    sample_path = "angstrom/2021/pwn/pawn"
    content = "This challenge involves a simple buffer overflow in the pawn binary."

    collection.add(
        documents=[content],
        metadatas=[{"source_path": sample_path}],
        ids=["pawn_2021"]
    )
    print("✅ Sample ingested with metadata.")

def poc_query(query):
    results = collection.query(query_texts=[query], n_results=1)
    doc = results['documents'][0][0]
    meta_path = results['metadatas'][0][0]['source_path']

    print(f"🔍 Found relevant writeup: {doc}")
    print(f"📂 MATCHING ARTIFACT FOLDER: data/artifacts/{meta_path}")

    # Simple check if folder exists
    artifact_dir = Path("data/artifacts") / meta_path
    if artifact_dir.exists():
        files = [f.name for f in artifact_dir.iterdir()]
        print(f"📦 Files found in artifacts: {files}")

if __name__ == "__main__":
    poc_ingest()
    poc_query("How to solve pawn overflow?")