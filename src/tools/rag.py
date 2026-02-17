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

def ingest_writeups(directory: str):
    """index your markdown files."""
    path = Path(directory)
    for file in path.rglob("*.md"):
        # Extract metadata from the folder path
        event_name = file.parts[-3]  # e.g., 'event_A'
        category = file.parts[-2]    # e.g., 'web'

        content = file.read_text(encoding="utf-8")

        collection.add(
            documents=[content],
            metadatas=[{
                "event": event_name,
                "category": category,
                "source": file.name,
                "source_path": str(file.relative_to("data/knowledge_base/writeups")),
            }],
            ids=[f"{event_name}_{file.name}"]
        )

# Updated ingest_writeups logic
def ingest_all(directory: str):
    path = Path(directory)
    for file in path.rglob("*.md"):
        # Categorize by the immediate parent folder name
        category = file.parent.name
        # Identify the source (e.g., 'hacktricks')
        source = file.parts[-2]

        content = file.read_text(encoding="utf-8")
        collection.add(
            documents=[content],
            metadatas=[{
                "category": category,
                "source": source
                 }],
            ids=[f"{source}_{file.name}"]
        )

def search_knowledge(query: str, n_results: int = 2):
    results = collection.query(query_texts=[query], n_results=n_results)

    formatted_results = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        # Extract the relative path from metadata to help the agent find artifacts
        rel_path = metadata.get("source_path", "Unknown")
        formatted_results.append(f"SOURCE PATH: {rel_path}\nCONTENT: {doc}")

    return "\n\n---\n\n".join(formatted_results)

def search_knowledge(query: str, n_results: int = 2):
    """Tool for the agent to look up past solutions."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return "\n\n".join(results['documents'][0])