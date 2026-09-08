import os
from fastmcp import FastMCP
import chromadb
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()

PERSISTENT_DIR = "./chroma_db"
COLLECTION_NAME = "rag_mcp"
DATA_DIR = "./data"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY","")

mcp = FastMCP("RAG Server")

def init_chroma():
  client = chromadb.PersistentClient(path=PERSISTENT_DIR)
  collection = client.get_or_create_collection(name=COLLECTION_NAME)
  return collection
init_chroma()

def get_chroma_client():
   return chromadb.PersistentClient(path=PERSISTENT_DIR)

@mcp.tool
def ingest_data_dir():
    """
    Initialize the ChromaDb client and collection so that the user can query that later.
    """
    chroma_client = get_chroma_client()
    chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="text")

    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(DATA_DIR, file_extractor=file_extractor).load_data()

    for doc in documents:
        collection.add(
            documents=[doc.text],
            metadatas=[doc.metadata],
            ids=[doc.doc_id],
        )

    final_count = collection.count()
    return f"Ingested {final_count} documents into the Chroma collection '{COLLECTION_NAME}'."

@mcp.tool
def query_document(query: str, n_results: int = 2) -> str:
    """
    Query the ChromaDb collection for documents relevant to the provided query.
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results['documents']:
        return "No relevant documents found."

    response = "Relevant documents:\n"
    for i, doc in enumerate(results['documents'][0]):
        response += f"{i + 1}. {doc}\n"

    return response

@mcp.tool
def get_db_status() -> str:
    """
    Get the status of the ChromaDb collection.
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    count = collection.count()
    return f"The Chroma collection '{COLLECTION_NAME}' contains {count} documents."

if __name__ == "__main__":
    init_chroma()
    mcp.run()