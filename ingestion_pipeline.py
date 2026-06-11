import os
import sys

# Windows console defaults to cp1252, which can't print some characters (e.g. BOM).
# Force UTF-8 output so prints don't crash on non-cp1252 text.
sys.stdout.reconfigure(encoding="utf-8")

# read text files/ppt from documents
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# for chunking
from langchain_text_splitters import CharacterTextSplitter
# embedding model
from langchain_huggingface import HuggingFaceEmbeddings
# vector DB (chroma)
from langchain_chroma import Chroma
# .env
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="doc"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    #check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")
    
    #Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path = docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8-sig"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" content length: {len(doc.page_content)} characters")
        print(f" content preview: {doc.page_content[:100]} ...")
        print(f" metadata: {doc.metadata}")

    return documents
        
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks)-5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB")

    # Local HuggingFace embedding model — runs on your machine, no API key/quota,
    # and your documents never leave the machine.
    # Stronger local embedding model (768-dim) - much better at semantic matching
    # than MiniLM. Documents are embedded as-is (no query instruction needed here).
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    #Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Main Function")
    
    #1. load the files
    documents = load_documents(docs_path="docs")

    #2. chunking the files
    chunks = split_documents(documents)

    #3. Embedding and storing in vector DB
    vectorDB = create_vector_store(chunks)

if __name__ == "__main__":
    main()