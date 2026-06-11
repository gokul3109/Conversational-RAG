from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persist_directory = "db/chroma_db"

# Load embeddings and vector store.
# Must match the model used in ingestion. bge-base-en-v1.5 (v1.5) performs well
# on short queries without a special instruction prefix.
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# loading the db
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


def retrieve(query, k=5):
    """Return the top-k most relevant chunks for a query."""
    retriever = db.as_retriever(search_kwargs={"k": k})  # get top k similar chunks
    return retriever.invoke(query)


if __name__ == "__main__":
    # Standalone test: run this file directly to inspect what retrieval returns.
    query = "Which island does SpaceX lease for its launches in the pacific?"
    relevant_docs = retrieve(query)

    print(f"User query: {query}")
    print("--- Context --")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
