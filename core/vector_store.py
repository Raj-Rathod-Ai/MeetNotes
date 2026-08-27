import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embedder = None


def get_embedder():
    """Singleton instance for embedding model to save RAM on Render."""
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embedder


def build_vector_store(transcript: str) -> Chroma:
    """
    Creates an Ephemeral (In-Memory) Chroma vector store.
    Zero disk storage used — completely purged when the session ends or user leaves.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_text(transcript)
    documents = [
        Document(page_content=chunk, metadata={"index": i})
        for i, chunk in enumerate(text_chunks)
    ]

    # In-memory ephemeral client (0 bytes written to disk!)
    ephemeral_client = chromadb.EphemeralClient()

    return Chroma.from_documents(
        documents=documents,
        embedding=get_embedder(),
        client=ephemeral_client,
    )


def get_retriever(store: Chroma, k: int = 4):
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )