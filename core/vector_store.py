import os
import chromadb
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_embedder = None


def get_embedder():
    """
    Ultra-lightweight cloud embeddings via Mistral AI.
    Uses 0 MB server RAM and 0 MB disk space.
    """
    global _embedder
    if _embedder is None:
        api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
        _embedder = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=api_key,
        )
    return _embedder


def build_vector_store(transcript: str) -> Chroma:
    """
    Creates an Ephemeral In-Memory Chroma vector store.
    Zero disk storage used — completely purged when session ends.
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