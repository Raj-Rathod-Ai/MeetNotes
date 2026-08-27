import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, get_retriever

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "open-mistral-7b")


def get_llm(temperature: float = 0.2):
    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=api_key,
        temperature=temperature,
        max_retries=5,
    )


def format_context(documents) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def build_rag_chain(transcript: str):
    store = build_vector_store(transcript)
    retriever = get_retriever(store, k=4)
    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a precise meeting assistant. Answer the question using ONLY the provided meeting context.
If the answer is not present in the context, respond with:
"I could not find this information in the meeting transcript."

Be direct, clear, and cite speakers or numbers when available.

Meeting Context:
{context}""",
        ),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def ask_question(chain, query: str) -> str:
    return chain.invoke(query.strip())