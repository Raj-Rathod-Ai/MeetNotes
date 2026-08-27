import os
import time
from typing import List
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "open-mistral-7b")


def get_llm():
    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=api_key,
        temperature=0.3,
        max_retries=5,
    )


def safe_invoke(chain, input_data: dict, retries: int = 3):
    """Invoke chain with exponential backoff if rate limited."""
    for attempt in range(retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait_time = (attempt + 1) * 3
                time.sleep(wait_time)
            else:
                raise e


def split_text(text: str, chunk_size: int = 15000) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=500,
    )
    return splitter.split_text(text)


def summarize(transcript: str) -> str:
    llm = get_llm()
    chunks = split_text(transcript, chunk_size=15000)

    if len(chunks) <= 1:
        single_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an executive assistant. Generate a structured, professional meeting summary in clear bullet points. Include key themes, discussion points, and outcomes.",
            ),
            ("human", "{text}"),
        ])
        chain = single_prompt | llm | StrOutputParser()
        target_text = chunks[0] if chunks else transcript
        return safe_invoke(chain, {"text": target_text})

    # For multi-chunk transcripts: summarize first 3 chunks with a safe pause
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely in bullet points:"),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()
    
    partial_summaries = []
    for c in chunks[:4]:
        partial = safe_invoke(map_chain, {"text": c})
        partial_summaries.append(partial)
        time.sleep(0.5)

    reduce_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Synthesize these segment summaries into a single cohesive executive summary with bullet points:",
        ),
        ("human", "{text}"),
    ])
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    return safe_invoke(reduce_chain, {"text": "\n\n".join(partial_summaries)})


def generate_title(transcript: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a concise, professional title for this meeting or video (max 7 words). Return ONLY the title text, with no extra punctuation or quotes.",
        ),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return safe_invoke(chain, {"text": transcript[:2500]}).strip(' "\'')
