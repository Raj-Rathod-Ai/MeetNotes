import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "open-mistral-7b")


def get_llm():
    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=api_key,
        temperature=0.1,
        max_retries=5,
    )


def extract_section(prompt_instruction: str, transcript: str, retries: int = 3) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_instruction),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    content = transcript[:15000]

    for attempt in range(retries):
        try:
            return chain.invoke({"text": content})
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait_time = (attempt + 1) * 3
                time.sleep(wait_time)
            else:
                raise e


def extract_action_items(transcript: str) -> str:
    instruction = (
        "Analyze the transcript and list all concrete action items. For every item, provide:\n"
        "• Task: [description]\n"
        "• Owner: [person responsible, or 'Unassigned']\n"
        "• Deadline: [timeframe/date, or 'Not specified']\n\n"
        "Format as a clean numbered list. If none are found, return 'No explicit action items identified.'"
    )
    return extract_section(instruction, transcript)


def extract_key_decisions(transcript: str) -> str:
    time.sleep(0.5)
    instruction = (
        "Identify and list all key decisions, alignments, or agreements made during the discussion. "
        "Format as bullet points. If none were reached, return 'No definitive decisions recorded.'"
    )
    return extract_section(instruction, transcript)


def extract_questions(transcript: str) -> str:
    time.sleep(0.5)
    instruction = (
        "Extract all unresolved questions, blockers, or open items requiring future follow-up. "
        "Format as bullet points. If all questions were resolved, return 'No unresolved questions remaining.'"
    )
    return extract_section(instruction, transcript)