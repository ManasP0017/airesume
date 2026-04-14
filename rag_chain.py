"""
RAG and matching chains.

Contains:
- Semantic resume-to-job matching
- Role-fit chatbot response generation
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from resume_parser import ParsedResume
from vector_store import _resume_to_query
from vector_store import build_or_load_vector_store


def get_retriever(search_kwargs: Dict[str, Any] | None = None):
    """Return a retriever backed by the persistent job Chroma vectorstore."""
    vectordb = build_or_load_vector_store()
    return vectordb.as_retriever(search_kwargs=search_kwargs or {"k": 6})


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create Gemini Flash chat model from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=api_key)


def _resume_context(parsed_resume: ParsedResume) -> str:
    """Convert parsed resume object into compact JSON context string."""
    return json.dumps(parsed_resume.model_dump(), indent=2, ensure_ascii=False)


def get_rag_chain(parsed_resume: ParsedResume):
    """
    Build a resume-aware retrieval QA chain.

    Returns a runnable chain that supports:
    - .invoke({"input": user_question})
    """
    if parsed_resume is None:
        raise ValueError("Parsed resume is required to initialize the RAG chain.")

    retriever = get_retriever(search_kwargs={"k": 6})
    llm = _get_llm()
    resume_ctx = _resume_context(parsed_resume)

    system_prompt = (
        "You are a professional career and job-fit assistant.\n"
        "You must be helpful, honest, and concise.\n"
        "Use only the provided resume context and retrieved job context.\n"
        "Do not invent missing facts.\n"
        "You can answer about: job fit, strengths, missing skills, preparation plan, interview readiness, salary expectations, "
        "and role prioritization.\n"
        "If the question is unrelated to career/job-fit topics, politely redirect to career-related questions.\n\n"
        "Candidate Resume (authoritative):\n{{resume_ctx}}\n\n"
        "Retrieved Job Context:\n{{context}}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{{input}}"),
        ],
        template_format="jinja2"
    )

    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # Add resume context to input before chain execution
    rag_chain_with_resume = (
        RunnablePassthrough.assign(resume_ctx=lambda x: resume_ctx)
        | rag_chain
    )

    return rag_chain_with_resume.with_config(run_name="resume_job_fit_rag")


def find_top_job_matches(
    parsed_resume: Dict[str, Any],
    vectordb: Chroma,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return top job matches from an existing vector store."""
    resume_obj = ParsedResume.model_validate(parsed_resume)
    query = _resume_to_query(resume_obj)
    rows: List[Dict[str, Any]] = []
    for doc, score in vectordb.similarity_search_with_relevance_scores(query, k=top_k):
        meta = doc.metadata or {}
        rows.append(
            {
                "id": meta.get("id", ""),
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "match_percentage": int(round(max(0.0, min(1.0, float(score))) * 100)),
                "why_you_fit": "Matched through semantic overlap with your resume profile.",
            }
        )
    return rows


def ask_fit_chatbot(
    question: str,
    parsed_resume: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> str:
    """Answer role-fit questions using resume context + retrieved job context."""
    _ = chat_history
    resume_obj = ParsedResume.model_validate(parsed_resume)
    chain = get_rag_chain(resume_obj)
    result = chain.invoke({"input": question})
    return str(result.get("answer", "")).strip() or "I could not generate an answer for that question."
