"""
Cover letter generation module.

Generates tailored letters based on parsed resume data and target job details.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI

from resume_parser import ParsedResume


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create Gemini Flash LLM from environment configuration."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, google_api_key=api_key)


def _normalize_resume(parsed_resume: ParsedResume | Dict[str, Any]) -> ParsedResume:
    """Normalize incoming resume payload to ParsedResume."""
    if isinstance(parsed_resume, ParsedResume):
        return parsed_resume
    return ParsedResume.model_validate(parsed_resume)


def generate_cover_letter(
    parsed_resume: ParsedResume,
    selected_job: dict,
    tone: str = "professional",
) -> str:
    """
    Generate a personalized markdown-formatted cover letter.

    The output is tuned for modern Indian tech startup communication style.
    """
    resume_obj = _normalize_resume(parsed_resume)
    if not selected_job:
        raise ValueError("Selected job details are required to generate a cover letter.")

    llm = _get_llm()
    resume_json = json.dumps(resume_obj.model_dump(), indent=2, ensure_ascii=False)
    job_json = json.dumps(selected_job, indent=2, ensure_ascii=False)

    system_prompt = (
        "You are an expert career writing assistant.\n"
        "Write a 1-page personalized cover letter in markdown.\n"
        "Audience: Indian tech startups in Noida/Gurugram/Bangalore.\n"
        "Tone options: professional, enthusiastic, concise.\n"
        "Current requested tone: {tone}.\n\n"
        "Writing requirements:\n"
        "1) Keep it confident, modern, and natural (not generic).\n"
        "2) Use candidate's real resume details only.\n"
        "3) Map experience, projects, skills, and leadership to job requirements.\n"
        "4) Show enthusiasm for the company/role and clear value proposition.\n"
        "5) End with a strong call-to-action.\n"
        "6) Output only the letter content in markdown, no extra commentary.\n"
    )

    human_prompt = (
        "Candidate Parsed Resume JSON:\n{resume_json}\n\n"
        "Selected Job JSON:\n{job_json}\n\n"
        "Generate the final tailored cover letter now."
    )

    prompt = f"{system_prompt}\n\n{human_prompt}".format(
        tone=tone,
        resume_json=resume_json,
        job_json=job_json,
    )
    response = llm.invoke(prompt)
    letter = str(response.content).strip()
    if not letter:
        raise ValueError("LLM returned an empty cover letter. Please try again.")
    return letter
