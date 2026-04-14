"""
Resume optimization module.

Suggests stronger, impact-focused bullet point rewrites for experience entries.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from resume_parser import ParsedResume


class BulletGroup(BaseModel):
    title: str = Field(default="")
    company_or_org: str = Field(default="")
    original_bullets: List[str] = Field(default_factory=list)
    optimized_bullets: List[str] = Field(default_factory=list)


class OptimizerOutput(BaseModel):
    optimized_experience: List[Dict[str, Any]] = Field(default_factory=list)
    optimized_projects: List[Dict[str, Any]] = Field(default_factory=list)
    optimized_positions: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions_summary: str = Field(default="")


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)


def _normalize_resume(parsed_resume: ParsedResume | Dict[str, Any]) -> ParsedResume:
    if isinstance(parsed_resume, ParsedResume):
        return parsed_resume
    return ParsedResume.model_validate(parsed_resume)


def optimize_resume_bullets(parsed_resume: ParsedResume, selected_job: dict) -> Dict[str, Any]:
    """
    Optimize resume bullets for a selected job.

    Returns:
    {
      "optimized_experience": List[dict],
      "optimized_projects": List[dict],
      "optimized_positions": List[dict],
      "suggestions_summary": str
    }
    """
    if not selected_job:
        raise ValueError("Selected job details are required for optimization.")

    resume_obj = _normalize_resume(parsed_resume)
    llm = _get_llm()

    class StructuredOptimizerOutput(BaseModel):
        optimized_experience: List[Dict[str, Any]] = Field(default_factory=list)
        optimized_projects: List[Dict[str, Any]] = Field(default_factory=list)
        optimized_positions: List[Dict[str, Any]] = Field(default_factory=list)
        suggestions_summary: str = Field(default="")

    structured_llm = llm.with_structured_output(StructuredOptimizerOutput)

    prompt = (
        "You are an expert ATS resume optimizer for Indian tech/startup hiring.\n"
        "Rewrite resume bullets to be sharper, achievement-oriented, and keyword-aligned to the target role.\n"
        "Rules:\n"
        "1) Keep original meaning but strengthen impact.\n"
        "2) Use strong action verbs and concise phrasing.\n"
        "3) Quantify outcomes where feasible (reasonable estimates allowed only if phrased carefully).\n"
        "4) Align bullets with the target job requirements and stack.\n"
        "5) Focus especially on these profile strengths when present:\n"
        "   - AI internship\n"
        "   - Semantic Book Recommender project\n"
        "   - Visual Bloggers' Club leadership\n"
        "   - Full-stack project work\n"
        "6) Return all sections, even if empty.\n\n"
        f"Parsed Resume JSON:\n{json.dumps(resume_obj.model_dump(), indent=2, ensure_ascii=False)}\n\n"
        f"Selected Job JSON:\n{json.dumps(selected_job, indent=2, ensure_ascii=False)}"
    )

    result = structured_llm.invoke(prompt)
    payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return OptimizerOutput.model_validate(payload).model_dump()
