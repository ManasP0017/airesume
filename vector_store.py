"""
Vector store module for job postings.

Responsibilities:
- Load job JSON records from jobs_data/
- Embed with sentence-transformers (all-MiniLM-L6-v2)
- Persist embeddings in local Chroma DB
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from resume_parser import ParsedResume

JOBS_DATA_DIR = Path("jobs_data")
CHROMA_DB_DIR = Path("chroma_db")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "jobs"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Create embedding model instance used across the app."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def _ensure_sample_jobs_exist() -> None:
    """
    Seed realistic sample jobs if `jobs_data/` is empty.

    This keeps the app usable out-of-the-box until the user provides real data.
    """
    JOBS_DATA_DIR.mkdir(exist_ok=True)
    existing = list(JOBS_DATA_DIR.glob("*.json"))
    if existing:
        return

    today = date.today().isoformat()
    samples: List[Dict[str, Any]] = [
        {
            "id": "JOB-IND-2026-001",
            "title": "AI Engineer (LLM Apps)",
            "company": "NimbusAI Labs",
            "location": "Bangalore, India",
            "description": (
                "Build production LLM features: embeddings search, RAG chatbots, evaluation, and monitoring. "
                "Work with product and engineering to ship GenAI experiences."
            ),
            "requirements": [
                "3+ years Python backend experience",
                "Hands-on with embeddings, vector DBs (Chroma/Pinecone/FAISS)",
                "LangChain/LlamaIndex experience",
                "Comfort with Docker and basic cloud deployment",
            ],
            "skills": ["Python", "LangChain", "RAG", "Chroma", "Sentence Transformers", "Docker", "FastAPI"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-002",
            "title": "Full-Stack Developer (Python + UI)",
            "company": "Gurugram FinTech Studio",
            "location": "Gurugram, India",
            "description": (
                "Own feature delivery across backend services and lightweight UIs. "
                "You’ll build internal tools and customer-facing dashboards."
            ),
            "requirements": [
                "Strong Python foundations and API design",
                "Experience building UIs (Gradio/Streamlit acceptable)",
                "SQL basics and data modeling",
                "Good engineering hygiene (testing, code reviews)",
            ],
            "skills": ["Python", "APIs", "Gradio", "SQL", "Git", "Testing"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-003",
            "title": "GenAI Product Engineer",
            "company": "Noida SaaS Startup",
            "location": "Noida, India",
            "description": (
                "Prototype and productionize GenAI workflows: prompt engineering, tool use, RAG, "
                "and guardrails. Collaborate closely with product for iterative delivery."
            ),
            "requirements": [
                "Experience with Gemini/OpenAI/Groq models",
                "RAG architecture and retrieval tuning",
                "Ability to translate product needs into reliable systems",
            ],
            "skills": ["Python", "Gemini", "Groq", "Prompting", "RAG", "LangChain", "Chroma"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-004",
            "title": "Software Engineer (Backend - Python)",
            "company": "Bengaluru Logistics Tech",
            "location": "Bangalore, India",
            "description": (
                "Build scalable backend services, data pipelines, and integrations. "
                "Work with product to deliver reliable systems in a high-growth environment."
            ),
            "requirements": [
                "2+ years Python experience",
                "Experience with background jobs and REST APIs",
                "Familiarity with logging/monitoring",
            ],
            "skills": ["Python", "REST APIs", "PostgreSQL", "Background Jobs", "Observability"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-005",
            "title": "ML Engineer (NLP + Retrieval)",
            "company": "Bangalore AI Platform",
            "location": "Bangalore, India",
            "description": (
                "Improve retrieval quality, embedding selection, evaluation, and deployment. "
                "Implement offline/online eval and quality dashboards."
            ),
            "requirements": [
                "Experience with sentence-transformers and semantic search",
                "Understanding of ranking, evaluation metrics, and embeddings",
                "Ability to ship models into production systems",
            ],
            "skills": ["Python", "NLP", "Sentence Transformers", "Embeddings", "Evaluation", "Chroma"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-006",
            "title": "AI Solutions Engineer (Client-facing)",
            "company": "Delhi NCR AI Consultancy",
            "location": "Noida, India",
            "description": (
                "Deliver AI/GenAI solutions for clients: discovery, POCs, and production rollouts. "
                "Work on RAG, document processing, and chat assistants."
            ),
            "requirements": [
                "Strong communication and client handling",
                "Hands-on with Python and LLM integrations",
                "Document parsing and data cleaning experience",
            ],
            "skills": ["Python", "RAG", "Document Parsing", "Gemini", "LangChain", "Chroma"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-007",
            "title": "Platform Engineer (AI Tooling)",
            "company": "Bangalore DevTools Startup",
            "location": "Bangalore, India",
            "description": (
                "Build internal developer tooling for AI features: prompt/version management, "
                "eval harnesses, and observability pipelines."
            ),
            "requirements": [
                "Python engineering maturity",
                "Experience with CI/CD and containers",
                "Interest in AI platform/infra problems",
            ],
            "skills": ["Python", "Docker", "CI/CD", "Observability", "LLM Tooling"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-008",
            "title": "Full-Stack Engineer (Startup - Noida)",
            "company": "RapidKart",
            "location": "Noida, India",
            "description": (
                "Build customer features end-to-end. Backend in Python; simple UIs for internal tools. "
                "High ownership and rapid iteration."
            ),
            "requirements": [
                "Solid Python and system design basics",
                "Ability to ship quickly with good quality",
                "Comfort working across the stack",
            ],
            "skills": ["Python", "APIs", "Gradio", "SQL", "Testing", "Git"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-009",
            "title": "RAG Engineer (Search + LLM)",
            "company": "Gurugram Enterprise AI",
            "location": "Gurugram, India",
            "description": (
                "Design and optimize RAG systems: chunking, retrieval strategies, re-ranking, and grounding. "
                "Build evaluation loops for continuous improvement."
            ),
            "requirements": [
                "Experience with vector DBs and retrieval",
                "LangChain RAG pipelines",
                "Evaluation mindset and experimentation",
            ],
            "skills": ["Python", "RAG", "LangChain", "Chroma", "Embeddings", "Evaluation"],
            "posted_date": today,
        },
        {
            "id": "JOB-IND-2026-010",
            "title": "Software Engineer (GenAI Integrations)",
            "company": "Bangalore HRTech",
            "location": "Bangalore, India",
            "description": (
                "Integrate LLM features into HR workflows: resume parsing, candidate matching, "
                "cover letter generation, and chat assistants."
            ),
            "requirements": [
                "Python + API development",
                "LLM integrations and prompt discipline",
                "Text processing and data quality focus",
            ],
            "skills": ["Python", "Text Processing", "Gemini", "LangChain", "Embeddings", "Chroma"],
            "posted_date": today,
        },
    ]

    for job in samples:
        path = JOBS_DATA_DIR / f"{job['id']}.json"
        path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")


def load_job_documents() -> List[Document]:
    """Load job JSON files and convert them into LangChain Documents."""
    _ensure_sample_jobs_exist()

    docs: List[Document] = []
    for json_path in sorted(JOBS_DATA_DIR.glob("*.json")):
        try:
            job = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        job_id = str(job.get("id", json_path.stem))
        title = str(job.get("title", ""))
        company = str(job.get("company", ""))
        location = str(job.get("location", ""))
        description = str(job.get("description", ""))
        requirements = job.get("requirements", [])
        skills = job.get("skills", [])
        posted_date = str(job.get("posted_date", ""))

        requirements_text = "\n".join(f"- {r}" for r in requirements) if isinstance(requirements, list) else str(requirements)
        skills_text = ", ".join(skills) if isinstance(skills, list) else str(skills)

        content = (
            f"Job Title: {title}\n"
            f"Company: {company}\n"
            f"Location: {location}\n"
            f"Posted: {posted_date}\n\n"
            f"Description:\n{description}\n\n"
            f"Requirements:\n{requirements_text}\n\n"
            f"Skills:\n{skills_text}\n"
        ).strip()

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "id": job_id,
                    "title": title,
                    "company": company,
                    "location": location,
                    "posted_date": posted_date,
                    "skills": ", ".join(skills) if isinstance(skills, list) else str(skills),
                    "source_file": str(json_path.name),
                },
            )
        )

    return docs


def build_or_load_vector_store() -> Chroma:
    """
    Build (if needed) or load persistent Chroma vector store.

    - Seeds sample jobs if `jobs_data/` is empty
    - Loads job documents and persists embeddings into `./chroma_db/`
    """
    CHROMA_DB_DIR.mkdir(exist_ok=True)
    embeddings = get_embedding_model()
    vectordb = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DB_DIR))

    # If collection is empty, add documents.
    try:
        existing = vectordb.get(include=[])
        existing_count = len(existing.get("ids", []))
    except Exception:
        existing_count = 0

    if existing_count == 0:
        docs = load_job_documents()
        if docs:
            vectordb.add_documents(docs)

    return vectordb


def _resume_to_query(resume: ParsedResume) -> str:
    """Build a compact semantic query string from ParsedResume."""
    parts: List[str] = []
    if resume.summary:
        parts.append(f"Summary: {resume.summary}")
    if resume.skills:
        parts.append("Skills: " + ", ".join(resume.skills[:40]))
    if resume.experience:
        exp_lines = []
        for e in resume.experience[:5]:
            bullets = " ".join(e.bullets[:3]) if e.bullets else ""
            exp_lines.append(f"{e.title} at {e.company}. {bullets}".strip())
        if exp_lines:
            parts.append("Experience: " + " | ".join(exp_lines))
    if resume.projects:
        proj_lines = []
        for p in resume.projects[:4]:
            proj_lines.append(f"{p.title} ({', '.join(p.technologies[:8])})")
        if proj_lines:
            parts.append("Projects: " + " | ".join(proj_lines))
    return "\n".join(parts).strip() or resume.name or "Resume"


def _why_you_fit(resume: ParsedResume, job_meta: Dict[str, Any]) -> str:
    """Generate a basic overlap-based fit explanation (non-LLM)."""
    resume_skills = {s.strip().lower() for s in (resume.skills or []) if s and s.strip()}
    job_skills = job_meta.get("skills") or []
    job_skills_set = {str(s).strip().lower() for s in job_skills if str(s).strip()}
    overlap = sorted(resume_skills.intersection(job_skills_set))

    if overlap:
        shown = ", ".join(overlap[:8])
        return f"Skill overlap: {shown}."
    if resume_skills:
        return "Your resume shows relevant transferable skills; consider tailoring keywords to this posting."
    return "Add a skills section to improve matching confidence."


def semantic_match_resume(parsed_resume: ParsedResume, k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top-k semantic job matches.

    Uses Chroma relevance scores (0..1) and reports match_percentage as score*100.
    """
    if parsed_resume is None:
        raise ValueError("Parsed resume is empty.")

    vectordb = build_or_load_vector_store()
    query = _resume_to_query(parsed_resume)

    results: List[Dict[str, Any]] = []
    for doc, score in vectordb.similarity_search_with_relevance_scores(query, k=k):
        # score is relevance (higher is better), typically in [0, 1]
        pct = int(round(max(0.0, min(1.0, float(score))) * 100))
        meta = doc.metadata or {}
        results.append(
            {
                "id": meta.get("id", ""),
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "posted_date": meta.get("posted_date", ""),
                "match_percentage": pct,
                "why_you_fit": _why_you_fit(parsed_resume, meta),
            }
        )

    # Ensure deterministic ordering by match % desc.
    results.sort(key=lambda x: x.get("match_percentage", 0), reverse=True)
    return results
