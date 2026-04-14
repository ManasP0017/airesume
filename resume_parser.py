import os
import fitz  # PyMuPDF
import re
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI


# ====================== Pydantic Schema ======================
class Contact(BaseModel):
    phone: str = Field(default="")
    email: str = Field(default="")
    linkedin: Optional[str] = Field(default=None)
    location: str = Field(default="")


class Education(BaseModel):
    institution: str = Field(default="")
    degree: str = Field(default="")
    dates: str = Field(default="")
    cgpa: Optional[str] = Field(default=None)


class Experience(BaseModel):
    title: str = Field(default="")
    company: str = Field(default="")
    dates: str = Field(default="")
    bullets: List[str] = Field(default_factory=list)


class Project(BaseModel):
    title: str = Field(default="")
    dates: str = Field(default="")
    description: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)


class Certification(BaseModel):
    name: str = Field(default="")
    issuer: str = Field(default="")
    dates: str = Field(default="")


class Position(BaseModel):
    title: str = Field(default="")
    organization: str = Field(default="")
    dates: str = Field(default="")
    bullets: List[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    name: str = Field(default="")
    contact: Contact = Field(default_factory=Contact)
    summary: Optional[str] = Field(default=None)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    positions_of_responsibility: List[Position] = Field(default_factory=list)


# ====================== Text Extraction & Aggressive Cleaning ======================
def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    full_text = "\n".join(text_parts)
    return full_text


def _clean_text(text: str) -> str:
    """Aggressively clean extracted text for reliable parsing."""
    # Remove all replacement characters and other problematic unicode
    text = re.sub(r"[\uFFFD\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008\u000B\u000E\u000F\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u001A\u001B\u001C\u001D\u001E\u001F]", "", text)
    
    # Replace en-dash and em-dash with regular hyphen
    text = re.sub(r"[\u2013\u2014\u2015]", "-", text)
    
    # Replace smart quotes with regular quotes
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    
    # Fix broken lines: join lines that don't end with punctuation
    # This handles cases where words are split across lines in PDFs
    lines = text.split("\n")
    fixed_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line and i > 0:
            prev_line = fixed_lines[-1] if fixed_lines else ""
            # If previous line doesn't end with sentence-ending punctuation, join with current line
            if prev_line and not re.search(r"[.!?]\s*$", prev_line):
                # Check if current line starts with lowercase (likely continuation)
                if line[0].islower() or not re.search(r"[A-Z]$", prev_line):
                    fixed_lines[-1] = prev_line + " " + line
                    continue
        if line:
            fixed_lines.append(line)
    
    text = "\n".join(fixed_lines)
    
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    # Remove common PDF artifacts
    text = re.sub(r"\f", "\n\n", text)  # Form feeds
    text = re.sub(r"\x0c", "\n\n", text)
    
    # Remove page numbers (common pattern: standalone numbers)
    text = re.sub(r"\n\s*\d+\s*\n", "\n\n", text)
    
    # Remove email/phone artifacts that sometimes appear as headers/footers
    text = re.sub(r"\n[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\s*\n", "\n", text, flags=re.IGNORECASE)
    
    return text.strip()


# ====================== LLM-Based Parsing ======================
def _get_llm() -> ChatGoogleGenerativeAI:
    """Create Gemini 2.5 Flash LLM for structured output."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=api_key)


def parse_resume(pdf_bytes: bytes) -> ParsedResume:
    """
    Parse ANY PDF resume into structured data using Gemini 2.5 Flash.
    
    Process:
    1. Extract text from PDF using PyMuPDF
    2. Aggressively clean and normalize the text
    3. Use Gemini with structured output to parse into ParsedResume schema
    4. Return fallback if parsing fails
    
    Returns:
        ParsedResume object with extracted data or minimal fallback
    """
    try:
        # Extract and clean text
        raw_text = _extract_text_from_pdf(pdf_bytes)
        cleaned_text = _clean_text(raw_text)
        
        if not cleaned_text or len(cleaned_text) < 50:
            # Return minimal fallback if text extraction failed
            return ParsedResume(
                name="Unknown",
                contact=Contact(),
                summary=None,
                education=[],
                experience=[],
                projects=[],
                skills=[],
                certifications=[],
                positions_of_responsibility=[]
            )
        
        # Use Gemini with structured output
        llm = _get_llm()
        structured_llm = llm.with_structured_output(ParsedResume)
        
        system_prompt = """You are an expert resume parser with deep experience extracting structured data from resumes in various formats (chronological, functional, hybrid, creative, academic, etc.).

Your task is to extract ALL available information from the resume text and return it in the exact JSON schema provided.

EXTRACTION RULES (STRICT):

1. NAME: Extract the candidate's full name from the header or top of the resume.

2. CONTACT INFORMATION:
   - phone: Extract phone number (any format, include country code if present)
   - email: Extract email address
   - linkedin: Extract LinkedIn profile URL if present (optional)
   - location: Extract city, state/country, or full location string

3. SUMMARY/OBJECTIVE:
   - Extract any professional summary, objective, or profile section
   - If none exists, leave as null

4. EDUCATION (extract each entry):
   - institution: University/College/School name
   - degree: Degree type and major (e.g., "Bachelor of Technology in Computer Science")
   - dates: Date range (e.g., "Sep 2022 - Present", "2018 - 2022")
   - cgpa: GPA/grade if mentioned (optional)

5. WORK EXPERIENCE (extract each role):
   - title: Job title/position
   - company: Company/organization name
   - dates: Employment period
   - bullets: Extract ALL bullet points as individual list items. Preserve the exact wording as much as possible.

6. PROJECTS (extract each project):
   - title: Project name
   - dates: Timeline if provided
   - description: Extract project description as list items (bullets or sentences)
   - technologies: Extract all technologies/tools mentioned as a list

7. SKILLS:
   - Extract ALL skills mentioned in the resume
   - Include technical skills, languages, tools, frameworks, soft skills
   - Return as a flat list of individual skill strings

8. CERTIFICATIONS (extract each):
   - name: Certification title
   - issuer: Issuing organization
   - dates: Date or validity period if provided

9. POSITIONS OF RESPONSIBILITY (extract each):
   - title: Position/role title
   - organization: Organization name
   - dates: Time period if provided
   - bullets: Extract responsibilities/achievements as list items

IMPORTANT HANDLING INSTRUCTIONS:
- If a section is completely absent, return an empty list for that field
- If individual fields within an entry are missing, use empty strings
- Preserve dates exactly as written in the resume
- Handle bullet points that may be numbered (1., 2., etc.) or bulleted (•, -, *)
- Be thorough - extract EVERYTHING present in the resume
- Do NOT invent or hallucinate information that isn't in the text
- Handle various formatting styles gracefully

Return ONLY the structured JSON matching the schema exactly. No additional commentary."""
        
        prompt = f"""RESUME TEXT TO PARSE:

{cleaned_text}

Parse this resume into the structured schema. Extract ALL available information."""
        
        parsed_resume = structured_llm.invoke(prompt)
        return parsed_resume
        
    except Exception as exc:
        # Fallback: return minimal resume with error info in name
        print(f"Resume parsing error: {exc}")
        return ParsedResume(
            name="Parsing Error",
            contact=Contact(),
            summary="Resume parsing encountered an error. Please try again with a different PDF format.",
            education=[],
            experience=[],
            projects=[],
            skills=[],
            certifications=[],
            positions_of_responsibility=[]
        )