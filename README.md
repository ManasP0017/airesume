# AI Resume & Job Matcher

Production-ready AI web app to parse a resume, match jobs semantically, answer fit questions with RAG, generate tailored cover letters, and optimize resume bullets for target roles.

## Why This Project

Job seekers usually do these tasks separately. This app brings everything into one workflow:
- Parse resume PDF into structured JSON
- Match against jobs using embeddings + Chroma
- Explain fit gaps through a RAG chatbot
- Generate role-specific cover letters
- Rewrite bullets to be more ATS-friendly and impact-focused

## Tech Stack

- Python 3.11+
- Gradio (multi-tab app UI)
- LangChain + LangChain-Community (modern runnable syntax)
- Gemini Flash (`gemini-1.5-flash`) via `langchain-google-genai`
- Hugging Face sentence-transformers (`all-MiniLM-L6-v2`)
- Chroma vector database (persistent local store)
- PyMuPDF (`fitz`) for PDF parsing
- Pydantic for strict structured outputs
- dotenv for secure environment variables

## Features

1. **Resume Parser**  
   Upload PDF and convert to structured resume schema.

2. **Semantic Job Matching**  
   Top-k relevant jobs with match percentage and fit explanation.

3. **RAG Chatbot**  
   Ask job-fit, skill-gap, prep-plan, and salary-expectation questions.

4. **Tailored Cover Letter Generator**  
   Choose a matched job and generate a personalized markdown letter with download options.

5. **Resume Optimizer**  
   Side-by-side original vs improved bullets for experience, projects, and leadership roles.

## Project Structure

```text
ai-resume-job-matcher/
├── app.py
├── resume_parser.py
├── vector_store.py
├── rag_chain.py
├── cover_letter.py
├── resume_optimizer.py
├── requirements.txt
├── .env
├── jobs_data/
├── chroma_db/
└── README.md
```

## Run Locally

### 1) Clone + setup venv

```bash
python -m venv .venv
```

PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

Create `.env`:

```env
# Preferred
GEMINI_API_KEY=your_gemini_api_key_here

# Optional fallback (if you later wire Groq paths)
# GROQ_API_KEY=your_groq_api_key_here
```

### 4) Launch app

```bash
python app.py
```

Open the Gradio URL shown in terminal (usually `http://127.0.0.1:7860`).

## How to Get Gemini API Key

1. Open [Google AI Studio](https://aistudio.google.com/).
2. Sign in and create an API key.
3. Paste it into `.env` as `GEMINI_API_KEY=...`.
4. Restart the app after updating `.env`.

## Deploy on Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Upload project files (`app.py`, modules, `requirements.txt`, etc.).
3. In Space settings, add secret:
   - `GEMINI_API_KEY`
4. Ensure `app.py` is entrypoint.
5. Commit and restart the Space.

Recommended Space hardware: CPU basic is enough for small usage; upgrade for faster embedding/model operations.

## Screenshot Placeholders

- `docs/screenshots/home.png` - Home tab
- `docs/screenshots/matches.png` - Top matches
- `docs/screenshots/chatbot.png` - RAG chatbot
- `docs/screenshots/cover-letter.png` - Cover letter output
- `docs/screenshots/optimizer.png` - Resume optimization side-by-side

## Notes

- `jobs_data/` auto-seeds realistic sample jobs if empty.
- `chroma_db/` is created automatically on first run.
- For best parsing quality, use text-based PDFs (not scanned image PDFs).
