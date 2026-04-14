"""
Main Gradio application for AI Resume & Job Matcher.

Tabs:
1) Upload Resume
2) Job Matches
3) RAG Chatbot
4) Cover Letter Generator
5) Resume Optimizer
"""

from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from dotenv import load_dotenv

from cover_letter import generate_cover_letter
from rag_chain import get_rag_chain
from resume_optimizer import optimize_resume_bullets
from resume_parser import ParsedResume, parse_resume
from vector_store import semantic_match_resume


load_dotenv()


APP_TITLE = "AI Resume & Job Matcher + Smart Cover Letter Generator"
APP_DESCRIPTION = (
    "Upload your resume, discover top job matches, chat about role fit, "
    "generate tailored cover letters, and improve your resume bullets."
)


def _normalize_resume_state(state_obj: Any) -> Dict[str, Any]:
    """Return resume state as a normalized dictionary."""
    if isinstance(state_obj, ParsedResume):
        return state_obj.model_dump()
    if isinstance(state_obj, dict):
        return state_obj
    return {}


def handle_resume_upload(pdf_file: Any) -> Tuple[Dict[str, Any], ParsedResume, str]:
    """Parse uploaded PDF resume bytes and return JSON preview + state + status."""
    if pdf_file is None:
        raise gr.Error("Please upload a PDF resume first.")

    try:
        with open(pdf_file.name, "rb") as f:
            pdf_bytes = f.read()
        parsed_obj = parse_resume(pdf_bytes)
    except Exception as exc:
        raise gr.Error(f"Resume parsing failed: {exc}") from exc

    parsed_json = parsed_obj.model_dump()
    success_msg = "Resume parsed successfully."
    return parsed_json, parsed_obj, success_msg


def handle_find_matches(parsed_resume: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """Find top matching jobs using semantic retrieval + LLM explanation."""
    parsed_resume_dict = _normalize_resume_state(parsed_resume)
    if not parsed_resume_dict:
        raise gr.Error("Please upload and parse your resume before matching.")

    try:
        resume_obj = parsed_resume if isinstance(parsed_resume, ParsedResume) else ParsedResume.model_validate(parsed_resume_dict)
    except Exception as exc:
        raise gr.Error(f"Invalid resume state: {exc}") from exc

    return semantic_match_resume(resume_obj, k=top_k)


def _badge(pct: int) -> str:
    color = "#16a34a" if pct > 70 else "#eab308" if pct >= 50 else "#dc2626"
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{color};color:white;font-weight:600;font-size:12px;">{pct}%</span>'


def format_matches_markdown(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "No matches found yet. Parse your resume first, then refresh matches."

    blocks: List[str] = []
    for m in matches:
        pct = int(m.get("match_percentage", 0) or 0)
        title = str(m.get("title", "") or "")
        company = str(m.get("company", "") or "")
        location = str(m.get("location", "") or "")
        why = str(m.get("why_you_fit", "") or "")
        blocks.append(
            "\n".join(
                [
                    f"### {title}",
                    f"**{company}** — {location}",
                    f"{_badge(pct)}",
                    "",
                    f"**Why you fit**: {why}",
                    "---",
                ]
            )
        )
    return "\n\n".join(blocks)


def refresh_matches(resume_obj: Any, k: int) -> Tuple[List[Dict[str, Any]], str]:
    """Compute matches and return (raw_matches, rendered_markdown)."""
    matches = handle_find_matches(resume_obj, top_k=int(k))
    return matches, format_matches_markdown(matches)


def _job_option_label(job: Dict[str, Any]) -> str:
    job_id = str(job.get("id", "JOB"))
    title = str(job.get("title", "Role"))
    company = str(job.get("company", "Company"))
    location = str(job.get("location", "Location"))
    return f"{job_id} | {title} - {company} ({location})"


def _job_dropdown_choices(matches: List[Dict[str, Any]]) -> List[str]:
    return [_job_option_label(job) for job in (matches or [])][:5]


def _find_selected_job(matches: List[Dict[str, Any]], selected_label: str) -> Dict[str, Any]:
    if not selected_label:
        return {}
    selected_id = selected_label.split(" | ", 1)[0].strip()
    for job in matches or []:
        if str(job.get("id", "")).strip() == selected_id:
            return job
    return {}


def _write_cover_letter_txt(letter_markdown: str) -> str:
    """Write generated letter to a temporary .txt file for download."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(letter_markdown)
        return tmp.name


def _write_cover_letter_docx(letter_markdown: str) -> str:
    """Write generated letter to a temporary .docx file for download."""
    try:
        from docx import Document
    except Exception as exc:
        raise ValueError("DOCX export requires python-docx. Install it with: pip install python-docx") from exc

    doc = Document()
    for paragraph in letter_markdown.split("\n\n"):
        clean = paragraph.replace("**", "").replace("#", "").strip()
        if clean:
            doc.add_paragraph(clean)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        doc.save(tmp.name)
        return tmp.name


def handle_cover_letter(
    parsed_resume: Dict[str, Any],
    matches: List[Dict[str, Any]],
    selected_job_label: str,
    tone: str,
    download_format: str,
) -> Tuple[str, str, str, str]:
    """Generate cover letter markdown + copy text + downloadable txt + status."""
    parsed_resume_dict = _normalize_resume_state(parsed_resume)
    if not parsed_resume_dict:
        raise gr.Error("Please upload and parse your resume first.")
    if not matches:
        raise gr.Error("No matched jobs found. Please refresh matches first.")
    if not selected_job_label:
        raise gr.Error("Please select a job from top matches.")

    selected_job = _find_selected_job(matches, selected_job_label)
    if not selected_job:
        raise gr.Error("Selected job was not found. Please reselect and try again.")

    try:
        resume_obj = parsed_resume if isinstance(parsed_resume, ParsedResume) else ParsedResume.model_validate(parsed_resume_dict)
        letter_md = generate_cover_letter(parsed_resume=resume_obj, selected_job=selected_job, tone=tone)
        file_path = _write_cover_letter_docx(letter_md) if download_format == "DOCX (.docx)" else _write_cover_letter_txt(letter_md)
        return letter_md, letter_md, file_path, "Cover letter generated successfully."
    except Exception as exc:
        raise gr.Error(f"Cover letter generation failed: {exc}") from exc


def copy_hint_text(letter_text: str) -> str:
    """Return copy helper status text."""
    if not letter_text.strip():
        return "No letter content available to copy yet."
    return "Copy-ready text is available below. Use the copy icon in the text box."


def _copy_status_text(text: str) -> str:
    if not text.strip():
        return "Nothing to copy yet."
    return "Use the copy icon in this box to copy to clipboard."


def _section_to_markdown(items: List[Dict[str, Any]], section_name: str, improved: bool) -> str:
    """Render original or improved bullets in markdown."""
    if not items:
        return f"### {section_name}\nNo content available."
    lines: List[str] = [f"### {section_name}"]
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title", "") or "")
        org = str(item.get("company", "") or item.get("organization", "") or "")
        key = "optimized_bullets" if improved else "original_bullets"
        bullets = item.get(key, []) or []
        lines.append(f"**{idx}. {title} - {org}**")
        if bullets:
            for b in bullets:
                lines.append(f"- {b}")
        else:
            lines.append("- No bullets.")
        lines.append("")
    return "\n".join(lines).strip()


def _collect_original_sections(resume_obj: ParsedResume) -> Dict[str, List[Dict[str, Any]]]:
    """Build original bullet groups for side-by-side rendering."""
    original_exp = [
        {
            "title": e.title,
            "company": e.company,
            "original_bullets": e.bullets or [],
        }
        for e in resume_obj.experience
    ]
    original_proj = [
        {
            "title": p.title,
            "company": "Project",
            "original_bullets": p.description or [],
        }
        for p in resume_obj.projects
    ]
    original_pos = [
        {
            "title": p.title,
            "organization": p.organization,
            "original_bullets": p.bullets or [],
        }
        for p in resume_obj.positions_of_responsibility
    ]
    return {
        "experience": original_exp,
        "projects": original_proj,
        "positions": original_pos,
    }


def _improved_copy_blocks(optimized: Dict[str, Any]) -> Tuple[str, str, str]:
    """Create copy-ready strings with one block per section."""
    def _flatten(items: List[Dict[str, Any]], label: str) -> str:
        lines = [f"{label}:"]
        for item in items or []:
            title = str(item.get("title", ""))
            org = str(item.get("company", "") or item.get("organization", ""))
            lines.append(f"- {title} ({org})")
            for b in item.get("optimized_bullets", []) or []:
                lines.append(f"  - {b}")
        return "\n".join(lines).strip()

    return (
        _flatten(optimized.get("optimized_experience", []), "Improved Experience"),
        _flatten(optimized.get("optimized_projects", []), "Improved Projects"),
        _flatten(optimized.get("optimized_positions", []), "Improved Positions"),
    )


def handle_optimizer(
    parsed_resume: Dict[str, Any],
    matches: List[Dict[str, Any]],
    selected_job_label: str,
) -> Tuple[str, str, str, str, str, str]:
    """Generate side-by-side original vs optimized bullets and summary suggestions."""
    parsed_resume_dict = _normalize_resume_state(parsed_resume)
    if not parsed_resume_dict:
        raise gr.Error("Please upload and parse your resume first.")
    if not matches:
        raise gr.Error("No matched jobs found. Please refresh matches first.")
    if not selected_job_label:
        raise gr.Error("Please select a job from top matches.")

    selected_job = _find_selected_job(matches, selected_job_label)
    if not selected_job:
        raise gr.Error("Selected job was not found. Please reselect and try again.")

    try:
        resume_obj = parsed_resume if isinstance(parsed_resume, ParsedResume) else ParsedResume.model_validate(parsed_resume_dict)
        optimized = optimize_resume_bullets(parsed_resume=resume_obj, selected_job=selected_job)
    except Exception as exc:
        raise gr.Error(f"Resume optimization failed: {exc}") from exc

    original = _collect_original_sections(resume_obj)
    original_md = "\n\n".join(
        [
            _section_to_markdown(original["experience"], "Original Experience", improved=False),
            _section_to_markdown(original["projects"], "Original Projects", improved=False),
            _section_to_markdown(original["positions"], "Original Positions", improved=False),
        ]
    )
    improved_md = "\n\n".join(
        [
            _section_to_markdown(optimized.get("optimized_experience", []), "Improved Experience", improved=True),
            _section_to_markdown(optimized.get("optimized_projects", []), "Improved Projects", improved=True),
            _section_to_markdown(optimized.get("optimized_positions", []), "Improved Positions", improved=True),
        ]
    )
    copy_exp, copy_proj, copy_pos = _improved_copy_blocks(optimized)
    summary = str(optimized.get("suggestions_summary", "") or "No additional suggestions.")
    return original_md, improved_md, summary, copy_exp, copy_proj, copy_pos


def clear_all_data():
    """Reset all states and key UI outputs to initial defaults."""
    return (
        None,  # state_resume_obj
        [],  # state_matches
        None,  # state_rag_chain
        {},  # parsed_resume_output
        "All data cleared. Upload a resume to begin again.",  # parse_status
        "No matches found yet. Parse your resume first, then refresh matches.",  # matches_md
        "",  # cover_output_md
        "",  # copy_textbox
        None,  # download_file
        "Upload and parse your resume first to enable cover letter generation.",  # cover_status
        "",  # original_md
        "",  # improved_md
        "",  # suggestions_summary
        "",  # copy_exp
        "",  # copy_proj
        "",  # copy_pos
        "Upload and parse your resume first to enable optimizer.",  # optimizer_status
        gr.update(value=None, choices=[]),  # job_selector
        gr.update(value=None, choices=[]),  # optimizer_job_selector
        gr.update(interactive=False),  # cover_btn
        gr.update(interactive=False),  # regenerate_btn
        gr.update(interactive=False),  # optimize_btn
        gr.update(interactive=False),  # matches_tab
        gr.update(interactive=False),  # chatbot_tab
        gr.update(interactive=False),  # cover_tab
        gr.update(interactive=False),  # optimizer_tab
        gr.update(selected="home_tab"),  # tabs container
    )


def handle_chat(
    user_message: str,
    history: List[Dict[str, str]],
    parsed_resume: Dict[str, Any],
    rag_chain_obj: Any,
) -> str:
    """Run real RAG chain for chatbot responses."""
    parsed_resume_dict = _normalize_resume_state(parsed_resume)
    if not parsed_resume_dict:
        raise gr.Error("Please upload and parse your resume first.")
    if not user_message.strip():
        return "Please ask a question about your fit."

    try:
        resume_obj = parsed_resume if isinstance(parsed_resume, ParsedResume) else ParsedResume.model_validate(parsed_resume_dict)
        rag_chain = rag_chain_obj or get_rag_chain(resume_obj)
        _ = history
        result = rag_chain.invoke({"input": user_message})
        return str(result.get("answer", "")).strip() or "I could not generate an answer for that question."
    except Exception as exc:
        raise gr.Error(f"Chatbot error: {exc}") from exc


def build_interface() -> gr.Blocks:
    """Create the multi-tab Gradio application UI."""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            """
            <div style="padding: 12px 0 4px 0;">
              <h1 style="margin-bottom:6px;">AI Resume & Job Matcher + Smart Cover Letter Generator</h1>
              <p style="margin-top:0;color:#4b5563;">Upload your resume, discover top role matches, chat about fit, generate cover letters, and optimize bullets.</p>
            </div>
            """
        )
        clear_btn = gr.Button("Clear All Data", variant="stop")

        state_resume_obj = gr.State(None)
        state_matches = gr.State([])
        state_rag_chain = gr.State(None)

        with gr.Tabs(selected="home_tab") as tabs:
            with gr.Tab("Home", id="home_tab"):
                gr.Markdown(
                    """
                    ### Welcome
                    1. Upload and parse your resume in **Upload**.
                    2. Review top role matches in **Matches**.
                    3. Ask fit questions in **Chatbot**.
                    4. Generate tailored letters in **Cover Letter**.
                    5. Improve bullets in **Optimizer**.
                    """
                )
                gr.Markdown("> Tip: Use a text-based PDF for best extraction quality.")

            with gr.Tab("Upload", id="upload_tab"):
                with gr.Row():
                    resume_file = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
                parse_btn = gr.Button("Parse Resume", variant="primary")
                parse_status = gr.Textbox(label="Status", interactive=False)
                parsed_resume_output = gr.JSON(label="Parsed Resume JSON")

            with gr.Tab("Matches", id="matches_tab", interactive=False) as matches_tab:
                gr.Markdown("## Job Matches")
                top_k = gr.Slider(label="Top K Matches", minimum=1, maximum=10, value=5, step=1)
                match_btn = gr.Button("Refresh Matches", variant="primary")
                matches_md = gr.Markdown(label="Top Matches", sanitize_html=False)

            with gr.Tab("Chatbot", id="chatbot_tab", interactive=False) as chatbot_tab:
                gr.Markdown("## Career Fit Chatbot")
                gr.ChatInterface(
                    fn=lambda message, history, parsed_resume, rag_chain_obj: handle_chat(
                        message,
                        history,
                        parsed_resume,
                        rag_chain_obj,
                    ),
                    additional_inputs=[state_resume_obj, state_rag_chain],
                    title="Role Fit Chatbot",
                    description=(
                        "Ask about job fit, missing skills, interview preparation, and salary expectations."
                    ),
                )

            with gr.Tab("Cover Letter", id="cover_tab", interactive=False) as cover_tab:
                gr.Markdown("## Cover Letter Generator")
                cover_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload and parse your resume first to enable cover letter generation.",
                )
                job_selector = gr.Dropdown(label="Select a Job (Top Matches)", choices=[], interactive=True)
                tone_selector = gr.Radio(label="Tone", choices=["professional", "enthusiastic", "concise"], value="professional")
                download_format = gr.Radio(
                    label="Download Format",
                    choices=["TXT (.txt)", "DOCX (.docx)"],
                    value="TXT (.txt)",
                )
                with gr.Row():
                    cover_btn = gr.Button("Generate Cover Letter", variant="primary", interactive=False)
                    regenerate_btn = gr.Button("Regenerate", variant="secondary", interactive=False)
                cover_output_md = gr.Markdown(label="Generated Cover Letter")
                copy_textbox = gr.Textbox(label="Copy-ready text", lines=10, interactive=False)
                with gr.Row():
                    copy_btn = gr.Button("Copy to Clipboard")
                    download_file = gr.File(label="Download Letter")
                copy_status = gr.Textbox(label="Copy Status", interactive=False)

            with gr.Tab("Optimizer", id="optimizer_tab", interactive=False) as optimizer_tab:
                gr.Markdown("## Resume Bullet Optimizer")
                optimizer_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload and parse your resume first to enable optimizer.",
                )
                optimizer_job_selector = gr.Dropdown(label="Select a Job (Top Matches)", choices=[], interactive=True)
                optimize_btn = gr.Button("Optimize My Resume Bullets", variant="primary", interactive=False)
                with gr.Row():
                    original_md = gr.Markdown(label="Original Bullets")
                    improved_md = gr.Markdown(label="Improved Bullets")
                suggestions_summary = gr.Textbox(label="Suggestions Summary", lines=4, interactive=False)
                copy_exp = gr.Textbox(label="Improved Experience (Copy)", lines=8, interactive=False)
                copy_proj = gr.Textbox(label="Improved Projects (Copy)", lines=8, interactive=False)
                copy_pos = gr.Textbox(label="Improved Positions (Copy)", lines=8, interactive=False)
                with gr.Row():
                    copy_exp_btn = gr.Button("Copy Experience Bullets")
                    copy_proj_btn = gr.Button("Copy Project Bullets")
                    copy_pos_btn = gr.Button("Copy Position Bullets")
                optimizer_copy_status = gr.Textbox(label="Copy Status", interactive=False)

        parse_evt = parse_btn.click(
            fn=handle_resume_upload,
            inputs=resume_file,
            outputs=[parsed_resume_output, state_resume_obj, parse_status],
            show_progress="full",
        )
        parse_evt.then(
            fn=lambda resume_obj: handle_find_matches(resume_obj, top_k=5),
            inputs=state_resume_obj,
            outputs=state_matches,
            show_progress="full",
        )
        parse_evt.then(
            fn=lambda resume_obj: get_rag_chain(resume_obj) if isinstance(resume_obj, ParsedResume) else None,
            inputs=state_resume_obj,
            outputs=state_rag_chain,
            show_progress="full",
        )
        parse_evt.then(fn=lambda: gr.update(interactive=True), outputs=matches_tab)
        parse_evt.then(fn=lambda: gr.update(interactive=True), outputs=chatbot_tab)
        parse_evt.then(fn=lambda: gr.update(interactive=True), outputs=cover_tab)
        parse_evt.then(fn=lambda: gr.update(interactive=True), outputs=optimizer_tab)
        parse_evt.then(fn=lambda: gr.update(interactive=True), outputs=cover_btn)
        parse_evt.then(fn=lambda: gr.update(interactive=True), outputs=regenerate_btn)
        parse_evt.then(fn=lambda: gr.update(interactive=True), outputs=optimize_btn)
        parse_evt.then(fn=lambda: gr.update(selected="matches_tab"), outputs=tabs)

        match_btn.click(
            fn=refresh_matches,
            inputs=[state_resume_obj, top_k],
            outputs=[state_matches, matches_md],
            show_progress="full",
        )
        state_matches.change(fn=format_matches_markdown, inputs=state_matches, outputs=matches_md)
        state_matches.change(
            fn=lambda matches: gr.update(choices=_job_dropdown_choices(matches), value=None),
            inputs=state_matches,
            outputs=job_selector,
        )
        state_matches.change(
            fn=lambda matches: gr.update(choices=_job_dropdown_choices(matches), value=None),
            inputs=state_matches,
            outputs=optimizer_job_selector,
        )

        cover_btn.click(
            fn=handle_cover_letter,
            inputs=[state_resume_obj, state_matches, job_selector, tone_selector, download_format],
            outputs=[cover_output_md, copy_textbox, download_file, cover_status],
            show_progress="full",
        )
        regenerate_btn.click(
            fn=handle_cover_letter,
            inputs=[state_resume_obj, state_matches, job_selector, tone_selector, download_format],
            outputs=[cover_output_md, copy_textbox, download_file, cover_status],
            show_progress="full",
        )
        copy_btn.click(fn=copy_hint_text, inputs=copy_textbox, outputs=copy_status)

        optimize_btn.click(
            fn=handle_optimizer,
            inputs=[state_resume_obj, state_matches, optimizer_job_selector],
            outputs=[original_md, improved_md, suggestions_summary, copy_exp, copy_proj, copy_pos],
            show_progress="full",
        )
        optimize_btn.click(fn=lambda: "Resume optimization completed.", outputs=optimizer_status)
        copy_exp_btn.click(fn=_copy_status_text, inputs=copy_exp, outputs=optimizer_copy_status)
        copy_proj_btn.click(fn=_copy_status_text, inputs=copy_proj, outputs=optimizer_copy_status)
        copy_pos_btn.click(fn=_copy_status_text, inputs=copy_pos, outputs=optimizer_copy_status)

        clear_btn.click(
            fn=clear_all_data,
            outputs=[
                state_resume_obj,
                state_matches,
                state_rag_chain,
                parsed_resume_output,
                parse_status,
                matches_md,
                cover_output_md,
                copy_textbox,
                download_file,
                cover_status,
                original_md,
                improved_md,
                suggestions_summary,
                copy_exp,
                copy_proj,
                copy_pos,
                optimizer_status,
                job_selector,
                optimizer_job_selector,
                cover_btn,
                regenerate_btn,
                optimize_btn,
                matches_tab,
                chatbot_tab,
                cover_tab,
                optimizer_tab,
                tabs,
            ],
        )

        gr.Markdown(
            "<div style='margin-top:14px;padding-top:10px;border-top:1px solid #e5e7eb;color:#6b7280;font-size:13px;'>"
            "Built by Manas Pant • Powered by LangChain + Gemini + Chroma"
            "</div>"
        )

    return demo


if __name__ == "__main__":
    # Ensure expected project folders exist.
    Path("jobs_data").mkdir(exist_ok=True)
    Path("chroma_db").mkdir(exist_ok=True)

    app = build_interface()
    app.launch()
