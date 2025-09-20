import os
import textwrap
import streamlit as st
from dotenv import load_dotenv

from parser import extract_text_from_file
from analyze import analyze_resume, analyze_with_jd


# -------------------- Config & Secrets --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="AI Resume Reviewer", page_icon="📄", layout="wide")
st.title("AI Resume Reviewer")

st.caption(
    "Note: Uploaded text is sent to the OpenAI API for analysis. "
    "Avoid including sensitive personal data."
)


# -------------------- Inputs --------------------
uploaded_file = st.file_uploader(
    "Upload your resume (.pdf / .docx / .txt)",
    type=["pdf", "docx", "txt"],
)
colA, colB = st.columns([2, 2])

with colA:
    job_role = st.text_input("Target job role (optional)", placeholder="e.g., AI Engineer")

with colB:
    jd_file = st.file_uploader(
        "Upload Job Description (.pdf / .docx / .txt) — optional",
        type=["pdf", "docx", "txt"],
        key="jd",
    )

analyze = st.button("Analyze")


# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def parse_once(file) -> str:
    """Cache parsing so we don't re-extract text on every rerun."""
    file.seek(0)
    return extract_text_from_file(file)


def _render_bullets(title: str, items: list[str]) -> None:
    """Render a section title followed by a clean bullet list."""
    st.subheader(title)
    if not items:
        st.write("—")
        return
    for it in items:
        st.markdown(f"- {it}")


def _make_review_md(review: dict) -> str:
    """Build Markdown for the generic/role-aware review."""
    md = []
    md.append(f"# Resume Review\n")
    md.append(f"**Overall score:** {review.get('overall_score', 0)}\n")
    md.append(f"## Executive Summary\n{review.get('executive_summary', '')}\n")
    md.append("## Strengths")
    md += [f"- {s}" for s in review.get("strengths", [])]
    md.append("\n## Issues")
    md += [f"- {s}" for s in review.get("issues", [])]
    md.append("\n## Action Items")
    md += [f"- {s}" for s in review.get("action_items", [])]
    md.append("\n## Missing Skills (general/role-aware)")
    md += [f"- {s}" for s in review.get("missing_skills", [])]
    md.append("\n## Role-specific Tips")
    md += [f"- {s}" for s in review.get("tailored_suggestions", [])]
    return "\n".join(md).strip() + "\n"


def _make_ats_md(ats: dict) -> str:
    """Build Markdown for the JD-aware ATS section."""
    md = []
    md.append(f"# ATS Match (Job Description-aware)\n")
    md.append(f"**Match score:** {ats.get('match_score', 0)}\n")
    md.append("## Skills Matched")
    md += [f"- {s}" for s in ats.get("skills_matched", [])]
    md.append("\n## Skills Missing (from JD perspective)")
    md += [f"- {s}" for s in ats.get("skills_missing", [])]
    md.append("\n## Tailoring Suggestions for this JD")
    md += [f"- {s}" for s in ats.get("suggestions_to_tailor", [])]
    return "\n".join(md).strip() + "\n"


# -------------------- Main Flow --------------------
if analyze:
    # Basic validations
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY (set it in .env or .streamlit/secrets.toml).")
        st.stop()
    if not uploaded_file:
        st.error("Please upload a resume file first.")
        st.stop()

    # Parse resume
    try:
        resume_text = parse_once(uploaded_file)
    except Exception as e:
        st.error(f"Failed to parse resume: {e}")
        st.stop()

    if not resume_text.strip():
        st.error("The resume file appears to be empty.")
        st.stop()

    # Optional JD parsing
    jd_text = None
    if jd_file:
        try:
            jd_text = parse_once(jd_file)
        except Exception as e:
            st.warning(f"Failed to parse Job Description: {e}")
            jd_text = None
        if jd_text and not jd_text.strip():
            st.warning("The uploaded Job Description is empty. Proceeding with role-based review only.")
            jd_text = None

    # Optional preview expanders (for transparency/debugging)
    with st.expander("Preview: parsed resume text (first 600 chars)"):
        st.code(resume_text[:600] + ("..." if len(resume_text) > 600 else ""))

    if jd_text:
        with st.expander("Preview: parsed JD text (first 600 chars)"):
            st.code(jd_text[:600] + ("..." if len(jd_text) > 600 else ""))

    # -------- 1) Always run the generic/role-aware review --------
    with st.spinner("Analyzing resume..."):
        review = analyze_resume(resume_text, job_role, OPENAI_API_KEY)

    st.header("Resume Review")
    st.metric("Overall score", review.get("overall_score", 0))

    st.subheader("Executive Summary")
    st.write(review.get("executive_summary", review.get("summary", "")))

    _render_bullets("Strengths", review.get("strengths", []))
    _render_bullets("Issues", review.get("issues", []))
    _render_bullets("Action Items", review.get("action_items", []))
    _render_bullets("Missing Skills (general/role-aware)", review.get("missing_skills", []))
    _render_bullets("Role-specific Tips", review.get("tailored_suggestions", []))

    # -------- 2) If JD provided, also run the ATS JD-aware analysis --------
    ats = None
    if jd_text:
        with st.spinner("Running ATS match against the Job Description..."):
            ats = analyze_with_jd(resume_text, jd_text, job_role, OPENAI_API_KEY)

        st.header("ATS Match (Job Description-aware)")
        st.metric("Match score", ats.get("match_score", 0))
        _render_bullets("Skills Matched (present in both Resume & JD)", ats.get("skills_matched", []))
        _render_bullets("Skills Missing (required by JD but not in Resume)", ats.get("skills_missing", []))
        _render_bullets("Tailoring Suggestions for this JD", ats.get("suggestions_to_tailor", []))

    # -------- Combined report download (Review + optional ATS) --------
    review_md = _make_review_md(review)
    report_md = review_md
    if ats:
        report_md += "\n\n" + _make_ats_md(ats)

    st.download_button(
        "Download full report (.md)",
        report_md,
        file_name="resume_review_report.md",
        type="primary",
    )
