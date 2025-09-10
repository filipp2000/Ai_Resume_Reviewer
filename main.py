import truststore
truststore.inject_into_ssl()  # <-- use Windows certificates for Python SSL

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from parser import extract_text_from_file 
from analyze import analyze_resume   

load_dotenv() # load OPENAI_API_KEY from .env file
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# Setup Streamlit UI
# Streamlit will automatically handle rendering it correctly
st.set_page_config(page_title="AI Resume Reviewer", page_icon="📃", layout="centered")
st.title("AI Resume Reviewer")
# Display string formatted as Markdown.
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

uploaded_file = st.file_uploader(
    "Upload your resume (.pdf / .docx / .txt)",
    type=["pdf", "docx", "txt"]
)

job_role = st.text_input("Enter the job role you're targeting (optional)")

# Add analyze button
analyze = st.button("Analyze Resume")


def cache_parsing(file) -> str:
    """
    Cache parsing results so Streamlit doesn't re-parse the same file on every rerun.
    Note: we must reset the file pointer before reading.
    """
    file.seek(0)
    return extract_text_from_file(file)

# ---- Main flow ----
if analyze:
    if not uploaded_file:
        st.error("Please upload a resume file first.")
        st.stop()

    try:
        # 1) Parse file content (robust PDF/DOCX/TXT) with caching
        file_content = cache_parsing(uploaded_file)
        if not file_content.strip():
            st.error("File appears to be empty.")
            st.stop()

        # 2) Call OpenAI to get structured JSON review (score, strengths, issues, etc.)
        if not OPENAI_API_KEY:
            st.error("Missing OPENAI_API_KEY (set in .env).")
            st.stop()

        with st.spinner("Analyzing resume..."):
            data = analyze_resume(file_content, job_role, OPENAI_API_KEY)

        # ---- Render results ----
        st.success("Analysis completed")

        # # Score block
        st.subheader("Overall Score")
        score = int(data.get("overall_score", 0))
        st.metric("Score", score)
        st.progress(score/100)

        st.subheader("Summary")
        st.write(data.get("summary", ""))

        st.subheader("Strengths")
        for s in data.get("strengths", []):
            st.markdown(f"- {s}")

        st.subheader("Issues")
        for s in data.get("issues", []):
            st.markdown(f"- {s}")

        st.subheader("Action items")
        for a in data.get("action_items", []):
            st.markdown(f"- {a}")

        st.subheader("Missing skills")
        if data.get("missing_skills"):
            for m in data.get("missing_skills", []):
                st.markdown(f"- {m}")
        else:
            st.write("—")

        st.subheader("Role-specific tips")
        for tip in data.get("tailored_suggestions", []):
            st.markdown(f"- {tip}")


        # ---- Download report (Markdown) ----
        report_md = (
            f"# Resume Review\n\n**Score:** {data.get('overall_score', 0)}\n\n"
            f"## Summary\n{data.get('summary', '')}\n\n"
            "## Strengths\n" + "\n".join(f"- {s}" for s in data.get("strengths", [])) + "\n\n"
            "## Issues\n" + "\n".join(f"- {s}" for s in data.get("issues", [])) + "\n\n"
            "## Action Items\n" + "\n".join(f"- {s}" for s in data.get("action_items", [])) + "\n\n"
            "## Missing Skills\n" + ", ".join(data.get("missing_skills", [])) + "\n\n"
            "## Role-specific Tips\n" + "\n".join(f"- {s}" for s in data.get("tailored_suggestions", []))
        )
        st.download_button("Download report (.md)", report_md, file_name="resume_review.md")

    except Exception as e:
        st.error(f"An error occured: {str(e)}", icon="🚨")
        