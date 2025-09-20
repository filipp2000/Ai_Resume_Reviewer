import json
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_openai(messages: list[dict], api_key: str) -> dict:
    """
    Call OpenAI with retries and parse a JSON response.
    """
    client = OpenAI(api_key=api_key, timeout=30)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    content = resp.choices[0].message.content
    return json.loads(content)
  

# === Generic Resume Reviewer ===

RUBRIC_GENERAL = """
Return JSON with keys:
- overall_score (0-100)

- summary (string)
  * Provide a cohesive and concrete overview of 150–200 words, 
  * Include 2–3 high-impact observations and the candidate's positioning for the target role

- strengths (string[])
  * Each bullet is a single, specific idea (no multi-sentence paragraphs)

- issues (string[])
  * Focus on clarity, structure, impact, metrics, formatting consistency

- action_items (string[])
  * Imperative voice (“Add…”, “Refactor…”, “Quantify…”, “Reorder…”)

- missing_skills (string[])
  * MUST be the must-have skills for the **target role** (tools, frameworks, cloud, MLOps, data, soft skills)
  * Prefer concise phrases”
  * Avoid duplicates and generic items like “communication”

- tailored_suggestions (string[])
  * Role-specific recommendations, resume tailoring ideas, and portfolio additions
"""

def build_message_general(resume_text: str, role: str | None):
    """
    Construct system/user message for a structured resume critique.
    We include a rubric, output valid JSON only and truncate overly long resumes to control cost.
    """
    role = (role or "general applications").strip()
    system = "You are an expert resume reviewer with years of experience in HR and recruitment. Be specific, concise, and actionable."
    user = f"""Analyze the following resume and score it on clarity, impact, skills relevance, and formatting.
Role to target: {role}

{RUBRIC_GENERAL}

Resume:
\"\"\"{resume_text[:20000]}\"\"\""""  # limit input length
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    

def analyze_resume(resume_text: str, role: str | None, api_key: str) -> dict:
    """
    Generic/role-aware resume review.
    Returns a dict with:
      overall_score, summary, strengths, issues, action_items,
      missing_skills, tailored_suggestions
    """
    data = _call_openai(build_message_general(resume_text, role), api_key)

    # Defensive normalization
    # overall_score -> int
    raw_score = data.get("overall_score", 0)
    data["overall_score"] = int(raw_score) if str(raw_score).isdigit() else 0

    # lists -> always lists
    for key in ("strengths", "issues", "action_items", "missing_skills", "tailored_suggestions"):
        if not isinstance(data.get(key), list):
            data[key] = []

    # summary -> always plain string
    summary = data.get("summary", "")
    if isinstance(summary, list):
        summary = " ".join([str(x).strip() for x in summary if str(x).strip()])
    elif not isinstance(summary, str):
        summary = str(summary)
    data["summary"] = summary.strip()

    return data


# === Resume Reviewer with job descrption provided ===
RUBRIC_JD = """
Return JSON with keys:
- match_score (0-100) : Overall score of the resume for this JD (skills, seniority, tooling, domain alignment)
- skills_matched (string[]): Skills/technologies present in BOTH the resume and the JD
- skills_missing (string[]): Important JD skills not covered (or weak) in the resume
- suggestions_to_tailor (string[]): Concrete tailoring tips specific to this JD
"""


def build_message_with_jd(resume_text: str, jd_text: str, role: str | None) -> list[dict]:
    """
    Build messages for ATS-style JD matching.
    We pass both the resume and the job description to the model.
    """
    role = (role or "general applications").strip()
    system = (
        "You are a senior technical recruiter and ATS specialist. "
        "Be precise, extract concrete skills, and output valid JSON only."
    )
    user = f"""Perform ATS-style matching for the target role: **{role}**.
Follow the rubric strictly and output valid JSON only.

{RUBRIC_JD}

Resume:
\"\"\"{resume_text[:20000]}\"\"\"

Job Description:
\"\"\"{jd_text[:20000]}\"\"\""""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def analyze_with_jd(resume_text: str, jd_text: str, role: str | None, api_key: str) -> dict:
    """
    JD-aware (ATS) review.
    Returns a dict with:
      match_score, skills_matched, skills_missing, suggestions_to_tailor
    """
    data = _call_openai(build_message_with_jd(resume_text, jd_text, role), api_key)

    # Defensive normalization
    try:
        score = int(data.get("match_score", 0))
    except Exception:
        score = 0
    data["match_score"] = max(0, min(100, score))

    for key in ("skills_matched", "skills_missing", "suggestions_to_tailor"):
        if key not in data or not isinstance(data[key], list):
            data[key] = []

    return data
