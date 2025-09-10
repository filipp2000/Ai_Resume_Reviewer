# Purpose: Build prompt, call OpenAI, and return a structured (JSON) review so the UI can render metrics/lists.
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

RUBRIC  = """
Return JSON with keys:
- overall_score (0-100)

- summary (string)
  * an overview of 150–200 words
  * Include 2–3 high-impact observations and the candidate's positioning for the target role

- strengths (string[])
  * Each bullet is a single, specific idea (no multi-sentence paragraphs)

- issues (string[])
  * Focus on clarity, structure, impact, metrics, formatting consistency

- action_items (string[])
  * Imperative voice (“Add…”, “Refactor…”, “Quantify…”, “Reorder…”)

- missing_skills (string[])
  * 
  * MUST be the must-have skills for the **target role** (tools, frameworks, cloud, MLOps, data, soft skills)
  * Prefer concise phrases”
  * Avoid duplicates and generic items like “communication”

- tailored_suggestions (string[])
  * Role-specific recommendations, resume tailoring ideas, and portfolio additions
"""

def build_prompt(resume_text: str, role: str | None):
    """
    Construct system/user messages for the chat completion.
    We include a rubric and truncate overly long resumes to control cost.
    """
    role = role or "general applications"
    system = "You are an expert resume reviewer with years of experience in HR and recruitment. Be specific, concise, and actionable."
    user = f"""Analyze the following resume and score it on clarity, impact, skills relevance, and formatting.
Role to target: {role}

{RUBRIC}

Resume:
\"\"\"{resume_text[:20000]}\"\"\""""  # guardrail: limit input length
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def analyze_resume(resume_text: str, role: str | None, api_key: str) -> dict:
    """
    Call OpenAI with retries and parse a JSON response.
    """
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=build_prompt(resume_text, role),
        temperature=0.2,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    content = resp.choices[0].message.content
    return json.loads(content)