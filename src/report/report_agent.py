"""
Report Agent: takes the Image Agent's predictions plus the RAG Agent's
retrieved knowledge, and asks an LLM (Groq, free tier) to draft a
structured, readable clinical-style report.

Requires a free API key from https://console.groq.com/keys — sign up with
email or Google account, no credit card required. Set it as:
    $env:GROQ_API_KEY="your-key-here"
"""
import os
from typing import Dict, List

import requests

from src.config import GROQ_API_URL, GROQ_MODEL

SYSTEM_PROMPT = """You are a radiology report drafting assistant helping a clinician \
review an AI-assisted chest X-ray analysis. You are NOT making a diagnosis — you are \
drafting a structured summary for a qualified radiologist/physician to review, edit, \
and sign off on.

Always:
- Clearly state this is an AI-generated draft requiring physician review.
- Reference the model's confidence (probabilities) for each flagged finding.
- Use the provided medical knowledge context to explain findings in plain, accurate terms.
- Recommend appropriate next steps (e.g. clinical correlation, further imaging) \
  without making definitive treatment decisions.
- Keep the tone professional, concise, and structured with clear sections.
"""

REPORT_TEMPLATE = """## Findings from AI Image Analysis

{findings_block}

## Relevant Medical Context

{knowledge_block}

## Task

Using the findings and medical context above, draft a structured chest X-ray report \
with these sections: **Impression**, **Findings Detail**, **Recommended Next Steps**, \
and a closing **Disclaimer** noting this is an AI-generated draft for physician review, \
not a final diagnosis.
"""


class ReportAgent:
    def __init__(self, model: str = GROQ_MODEL):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. Get a free key at "
                "https://console.groq.com/keys and set it with:\n"
                '  $env:GROQ_API_KEY="your-key-here"'
            )
        self.model = model

    @staticmethod
    def _format_findings(flagged: Dict[str, float]) -> str:
        """`flagged` should already be the output of DiseasePredictor.predict_flagged()
        (i.e. per-label thresholds already applied) — this just formats it."""
        if not flagged:
            return "No conditions crossed their respective flagging thresholds — " \
                   "study appears unremarkable by the model."
        lines = [f"- {cond}: probability {prob:.2f}" for cond, prob in
                 sorted(flagged.items(), key=lambda x: -x[1])]
        return "\n".join(lines)

    @staticmethod
    def _format_knowledge(knowledge_chunks: List[dict]) -> str:
        if not knowledge_chunks:
            return "No additional knowledge context retrieved."
        lines = [f"**{chunk['condition']}**: {chunk['text']}" for chunk in knowledge_chunks]
        return "\n\n".join(lines)

    def generate_report(self, flagged: Dict[str, float], knowledge_chunks: List[dict]) -> str:
        findings_block = self._format_findings(flagged)
        knowledge_block = self._format_knowledge(knowledge_chunks)

        user_prompt = REPORT_TEMPLATE.format(
            findings_block=findings_block,
            knowledge_block=knowledge_block,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 1200,
            "temperature": 0.3,
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Groq response format: {data}")
