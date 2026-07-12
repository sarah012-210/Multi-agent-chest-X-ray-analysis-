"""
Chat Agent: lets the user ask follow-up questions about the generated
report. Grounds each answer by pulling extra context from the RAG Agent
based on the user's question, then answers using the report + that context.

Uses Groq's free API — same key as the Report Agent:
    $env:GROQ_API_KEY="your-key-here"
"""
import os
from typing import List

import requests

from src.config import GROQ_API_URL, GROQ_MODEL
from src.knowledge.rag_agent import RAGAgent

CHAT_SYSTEM_PROMPT = """You are a medical assistant chatbot helping a user understand an \
AI-generated chest X-ray report. Answer questions using ONLY the report and the medical \
knowledge context provided to you. If a question requires clinical judgment beyond what's \
in the context (e.g. "should I be worried?", treatment decisions), gently redirect the user \
to discuss it with their physician or radiologist rather than speculating. \
Never present yourself as providing a diagnosis."""


class ChatAgent:
    def __init__(self, report: str, model: str = GROQ_MODEL):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. Get a free key at "
                "https://console.groq.com/keys and set it with:\n"
                '  $env:GROQ_API_KEY="your-key-here"'
            )
        self.model = model
        self.report = report
        self.rag_agent = RAGAgent()
        # OpenAI-style chat history: list of {"role": "user"/"assistant", "content": ...}
        self.history: List[dict] = []

    def ask(self, user_question: str) -> str:
        extra_context = self.rag_agent.retrieve_by_query(user_question, top_k=2)
        context_block = "\n\n".join(
            f"**{c['condition']}**: {c['text']}" for c in extra_context
        )

        grounded_prompt = f"""## Report
{self.report}

## Additional Medical Context
{context_block}

## User Question
{user_question}
"""

        messages = (
            [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
            + self.history
            + [{"role": "user", "content": grounded_prompt}]
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.3,
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Groq response format: {data}")

        # Keep prior turns as plain question/answer (not the huge grounded
        # prompt) so context sent to the API doesn't balloon over a long chat.
        self.history.append({"role": "user", "content": user_question})
        self.history.append({"role": "assistant", "content": answer})

        return answer
