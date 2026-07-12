"""
Test the Chat Agent by itself, using a sample report — no Image Agent or
real X-ray needed. Confirms grounded Q&A works via the Groq API.

Usage:
    python -m src.evaluation.test_chat_agent
"""
from src.chat.chat_agent import ChatAgent

SAMPLE_REPORT = """## Impression
The chest X-ray analysis suggests potential cardiomegaly and pulmonary edema.

## Findings Detail
- Cardiomegaly: probability 0.72
- Edema: probability 0.65

## Disclaimer
This is an AI-generated draft requiring physician review, not a diagnosis.
"""

TEST_QUESTIONS = [
    "What does Cardiomegaly mean in simple terms?",
    "Should I be worried about these results?",
]


def main():
    print("Loading Chat Agent...")
    agent = ChatAgent(report=SAMPLE_REPORT)

    for question in TEST_QUESTIONS:
        print(f"\n> {question}")
        answer = agent.ask(question)
        print(answer)


if __name__ == "__main__":
    main()
