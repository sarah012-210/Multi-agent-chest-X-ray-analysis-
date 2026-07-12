"""
Test the Report Agent by itself, using fake/sample predictions — no Image
Agent or real X-ray needed. Confirms your ANTHROPIC_API_KEY works and the
Report Agent produces a properly formatted clinical report.

Usage:
    python -m src.evaluation.test_report_agent
"""
from src.knowledge.rag_agent import RAGAgent
from src.report.report_agent import ReportAgent

# Fake predictions, standing in for what the Image Agent would normally produce.
SAMPLE_PREDICTIONS = {
    "Cardiomegaly": 0.72,
    "Edema": 0.65,
    "Consolidation": 0.10,
    "Atelectasis": 0.20,
    "Pleural Effusion": 0.15,
}

FLAGGED_CONDITIONS = ["Cardiomegaly", "Edema"]


def main():
    print("Loading RAG Agent...")
    rag = RAGAgent()
    knowledge = rag.retrieve_for_conditions(FLAGGED_CONDITIONS)

    print("Loading Report Agent (calls the Anthropic API)...")
    agent = ReportAgent()

    flagged_dict = {k: SAMPLE_PREDICTIONS[k] for k in FLAGGED_CONDITIONS}
    report = agent.generate_report(flagged_dict, knowledge)

    print("\n=== Generated Report ===\n")
    print(report)


if __name__ == "__main__":
    main()
