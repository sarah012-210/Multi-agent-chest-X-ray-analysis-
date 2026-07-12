"""
End-to-end pipeline:

    X-ray image -> Image Agent -> Disease Prediction -> RAG Agent ->
    Medical Knowledge -> Report Agent -> Clinical Report -> Chat Agent

Run directly for a quick command-line test:
    python src/pipeline.py --image path/to/xray.jpg
"""
import argparse

from src.chat.chat_agent import ChatAgent
from src.inference.predict import DiseasePredictor
from src.knowledge.rag_agent import RAGAgent
from src.report.report_agent import ReportAgent


class CheXpertPipeline:
    def __init__(self):
        print("Loading Image Agent...")
        self.predictor = DiseasePredictor()
        print("Loading RAG Agent (embedding + indexing knowledge base)...")
        self.rag_agent = RAGAgent()
        print("Loading Report Agent...")
        self.report_agent = ReportAgent()
        self.chat_agent = None  # created after a report exists

    def run(self, image_path: str, threshold: float = None):
        # 1. Image Agent -> Disease Prediction
        predictions = self.predictor.predict(image_path)
        # Uses per-label thresholds from checkpoints/thresholds.json when
        # `threshold` is left as None (recommended for this model, since
        # different conditions have very different optimal cutoffs).
        flagged_dict = self.predictor.predict_flagged(image_path, threshold=threshold)
        flagged = list(flagged_dict.keys())

        # 2. RAG Agent -> Medical Knowledge
        knowledge_chunks = self.rag_agent.retrieve_for_conditions(flagged)

        # 3. Report Agent -> Clinical Report
        report = self.report_agent.generate_report(flagged_dict, knowledge_chunks)

        # 4. Chat Agent, ready for Q&A grounded in this report
        self.chat_agent = ChatAgent(report=report)

        return {
            "predictions": predictions,
            "flagged_conditions": flagged,
            "knowledge_chunks": knowledge_chunks,
            "report": report,
        }

    def chat(self, question: str) -> str:
        if self.chat_agent is None:
            raise RuntimeError("Run the pipeline on an image before chatting.")
        return self.chat_agent.ask(question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override with one flat threshold for all labels. "
                              "Omit to use the per-label thresholds in checkpoints/thresholds.json.")
    args = parser.parse_args()

    pipeline = CheXpertPipeline()
    result = pipeline.run(args.image, threshold=args.threshold)

    print("\n=== Predictions ===")
    for cond, prob in sorted(result["predictions"].items(), key=lambda x: -x[1]):
        print(f"{cond:30s} {prob:.3f}")

    print("\n=== Clinical Report (AI Draft) ===\n")
    print(result["report"])

    print("\nType a question about the report (or 'quit' to exit):")
    while True:
        q = input("> ")
        if q.strip().lower() in ("quit", "exit"):
            break
        print(pipeline.chat(q))


if __name__ == "__main__":
    main()
