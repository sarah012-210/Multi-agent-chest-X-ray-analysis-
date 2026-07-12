"""
Image Agent inference wrapper: takes an X-ray image path, returns a dict of
{condition: probability}. This is the "Image Agent" -> "Disease Prediction"
step of the pipeline.
"""
import json
import os
from typing import Dict, Optional

import torch
from PIL import Image

from src.config import CHEXPERT_LABELS, IMAGE_AGENT_CHECKPOINT, PREDICTION_THRESHOLD
from src.models.image_agent import load_image_agent
from src.preprocessing.dataset import build_transforms

# Per-label decision thresholds (found via F1-optimal search on the
# validation set during training). Falls back to PREDICTION_THRESHOLD for
# any label not present in this file, or if the file doesn't exist at all.
THRESHOLDS_PATH = os.path.join(
    os.path.dirname(IMAGE_AGENT_CHECKPOINT), "thresholds.json"
)


def load_thresholds() -> Dict[str, float]:
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, "r") as f:
            return json.load(f)
    return {}


class DiseasePredictor:
    """Thin, reusable wrapper around the trained Image Agent."""

    def __init__(self, checkpoint_path: str = IMAGE_AGENT_CHECKPOINT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_image_agent(checkpoint_path, device=self.device)
        self.transform = build_transforms(train=False)
        self.thresholds = load_thresholds()
        if self.thresholds:
            print(f"Loaded per-label thresholds from {THRESHOLDS_PATH}")

    def predict(self, image_path: str) -> Dict[str, float]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        return {label: float(prob) for label, prob in zip(CHEXPERT_LABELS, probs)}

    def predict_flagged(self, image_path: str, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Return conditions whose probability crosses their decision threshold.
        Uses the per-label thresholds from thresholds.json when available
        (recommended — this model's labels have very different optimal cutoffs,
        e.g. Cardiomegaly ~0.30 vs Atelectasis ~0.69). Pass an explicit
        `threshold` to override with one flat cutoff for all labels instead.
        """
        all_probs = self.predict(image_path)
        flagged = {}
        for label, prob in all_probs.items():
            if label == "No Finding":
                continue
            cutoff = threshold if threshold is not None else self.thresholds.get(label, PREDICTION_THRESHOLD)
            if prob >= cutoff:
                flagged[label] = prob
        return flagged


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.predict <path_to_xray_image>")
        sys.exit(1)

    predictor = DiseasePredictor()
    results = predictor.predict(sys.argv[1])
    for label, prob in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{label:30s} {prob:.3f}")
