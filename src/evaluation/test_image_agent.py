"""
Test the Image Agent by itself — no RAG/Report/Chat agents involved.
Runs it against the validation set and reports per-label AUROC, plus
prints predicted vs. actual for a handful of sample images.

Usage:
    python -m src.evaluation.test_image_agent
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CHEXPERT_LABELS, DATA_DIR, KAGGLEHUB_PATH
from src.inference.predict import DiseasePredictor
from src.preprocessing.dataset import _resolve_image_path

VALID_CSV_TO_USE = f"{DATA_DIR}/valid.csv"  # official CheXpert validation set — no uncertain values, no filtering needed
N_SAMPLE_IMAGES_TO_PRINT = 5


def main():
    df = pd.read_csv(VALID_CSV_TO_USE)
    predictor = DiseasePredictor()

    print(f"\nRunning Image Agent on {len(df)} validation images...\n")

    all_probs, all_labels = [], []
    for _, row in df.iterrows():
        img_path = _resolve_image_path(row["Path"], KAGGLEHUB_PATH)
        preds = predictor.predict(img_path)
        all_probs.append([preds[label] for label in CHEXPERT_LABELS])
        all_labels.append([row[label] for label in CHEXPERT_LABELS])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("=== Per-label AUROC on validation set ===")
    for i, label in enumerate(CHEXPERT_LABELS):
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        print(f"  {label:20s} {auc:.3f}")

    print(f"\n=== Sample predictions (first {N_SAMPLE_IMAGES_TO_PRINT} images) ===")
    for idx in range(min(N_SAMPLE_IMAGES_TO_PRINT, len(df))):
        row = df.iloc[idx]
        img_path = _resolve_image_path(row["Path"], KAGGLEHUB_PATH)
        flagged = predictor.predict_flagged(img_path)

        print(f"\n{row['Path']}")
        for i, label in enumerate(CHEXPERT_LABELS):
            truth = int(row[label])
            prob = all_probs[idx][i]
            flag_marker = " <- FLAGGED" if label in flagged else ""
            print(f"  {label:20s} truth={truth}  predicted={prob:.3f}{flag_marker}")


if __name__ == "__main__":
    main()