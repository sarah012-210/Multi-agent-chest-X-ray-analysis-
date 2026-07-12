"""
Image Agent: the disease-prediction model.

Uses a DenseNet121 backbone (the standard CheXpert baseline architecture)
with the final classification layer replaced for multi-label output
(sigmoid over NUM_CLASSES independent conditions, since a single X-ray can
show more than one finding at once).
"""
import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


class ImageAgent(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)

        in_features = backbone.classifier.in_features
        # NOTE: must match the architecture used during training on Kaggle
        # (Dropout + Linear, not a bare Linear) or the saved checkpoint's
        # state_dict keys won't line up with this model's keys.
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # raw logits; apply sigmoid outside for probs


def load_image_agent(checkpoint_path: str = None, device: str = "cpu") -> ImageAgent:
    """
    Loads the Image Agent. If checkpoint_path doesn't exist, returns the
    ImageNet-pretrained backbone so the rest of the pipeline can still be
    exercised end-to-end (predictions just won't be clinically meaningful
    until you actually train it with src/train.py).
    """
    model = ImageAgent(pretrained=True)

    if checkpoint_path:
        import os
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded trained weights from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path} — using "
                  f"ImageNet-pretrained weights only (untrained for CheXpert).")

    model.to(device)
    model.eval()
    return model
