"""A small CNN classifier used as a fixed feature extractor for FID.

For MNIST-scale grayscale data the canonical Inception-V3 FID is inappropriate
(it expects 299x299 RGB natural images). Instead we train a compact LeNet-style
classifier *once* on the real Simpsons-MNIST training set and reuse its
penultimate-layer activations as the feature space for the Frechet distance.

Because the extractor is trained only on real data and then frozen and cached to
disk, every generation's FID is measured in the *same* feature space, which is
exactly what we need to compare model quality across generations of a
model-collapse study.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import build_loader

FEATURE_DIM = 128
_CACHE_PATH = os.path.join("outputs", "feature_extractor.pt")


class FeatureCNN(nn.Module):
    """LeNet-ish classifier; ``features()`` returns the 128-d penultimate vector."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, FEATURE_DIM)
        self.fc2 = nn.Linear(FEATURE_DIM, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # -> 7x7
        x = x.flatten(1)
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))


def train_feature_extractor(
    device: torch.device,
    epochs: int = 8,
    cache_path: str = _CACHE_PATH,
) -> FeatureCNN:
    """Train the classifier on the real train split and cache its weights."""
    model = FeatureCNN().to(device)
    loader = build_loader("folder", split="train", batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    # Force grad on: this may be called lazily from within a torch.no_grad()
    # metrics context (e.g. compute_collapse_metrics), where backward would fail.
    with torch.enable_grad():
        for epoch in range(epochs):
            total, correct = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                opt.step()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.numel()
            print(f"  [feature-extractor] epoch {epoch + 1}/{epochs} "
                  f"train_acc={correct / total:.4f}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(model.state_dict(), cache_path)
    return model


def load_feature_extractor(
    device: torch.device,
    cache_path: str = _CACHE_PATH,
) -> FeatureCNN:
    """Load the cached feature extractor, training it first if absent.

    The extractor is always trained on *real* data only, regardless of which
    generation is currently being evaluated, so the FID feature space is fixed.
    """
    model = FeatureCNN().to(device)
    if os.path.exists(cache_path):
        model.load_state_dict(torch.load(cache_path, map_location=device))
        model.eval()
        return model
    print("[feature-extractor] no cache found; training on real data once...")
    model = train_feature_extractor(device, cache_path=cache_path)
    model.eval()
    return model


@torch.no_grad()
def extract_features(
    model: FeatureCNN,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return ``(N, FEATURE_DIM)`` features for a batch of ``(N, 1, 28, 28)`` images."""
    model.eval()
    feats = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        feats.append(model.features(batch).cpu())
    return torch.cat(feats, dim=0)
