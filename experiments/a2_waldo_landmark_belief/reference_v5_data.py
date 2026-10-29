"""Role-safe V5 data view with target-balanced TRAIN loss weights."""
import torch

from .reference_overlay_data import ReferenceOverlayViews
from .reference_v5_weighting import target_balanced_weights


class TargetBalancedReferenceViews:
    """Wrap the immutable V4 cache and adjust only TRAIN sample weights."""

    def __init__(self, role, *, directory=None):
        kwargs = {} if directory is None else {"directory": directory}
        self.base = ReferenceOverlayViews(role, **kwargs)
        self.role = role
        self.entries = self.base.entries
        self.weights = (target_balanced_weights(self.entries)
                        if role == "train" else None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base[index]
        if self.role == "train":
            row["labels"]["sample_weight"] = torch.tensor(
                self.weights[index], dtype=torch.float32)
        return row
