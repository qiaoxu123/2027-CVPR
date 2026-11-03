"""TRAIN-only dense region objective; deployment V3 architecture is unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


class RegionAuxiliaryHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.classifier = nn.Linear(hidden*3, 3)

    def forward(self, *, image_features, projection):
        if image_features.shape[-2] != 3:
            raise ValueError('three actual observed feature scales required')
        features = F.gelu(projection(F.normalize(image_features.float(), dim=-1)))
        return self.classifier(features.flatten(-2))


def region_auxiliary_loss(logits, labels, valid, class_weights):
    if logits.shape[:-1] != labels.shape or valid.shape != labels.shape or logits.shape[-1] != 3:
        raise ValueError('aligned region scores, private labels and validity required')
    if valid.dtype != torch.bool or not ((labels >= -1) & (labels < 3)).all():
        raise ValueError('unknown or three region labels only')
    known = (labels >= 0) & valid
    if not known.any():
        return logits.sum()*0, 0
    return F.cross_entropy(logits[known], labels[known].long(), weight=class_weights), int(known.sum())
