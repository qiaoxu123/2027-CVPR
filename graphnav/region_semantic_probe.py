"""Vision-only linear diagnostic, not a replacement navigation policy."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RegionSemanticProbe(nn.Module):
    def __init__(self, dimension=1024):
        super().__init__()
        self.classifier = nn.Linear(3*dimension, 3)

    def forward(self, *, image_features):
        if image_features.shape[-2] != 3:
            raise ValueError('three frozen native image scales required')
        return self.classifier(F.normalize(image_features.float(), dim=-1).flatten(-2))


def shuffle_observed_features(features, valid):
    output = features.clone()
    ids = valid.nonzero().flatten()
    output[ids] = features[ids.flip(0)]
    return output


def classification_metrics(truth, prediction):
    truth = np.asarray(truth); prediction = np.asarray(prediction)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    support = confusion.sum(-1); predicted = confusion.sum(0)
    diagonal = confusion.diagonal()
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2*diagonal, support+predicted, out=np.zeros(3), where=support+predicted > 0)
    return dict(patches=len(truth), accuracy=float(diagonal.sum()/max(1, len(truth))),
                balanced_accuracy=float(recall[support > 0].mean()) if (support > 0).any() else None,
                macro_f1=float(f1.mean()), class_recall=recall.tolist(),
                class_support=support.tolist(), confusion=confusion.tolist())
