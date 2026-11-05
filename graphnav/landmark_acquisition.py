"""Conservative per-episode acquisition of a public, named visual landmark.

Important distinction: projected public Polygon overlap is a *candidate ROI*,
not RGB identification. A track is only provisionally RGB-supported after
PGVB visually separates positive/negative patches more strongly than the same
image with its patch features deliberately displaced, for consecutive frames.
No GT target, oracle position, camera distance threshold or new image encoder.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AcquisitionThresholds:
    min_positive: int = 2
    min_negative: int = 2
    min_auc: float = 0.75
    min_margin: float = 0.25
    min_counterfactual_gain: float = 0.10
    confirmation_frames: int = 2
    coast_frames: int = 2
    ramp_frames: int = 2
    max_visual_gain: float = 2.0

    def __post_init__(self):
        if self.min_positive < 1 or self.min_negative < 1:
            raise ValueError("At least one positive and negative patch required")
        if not 0.5 <= self.min_auc <= 1:
            raise ValueError("AUC threshold must be [0.5, 1]")
        if self.min_margin < 0 or self.min_counterfactual_gain < 0:
            raise ValueError("Visual margins must be nonnegative")
        if min(self.confirmation_frames, self.ramp_frames) < 1 or self.coast_frames < 0:
            raise ValueError("Invalid temporal thresholds")
        if not 0 < self.max_visual_gain <= 3:
            raise ValueError("Visual gain must be in (0, 3]")


class LandmarkAcquisition:
    """State resets per episode; tracking is scored from RGB-derived logits.

    This is *provisional visual tracking in a registered public ROI*, not
    an independent proof that an arbitrary proper name is visually readable.
    The counterfactual test makes pure geometric ROI visibility insufficient.
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or AcquisitionThresholds()
        self.reset()

    def reset(self):
        self.state = "geometry"
        self.confirmations = 0
        self.misses = 0
        self.acquisitions = 0
        self.last = self._audit("not_observed", visual_gain=0.0)

    def _audit(self, reason, *, visual_gain, positive=0, negative=0,
               auc=None, margin=None, counterfactual_auc=None):
        return {
            "state": self.state,
            "reason": reason,
            "visual_gain": float(visual_gain),
            "consecutive_confirmations": int(self.confirmations),
            "consecutive_misses": int(self.misses),
            "acquisition_events": int(self.acquisitions),
            "positive_patches": int(positive),
            "negative_patches": int(negative),
            "visual_auc": auc,
            "visual_margin": margin,
            "counterfactual_auc": counterfactual_auc,
            "identity_independently_verified": False,
            "method": "public_ROI_plus_RGB_contrast_and_temporal_consistency",
        }

    def observe(self, scores=None, counterfactual=None, labels=None, valid=None):
        """Update from the *same frame* scores of real and displaced RGB patches.

        labels are weak, public-Polygon-aligned 1=inside, 0=outside, -1=unknown.
        They select comparisons but DO NOT themselves establish acquisition.
        """
        cfg = self.thresholds
        auc = margin = fake_auc = None
        pcount = ncount = 0
        confirmed = False
        reason = "no_visual_candidate"
        if scores is not None:
            if any(x is None for x in (counterfactual, labels, valid)):
                raise ValueError("Real/counterfactual scores, labels and validity are required")
            if (scores.ndim != 2 or scores.shape[0] != 1
                    or counterfactual.shape != scores.shape
                    or labels.shape != scores.shape
                    or valid.shape != scores.shape):
                raise ValueError("Only consistent single-frame aligned patch evidence is supported")
            if not torch.isfinite(scores).all() or not torch.isfinite(counterfactual).all():
                raise ValueError("Nonfinite RGB scores")
            positive = (labels == 1) & valid.bool()
            negative = (labels == 0) & valid.bool()
            pcount, ncount = int(positive.sum()), int(negative.sum())
            if pcount >= cfg.min_positive and ncount >= cfg.min_negative:
                pos = scores[positive].detach().float()
                neg = scores[negative].detach().float()
                fake_pos = counterfactual[positive].detach().float()
                fake_neg = counterfactual[negative].detach().float()

                def auc_score(a, b):
                    differences = a[:, None] - b[None, :]
                    return float(((differences > 0).float()
                                  + .5 * (differences == 0).float()).mean())

                auc = auc_score(pos, neg)
                fake_auc = auc_score(fake_pos, fake_neg)
                margin = float(pos.mean() - neg.mean())
                confirmed = (
                    auc >= cfg.min_auc
                    and margin >= cfg.min_margin
                    and auc - fake_auc >= cfg.min_counterfactual_gain
                )
                reason = "rgb_discriminative" if confirmed else "weak_rgb_discrimination"
            else:
                reason = "insufficient_positive_or_negative_patches"

        if confirmed:
            self.misses = 0
            self.confirmations += 1
            if self.confirmations >= cfg.confirmation_frames:
                if self.state != "tracking":
                    self.acquisitions += 1
                self.state = "tracking"
                progress = (self.confirmations - cfg.confirmation_frames + 1) / cfg.ramp_frames
                gain = cfg.max_visual_gain * min(1.0, progress)
            else:
                self.state = "candidate"
                gain = 0.0
        else:
            self.misses += 1
            self.confirmations = 0
            self.state = (
                "coasting"
                if self.state in ("tracking", "coasting") and self.misses <= cfg.coast_frames
                else "geometry"
            )
            # No stale visual evidence may be added during lost observations.
            gain = 0.0
        self.last = self._audit(
            reason, visual_gain=gain, positive=pcount, negative=ncount,
            auc=auc, margin=margin, counterfactual_auc=fake_auc,
        )
        return self.last
