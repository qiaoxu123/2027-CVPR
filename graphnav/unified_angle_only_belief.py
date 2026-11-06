"""Angle-only landmark->target modality for the EXISTING full-text Unified V2.

New information to the head: ONLY an angular bearing on the public map's
planar XY axes. NO dx/dy displacement, Euclidean distance, scale/range,
target coordinates or future observations are model inputs.

Train uses true target locations EXCLUSIVELY to derive 8-class supervision.
Inference predicts the direction distribution from existing full instruction
BERT tokens and public landmark-name embeddings. True bearing at inference
would be an ORACLE and is explicitly forbidden for deployment.

Angle is measured counterclockwise from map +X. The CityNav TIF does not
independently verify geographic north; NEVER claim true world-north accuracy.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def public_candidate_bearings(pair_geometry: torch.Tensor) -> torch.Tensor:
    """[B,Q,M,10] EXISTING public pair features -> [B,Q,M] ANGLES ONLY.

    The transient first two geometric terms are used only to evaluate atan2.
    Their magnitude is not passed to the added model. All existing main-model
    features remain untouched, ensuring a matched baseline.
    """
    if pair_geometry.ndim!=4 or pair_geometry.shape[-1]!=10:
        raise ValueError("Expected original [B,Q,M,10] public Polygon geometry")
    if not torch.isfinite(pair_geometry).all():
        raise ValueError("Nonfinite public Polygon features")
    return torch.atan2(pair_geometry[...,1].float(),
                       pair_geometry[...,0].float())


def named_anchor_mask(entity_valid: torch.Tensor,
                      reference_counts: torch.Tensor) -> torch.Tensor:
    """Only original public named-landmark slots, not road/rule graph nodes."""
    if entity_valid.ndim!=2 or reference_counts.shape!=(entity_valid.shape[0],):
        raise ValueError("Invalid anchor mask/count shape")
    b,m=entity_valid.shape
    if (reference_counts<0).any() or (reference_counts>m).any():
        raise ValueError("Public landmark count exceeds capacity")
    return entity_valid.bool() & (
        torch.arange(m,device=entity_valid.device)[None,:]
        < reference_counts.long()[:,None])


def bearing_supervision(target_xy: torch.Tensor, anchor_centers_xy: torch.Tensor,
                        map_sizes_m: torch.Tensor, reference_valid: torch.Tensor,
                        *, bins=8, center_tolerance_m=1.0):
    """TRAIN LABELS ONLY. Do NOT call during inference or closed loop.

    Inputs are existing GT target_xy and PUBLIC anchor centers; the output is
    direction-class labels only, never position/distance features.
    """
    if (target_xy.ndim!=2 or target_xy.shape[-1]!=2 or
            anchor_centers_xy.ndim!=3 or anchor_centers_xy.shape[-1]!=2 or
            anchor_centers_xy.shape[0]!=target_xy.shape[0] or
            map_sizes_m.shape!=target_xy.shape or
            reference_valid.shape!=anchor_centers_xy.shape[:2]):
        raise ValueError("Invalid supervision input shapes")
    # Temporary physical XY separation for constructing supervision only.
    difference=(target_xy[:,None,:].float()-anchor_centers_xy.float())*map_sizes_m[:,None,:].float()
    length=difference.norm(dim=-1)
    angle=torch.atan2(difference[...,1],difference[...,0])
    label=torch.remainder(torch.round(
        torch.remainder(angle,2*math.pi)*bins/(2*math.pi)).long(),bins)
    valid=reference_valid.bool() & (length>center_tolerance_m)
    return label,valid


class AngleOnlyBelief(nn.Module):
    def __init__(self, *, text_dim=768, hidden=128, bins=8,
                 concentration=4.0, max_logit_delta=1.0):
        super().__init__()
        if hidden<8 or bins<4 or bins%2 or concentration<=0 or max_logit_delta<=0:
            raise ValueError("Invalid angular modality parameters")
        self.bins=int(bins)
        self.concentration=float(concentration)
        self.max_logit_delta=float(max_logit_delta)
        self.sentence=nn.Sequential(nn.Linear(text_dim,hidden),nn.LayerNorm(hidden))
        self.landmark=nn.Sequential(nn.Linear(text_dim,hidden),nn.LayerNorm(hidden))
        self.direction_head=nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,bins),
        )
        # Exact 0 on initialization preserves the COMPLETE full-text baseline.
        self.fusion_gain=nn.Parameter(torch.zeros(()))
        self.register_buffer("bin_angles",torch.arange(bins,dtype=torch.float32)
                             * (2*math.pi/bins),persistent=False)

    def predict_angle(self, *, tokens, token_mask, node_emb,
                      landmark_valid):
        """Predicted K-bin direction distribution per named public landmark.

        No target label, target coordinate, target-to-anchor difference,
        or angle oracle argument exists in this method.
        """
        if (tokens.ndim!=3 or node_emb.ndim!=3 or
                token_mask.shape!=tokens.shape[:2] or
                landmark_valid.shape!=node_emb.shape[:2] or
                tokens.shape[0]!=node_emb.shape[0] or
                tokens.shape[-1]!=self.sentence[0].in_features or
                node_emb.shape[-1]!=self.landmark[0].in_features):
            raise ValueError("Invalid frozen BERT / public anchor features")
        mask=token_mask.bool()
        if not mask.any(-1).all():
            raise ValueError("Full instruction must contain valid BERT tokens")
        # Token-level pooling retains all original full instruction words.
        word=(tokens.float()*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        language=self.sentence(word)
        reference=self.landmark(node_emb.float())
        return self.direction_head(torch.cat((
            language[:,None,:].expand_as(reference),reference),-1))

    def apply_angle(self, *, base_logits, candidate_bearings,
                    landmark_valid, bearing_probabilities):
        """Angular compatibility ONLY, no displacement/radius in new channel.

        Bearing class 0 is +X, classes increase counterclockwise; these are
        relative to the public planar map axes, not verified geographic north.
        """
        if (base_logits.ndim!=2 or candidate_bearings.ndim!=3 or
                landmark_valid.ndim!=2 or bearing_probabilities.ndim!=3):
            raise ValueError("Invalid angular fusion tensor ranks")
        b,q=base_logits.shape
        m=candidate_bearings.shape[2]
        if (candidate_bearings.shape!=(b,q,m) or
                landmark_valid.shape!=(b,m) or
                bearing_probabilities.shape!=(b,m,self.bins)):
            raise ValueError("Misaligned angle/candidate/anchor tensors")
        if not torch.isfinite(candidate_bearings).all():
            raise ValueError("Nonfinite public bearing")
        # Conditional kernel for each candidate direction and each bin.
        kernel=F.softmax(
            self.concentration*torch.cos(
                candidate_bearings.float()[...,None]-
                self.bin_angles[None,None,None,:]),
            dim=-1,
        )
        score=(kernel*bearing_probabilities.float()[:,None,:,:]).sum(-1)
        valid=landmark_valid.bool()
        weighted=(score*valid[:,None,:]).sum(-1)/valid.sum(-1).clamp_min(1)[:,None]
        # A uniform angular distribution produces no distinctive residual.
        centered=weighted-weighted.mean(-1,keepdim=True)
        supported=valid.any(-1)
        delta=(self.max_logit_delta*torch.tanh(
            self.fusion_gain*centered))*supported[:,None].float()
        return dict(logits=base_logits.float()+delta,delta=delta,
                    supported=supported)

    def forward(self, *, base_logits, candidate_bearings, landmark_valid,
                tokens, token_mask, node_emb):
        predicted=self.predict_angle(tokens=tokens,token_mask=token_mask,
                                     node_emb=node_emb,
                                     landmark_valid=landmark_valid)
        probs=predicted.softmax(-1)
        result=self.apply_angle(
            base_logits=base_logits,candidate_bearings=candidate_bearings,
            landmark_valid=landmark_valid,bearing_probabilities=probs)
        result["angle_logits"]=predicted
        result["angle_probs"]=probs
        return result
