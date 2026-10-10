"""Small matched navigation heads over immutable cached A3/SigLIP outputs.

Destination, future waypoints and ROI identity supervision are loss-only inputs.
Unknown grid cells retain their original static scores exactly.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from graphnav.appearance_matching import MultiScaleAppearanceMatcher
from graphnav.trajectory_policy import CandidateTrajectoryStop, trajectory_losses


@dataclass
class CoreOutput:
    belief_logits: torch.Tensor
    candidate_ids: torch.Tensor
    trajectory: object
    roi_logits: torch.Tensor
    roi_valid: torch.Tensor
    observed: torch.Tensor


class CoreModel(nn.Module):
    PUBLIC_KEYS = {'static_logits', 'static_context', 'grid_xy', 'bounds',
                   'pose_world', 'ground_level', 'native_text', 'roi_features',
                   'roi_grid_ids', 'roi_valid', 'roi_quality'}

    def __init__(self, hidden=128, top_k=20, vision_dim=1024):
        super().__init__()
        self.hidden, self.top_k = hidden, top_k
        self.matcher = MultiScaleAppearanceMatcher(vision_dim, hidden=64)
        self.visual_gain = nn.Parameter(torch.zeros(()))
        self.trajectory = CandidateTrajectoryStop(hidden=hidden)

    def forward(self, public):
        if set(public) != self.PUBLIC_KEYS:
            raise ValueError('public contract mismatch; private labels forbidden: '+str(set(public)^self.PUBLIC_KEYS))
        logits = public['static_logits'].float()
        b, q = logits.shape
        context, grid = public['static_context'], public['grid_xy']
        if context.shape != (b, q, self.hidden) or grid.shape != (b, q, 2):
            raise ValueError('frozen candidate alignment mismatch')
        ids, valid = public['roi_grid_ids'].long(), public['roi_valid'].bool()
        if ids.shape != valid.shape or ((ids[valid] < 0) | (ids[valid] >= q)).any():
            raise ValueError('invalid observed grid association')
        safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
        scores, known_roi, _ = self.matcher(public['native_text'],
            public['roi_features'][:, :, None], valid[:, :, None],
            public['roi_quality'][:, :, None])
        sums = torch.zeros_like(logits).scatter_add(1, safe_ids, scores * known_roi)
        counts = torch.zeros_like(logits).scatter_add(1, safe_ids, known_roi.float())
        known = counts > 0
        cells = sums / counts.clamp_min(1)
        centered = cells - (cells * known).sum(-1, keepdim=True) / known.sum(-1, keepdim=True).clamp_min(1)
        updated = logits + torch.tanh(self.visual_gain) * centered * known
        candidates = updated.topk(min(self.top_k, q), -1).indices
        candidate_xy = grid.gather(1, candidates[..., None].expand(-1, -1, 2))
        candidate_context = context.gather(1, candidates[..., None].expand(-1, -1, self.hidden))
        probabilities = updated.softmax(-1)
        trajectory = self.trajectory(candidate_xy=candidate_xy,
            candidate_context=candidate_context,
            candidate_logits=updated.gather(1, candidates),
            pose_world=public['pose_world'], bounds=public['bounds'],
            ground_level=public['ground_level'],
            belief_entropy=-(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1)/math.log(max(q, 2)),
            observed_fraction=known.float().mean(-1))
        return CoreOutput(updated, candidates, trajectory, scores, known_roi, known)


def core_losses(output, public, labels, *, vl_weight=0.0):
    """Navigation targets and conservative multi-positive identities are private."""
    bounds = public['bounds']
    grid = bounds[:, None, :2] + public['grid_xy'] * (bounds[:, None, 2:] - bounds[:, None, :2])
    candidates = grid.gather(1, output.candidate_ids[..., None].expand(-1, -1, 2))
    # Metric-coordinate loss reductions are FP32 even under BF16 head inference.
    for key in ('paths_world', 'mode_logits', 'candidate_logits', 'stop_logits'):
        setattr(output.trajectory, key, getattr(output.trajectory, key).float())
    nav = trajectory_losses(output.trajectory, candidate_world_xy=candidates,
        pose_world=public['pose_world'], target_world=labels['target_world'],
        teacher_world=labels['teacher_world'], valid_teacher_z=labels['valid_teacher_z'],
        stop_pos_weight=torch.tensor(20., device=grid.device))
    positives = labels['positive_roi_mask'].bool() & output.roi_valid
    judged = labels['semantic_valid_mask'].bool() & output.roi_valid
    if (positives & ~judged).any():
        raise ValueError('positive identity ROI must be a judged valid ROI')
    usable = positives.any(-1) & (judged & ~positives).any(-1)
    scores = output.roi_logits.float()
    numerator = torch.logsumexp(scores.masked_fill(~positives, -1e4), -1)
    denominator = torch.logsumexp(scores.masked_fill(~judged, -1e4), -1)
    vl = (denominator - numerator)[usable].mean() if usable.any() else scores.sum()*0
    return dict(total=nav['total']+vl_weight*vl, navigation=nav['total'],
                visual_language=vl, visual_language_eligible=usable.sum(),
                **{k:v for k,v in nav.items() if k != 'total'})


def collate_core(rows):
    """Pad observed ROI lists, retaining explicit validity and unknown labels."""
    public = {}
    ragged = {'roi_features', 'roi_grid_ids', 'roi_valid', 'roi_quality'}
    for key in CoreModel.PUBLIC_KEYS - ragged:
        public[key] = torch.stack([torch.as_tensor(r['public'][key]) for r in rows])
    k = max(1, max(len(r['public']['roi_grid_ids']) for r in rows))
    dim = rows[0]['public']['native_text'].shape[-1]
    public.update(roi_features=torch.zeros(len(rows), k, dim),
        roi_grid_ids=torch.zeros(len(rows), k, dtype=torch.long),
        roi_valid=torch.zeros(len(rows), k, dtype=torch.bool),
        roi_quality=torch.zeros(len(rows), k, 3))
    labels = {key:torch.stack([torch.as_tensor(r['labels'][key]) for r in rows])
              for key in ('target_world', 'teacher_world', 'valid_teacher_z')}
    labels.update(positive_roi_mask=torch.zeros(len(rows), k, dtype=torch.bool),
                  semantic_valid_mask=torch.zeros(len(rows), k, dtype=torch.bool))
    for i, row in enumerate(rows):
        n = len(row['public']['roi_grid_ids'])
        for key in ragged:
            public[key][i, :n] = torch.as_tensor(row['public'][key])
        for key in ('positive_roi_mask', 'semantic_valid_mask'):
            if key in row['labels']:
                labels[key][i, :n] = torch.as_tensor(row['labels'][key])
    metadata = [r.get('metadata', {}) for r in rows]
    return public, labels, metadata
