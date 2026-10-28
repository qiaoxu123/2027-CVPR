"""Audit actual V3 inference records, then score targets privately offline."""
import torch
from .relation_v3_fusion import proposal_with_quality
from .revision_fusion import batch_belief_mix
from .risk_probe import public_risk_features
from .calibration_protocol import calibration_fold


def close(actual,expected,name):
    if (actual.shape!=expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
        or not torch.allclose(actual.float(),expected.float(),atol=1e-6,rtol=1e-5)):
        raise ValueError('Actual inference record does not reproduce: '+name)


@torch.no_grad()
def make_packet(row,model):
    spatial=row['public']['spatial'];out=row['prediction'];risk=row['risk']
    base=spatial['base_logits'].float();grid=spatial['grid_world'].float();q=risk['q'].float()
    if row['fold'] not in ('fit','verify') or calibration_fold(*row['target_instance'])!=row['fold']:raise ValueError('Offline target fold changed')
    if base.ndim!=1 or grid.shape!=(len(base),2) or q.shape!=base.shape or not torch.isfinite(base).all() or not torch.isfinite(grid).all():raise ValueError('Invalid public field axes')
    if risk['available'].ndim or risk['available'].dtype!=torch.bool:raise ValueError('Invalid actual proposal availability')
    available=bool(risk['available'])
    if not torch.isfinite(q).all() or (q<0).any() or (available and not torch.isclose(q.sum(),q.new_tensor(1.),atol=1e-6,rtol=1e-6)):
        raise ValueError('Actual proposal is not finite normalized nonnegative mass')
    if not available and q.any():raise ValueError('Unavailable proposal must be zero')
    recomputed,valid,quality=proposal_with_quality({k:v[None] for k,v in spatial.items()},out['scores'][None],out['candidate_match_logits'][None],out['reliable'][None])
    if bool(valid[0])!=available:raise ValueError('Actual proposal availability differs from public evidence')
    close(q,recomputed[0],'stable observed4m proposal');close(risk['quality'].reshape(1),quality,'evidence match quality')
    allowed=(spatial['candidate_support'].bool()&out['reliable'][:,None].bool()).any(0)
    if q[~allowed].any():raise ValueError('Proposal contains mass outside reliable observed support')
    features=public_risk_features(base[None],q[None],grid[None],risk['quality'].reshape(1),risk['available'].reshape(1))
    close(risk['features'].reshape(1,-1),features,'public risk features')
    logits=model.predict_risk(risk['features'].reshape(1,-1));close(risk['head_logits'],logits[0],'frozen risk head')
    corrected=model.corrected_risk_logits(risk['head_logits'][None])[0]
    close(risk['corrected_logits'],corrected,'class-bias corrected logits')
    scores=risk['scores'].float()
    if (scores.shape!=(3,) or not torch.isfinite(scores).all() or (scores<0).any()
        or not torch.isclose(scores.sum(),scores.new_tensor(1.),atol=1e-6,rtol=1e-6)):raise ValueError('Invalid actual T1 scores')
    close(scores,corrected.softmax(-1),'actual T1scores')
    fixed=risk['fixed_action_logits'].float()
    expected=batch_belief_mix(base[None],q[None],torch.tensor([.25],device=base.device))[0]
    close(fixed,expected,'actual fixed.25 action')
    target=row['labels']['target_xy'].float()
    if target.shape!=(2,) or not torch.isfinite(target).all():raise ValueError('Invalid private offline target')
    distance=(grid-target).norm(dim=-1)
    # Saved GPU scores and fixed action remain authoritative. CPU recomputations
    # above are integrity screens, never replacement scores/actions or labels.
    return dict(group=row['target_instance'],fold=row['fold'],episode=row['episode'],step=row['step'],
        scores=scores.clone(),available=available,changed=bool(available and not torch.equal(base,fixed)),
        before_ne=float(distance[base.argmax()]),action_ne=float(distance[fixed.argmax()]),
        inference_authority='saved actual V3 GPU T1 scores and fixed.25 action; CPU integrity checked')
