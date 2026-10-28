"""Separate public CPU inference from private offline outcome attachment."""
import torch

from experiments.a2_waldo_landmark_belief.relation_v3_data import NETWORK_KEYS
from experiments.a2_waldo_landmark_belief.reference_rebuild import PUBLIC_SPATIAL_KEYS
from experiments.a2_waldo_landmark_belief.relation_v3_runtime import evidence_forward
from .public_features import public_features


@torch.no_grad()
def public_forward(model, public):
    """No private labels accepted; retain original fixed action and features."""
    required_spatial = {'base_logits', 'grid_world', 'candidate_xy', 'candidate_support'}
    if (set(public) != {'network', 'spatial'}
            or set(public['network']) != NETWORK_KEYS | {'box_kind'}
            or not required_spatial <= set(public['spatial']) <= PUBLIC_SPATIAL_KEYS):
        raise ValueError('Unexpected or missing public-only inference fields')
    if model.training or any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Diagnostic requires an evaluation-only CPU model')
    tensors = (*public['network'].values(), *public['spatial'].values())
    if any(not torch.is_tensor(t) or t.device.type != 'cpu' for t in tensors):
        raise ValueError('Diagnostic public inputs must be CPU tensors')
    network = {k:v[None] for k,v in public['network'].items()}
    spatial = {k:v[None] for k,v in public['spatial'].items()}
    output = model(**network)
    packet = evidence_forward(spatial, output, model)
    features = public_features(
        base=spatial['base_logits'], grid=spatial['grid_world'],
        candidate_xy=spatial['candidate_xy'], candidate_support=spatial['candidate_support'],
        scores=output['scores'], match_logits=output['candidate_match_logits'],
        reliable=output['reliable'], box_valid=network['box_valid'],
        reference_valid=output['reference_valid'], pair_valid=network['pair_valid'],
        pair_geometry=network['pair_geometry'])
    if not torch.equal(features[:, :15], packet['features']):
        raise ValueError('Original15 feature values differ from actual actor inference')
    base = spatial['base_logits'][0]
    fixed = packet['fixed_action_logits'][0]
    available = packet['available'][0]
    if not available and not torch.equal(base, fixed):
        raise ValueError('Unavailable proposal modified original A0 action')
    grid = spatial['grid_world'][0]
    return dict(features=features[0].clone(), available=available.clone(),
                before_xy=grid[base.argmax()].clone(), after_xy=grid[fixed.argmax()].clone())


def private_outcomes(public_result, target_xy):
    """Attach distances only after the caller has completed public inference."""
    if (not torch.is_tensor(target_xy) or target_xy.device.type != 'cpu'
            or target_xy.shape != (2,) or not torch.isfinite(target_xy).all()):
        raise ValueError('Invalid private offline target')
    before = float((public_result['before_xy'].float() - target_xy.float()).norm())
    after = float((public_result['after_xy'].float() - target_xy.float()).norm())
    gain, harm = before > 20 and after <= 20, before <= 20 and after > 20
    return dict(before=before, after=after, outcome=int(gain) + 2*int(harm))
