"""Fixed CPU-only linear probe; never selects a live A2 policy."""
import hashlib
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


CLASS_WEIGHTS = (1., 8., 6.)


def training_partition(identities):
    """TRAIN target-group hash; metadata controls isolation, not model input."""
    result = []
    for row in identities:
        if row.get('role') != 'train':
            raise ValueError('Only original TRAIN identities may enter fitting')
        name, target = row.get('map_name'), row.get('object_id')
        if (not isinstance(name, str) or not name or '\0' in name
                or not isinstance(target, Integral) or isinstance(target, bool)):
            raise ValueError('Invalid target-group identity')
        payload = f'a2-evidence-quality-seed0\0{name}\0{int(target)}'.encode()
        fraction = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') / 2**64
        result.append(fraction < .8)
    return torch.tensor(result, dtype=torch.bool)


def _features(features):
    if (not torch.is_tensor(features) or features.device.type != 'cpu'
            or features.ndim != 2 or features.shape[1] != 22
            or not torch.is_floating_point(features) or not torch.isfinite(features).all()):
        raise ValueError('Expected finite CPU public features [observations,22]')
    return features.detach().float()


def fit_linear(features, outcomes, identities, *, width):
    """Fit exactly20 full-batch steps using only the target-disjoint fit rows."""
    fit = training_partition(identities)  # Role refusal precedes tensor access.
    x = _features(features)
    if (width not in (15, 22) or len(x) != len(fit) or not fit.any() or fit.all()
            or not torch.is_tensor(outcomes) or outcomes.device.type != 'cpu'
            or outcomes.shape != (len(x),) or outcomes.dtype != torch.long
            or ((outcomes < 0) | (outcomes > 2)).any()):
        raise ValueError('Invalid fixed-probe population, outcomes or feature width')
    selected = x[fit, :width]
    mean = selected.mean(0)
    scale = selected.std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    z = (selected - mean) / scale
    y = outcomes[fit]
    history = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = nn.Linear(width, 3)
        # Equal zero starts prevent feature count from changing the baseline.
        with torch.no_grad():
            model.weight.zero_(); model.bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        weights = torch.tensor(CLASS_WEIGHTS)
        for epoch in range(1, 21):
            loss = F.cross_entropy(model(z), y, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CPU probe training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            history.append(dict(epoch=epoch, loss=float(loss.detach())))
    return dict(width=width, mean=mean.clone(), scale=scale.clone(),
                weight=model.weight.detach().clone(), bias=model.bias.detach().clone(),
                history=history, fit_rows=int(fit.sum()), selection_rows=int((~fit).sum()),
                initialization='zero', seed=0, epochs=20)


@torch.no_grad()
def predict(state, features):
    """Class-weight-corrected probabilities; no fitting or private outcomes."""
    x = _features(features)
    width = state['width']
    if width not in (15, 22):
        raise ValueError('Unsupported fixed-probe feature width')
    for name, shape in (('mean', (width,)), ('scale', (width,)),
                        ('weight', (3, width)), ('bias', (3,))):
        t = state[name]
        if (not torch.is_tensor(t) or t.device.type != 'cpu' or tuple(t.shape) != shape
                or not torch.isfinite(t).all()):
            raise ValueError('Invalid CPU probe state')
    if (state['scale'] <= 0).any():
        raise ValueError('Normalization scale must be positive')
    z = (x[:, :width] - state['mean']) / state['scale']
    logits = F.linear(z, state['weight'], state['bias'])
    result = (logits - torch.tensor(CLASS_WEIGHTS).log()).softmax(-1)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite CPU probe predictions')
    return result
