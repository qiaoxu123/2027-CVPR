"""TRAIN-selection-only thresholds and one fixed internal-DEV assessment."""
from copy import deepcopy
from numbers import Integral

import numpy as np

from experiments.a2_waldo_landmark_belief.direct_risk_protocol import (
    group_effects, operating_points, select_point,
)
from .linear_fit import training_partition


def _roles(identities, role):
    if not identities:
        raise ValueError('Empty diagnostic population')
    for row in identities:
        if (row.get('role') != role or not isinstance(row.get('map_name'), str)
                or not row['map_name'] or '\0' in row['map_name']
                or not isinstance(row.get('object_id'), Integral)
                or isinstance(row['object_id'], bool)):
            raise ValueError('Unexpected diagnostic population identity/role')


def _arrays(identities, scores, available, before, after):
    scores, available = np.asarray(scores), np.asarray(available)
    n = len(identities)
    if (scores.shape != (n, 3) or available.shape != (n,)
            or available.dtype != np.bool_ or not np.isfinite(scores).all()
            or (scores < 0).any() or not np.allclose(scores.sum(-1), 1., atol=1e-6)):
        raise ValueError('Invalid diagnostic scores/availability')
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    group_effects(identities, before, after, np.zeros(n, dtype=bool))
    return scores, available, before, after


def _passes(summary):
    return bool(summary['supported_groups'] >= 30 and summary['gain_groups'] >= 5
                and summary['mean_r1_difference'] > 0
                and summary['mean_ne_difference_m'] <= 0
                and summary['broken_group_rate'] is not None
                and summary['broken_group_rate'] <= .02)


def choose_condition(*, identities, scores, available, before, after):
    """No DEV parameter exists: choose once using only TRAIN held-out groups."""
    _roles(identities, 'train')
    if training_partition(identities).any():
        raise ValueError('Fitting groups cannot select diagnostic thresholds')
    scores, available, before, after = _arrays(identities, scores, available, before, after)
    points = operating_points(scores, available, identities, before, after)
    choice = select_point(points, dev=True)  # Reuse strict30-group/5-gain rule.
    if choice is None:
        return None
    result = deepcopy(choice)
    result['selection_targets'] = [list(key) for key in sorted(
        {(row['map_name'], int(row['object_id'])) for row in identities})]
    return result


def assess_condition(choice, *, identities, scores, available, before, after):
    """Apply each model's TRAIN-chosen condition unchanged; never search on DEV."""
    _roles(identities, 'dev')
    scores, available, before, after = _arrays(identities, scores, available, before, after)
    allow = np.zeros(len(identities), dtype=bool)
    parameters = None
    if choice is not None:
        selected = {tuple(key) for key in choice['selection_targets']}
        current = {(row['map_name'], int(row['object_id'])) for row in identities}
        if selected & current:
            raise ValueError('Selection targets overlap internal DEV')
        parameters = deepcopy(choice['parameters'])
        if set(parameters) != {'minimum_gain', 'maximum_harm'} or any(
                not np.isfinite(v) or not 0 <= v <= 1 for v in parameters.values()):
            raise ValueError('Invalid fixed diagnostic thresholds')
        allow = (available & (scores[:, 1] >= parameters['minimum_gain'])
                 & (scores[:, 2] <= parameters['maximum_harm']))
    summary = group_effects(identities, before, after, allow)
    by_map, by_target = {}, []
    maps = sorted({row['map_name'] for row in identities})
    targets = sorted({(row['map_name'], int(row['object_id'])) for row in identities})

    def subset(mask):
        rows = [row for row, keep in zip(identities, mask) if keep]
        return group_effects(rows, before[mask], after[mask], allow[mask])

    for name in maps:
        by_map[name] = subset(np.array([row['map_name'] == name for row in identities]))
    for name, target in targets:
        mask = np.array([row['map_name'] == name and row['object_id'] == target for row in identities])
        by_target.append(dict(map_name=name, object_id=target, summary=subset(mask)))
    return dict(parameters=parameters, summary=summary, by_map=by_map, by_target=by_target,
                passes_internal=choice is not None and _passes(summary),
                status='fixed_internal_assessment' if choice is not None else 'no_safe_train_condition')
