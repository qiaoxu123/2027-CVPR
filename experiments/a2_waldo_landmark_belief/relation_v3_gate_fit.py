"""One complete FIT84 choice, then one complete VERIFY209 evaluation."""
import math
import numpy as np
from .calibration_protocol import calibration_fold
from .direct_risk_calibrate import fit_choice,evaluate_risk


def validate_groups(packets,expected,fold,count):
    if len(expected)!=count:raise ValueError('Complete fixed '+fold+' target inventory required')
    keys=[];observations=set()
    for p in packets:
        key=tuple(p['group']);identity=(p['episode'],p['step'])
        if p['fold']!=fold or calibration_fold(*key)!=fold or key not in expected:raise ValueError('Cross-role target entered '+fold)
        if identity in observations:raise ValueError('Duplicate paired observation')
        if not all(math.isfinite(float(p[k])) and p[k]>=0 for k in ('before_ne','action_ne')):raise ValueError('Invalid offline distance')
        scores=np.asarray(p['scores'],dtype=float)
        if scores.shape!=(3,) or not np.isfinite(scores).all() or (scores<0).any() or not np.isclose(scores.sum(),1.,atol=1e-6,rtol=1e-6):
            raise ValueError('Invalid actual threeclass scores')
        if (p['changed'] and not p['available']) or (not p['changed'] and p['action_ne']!=p['before_ne']):raise ValueError('Inconsistent fixed-action availability/outcome')
        observations.add(identity);keys.append(key)
    if set(keys)!=set(expected):raise ValueError('Partial '+fold+' target population')


def validate_parameters(parameters):
    if parameters is None:raise ValueError('No FIT choice; VERIFY must remain unscored')
    if set(parameters)!= {'minimum_gain','maximum_harm','temperature'}:raise ValueError('Fixed T1 score-gate schema required')
    if (parameters['temperature']!=1. or not all(math.isfinite(float(v)) for v in parameters.values())
        or not 0<=parameters['minimum_gain']<=1 or not 0<=parameters['maximum_harm']<=1):raise ValueError('Invalid frozen FIT thresholds')


def choose_fit(packets,expected_groups):
    validate_groups(packets,expected_groups,'fit',84)
    choices,chosen=fit_choice(packets)
    for point in choices:point['parameters']=dict(point['parameters'],temperature=1.)
    if chosen is not None:validate_parameters(chosen['parameters'])
    return choices,chosen


def evaluate_verify(packets,parameters,expected_groups):
    validate_parameters(parameters)  # Must precede ANY heldout iteration.
    validate_groups(packets,expected_groups,'verify',209)
    result=evaluate_risk(packets,parameters)
    if result['groups']!=209:raise ValueError('Incomplete independent VERIFY aggregate')
    return result
