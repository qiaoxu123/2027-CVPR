"""Catch DEV tuning, relaxed support requirements and unsafe fixed conditions."""
import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module('diagnostics.evidence_quality.conditions')


SELECTION_IDS = [5,7,8,17,30,32,37,38,43,46,56,57,60,63,65,68,78,94,96,102,
                 103,116,120,123,124,137,139,142,150,158,159,163,168,170,171,
                 172,173,179,182,183]


def population(role='train'):
    ids = [dict(role=role, map_name='map' if role=='train' else 'heldout', object_id=i)
           for i in SELECTION_IDS]
    before = np.full(40, 10.); before[:6] = 30.
    return dict(identities=ids, scores=np.tile([.1,.8,.1],(40,1)),
                available=np.ones(40,dtype=bool), before=before, after=np.full(40,10.))


def test_condition_selected_from_train_selection_and_reused_without_dev_tuning():
    choice = api().choose_condition(**population())
    assert choice is not None
    assert choice['summary']['supported_groups'] == 40
    assert choice['summary']['gain_groups'] == 6
    dev = population('dev'); dev['after'][6:8] = 30.
    report = api().assess_condition(choice, **dev)
    assert report['summary']['corrected'] == 6
    assert report['summary']['broken'] == 2
    assert report['passes_internal'] is False
    assert report['parameters'] == choice['parameters']
    assert report['by_map']['heldout']['groups'] == 40
    assert len(report['by_target']) == 40


def test_less_than_30_supported_groups_does_not_relax_rule():
    p = {k:v[:29] for k,v in population().items()}
    assert api().choose_condition(**p) is None


@pytest.mark.parametrize('role', ['dev','calibration_fit','val_unseen'])
def test_nontrain_population_cannot_choose_conditions(role):
    with pytest.raises(ValueError):
        api().choose_condition(**population(role))


def test_fitting_groups_cannot_be_used_as_selection_groups():
    p = population(); p['identities'][0]['object_id'] = 0
    with pytest.raises(ValueError):
        api().choose_condition(**p)


def test_no_condition_preserves_all_baseline_outcomes():
    report = api().assess_condition(None, **population('dev'))
    assert report['summary']['corrected'] == report['summary']['broken'] == 0
    assert report['summary']['supported_groups'] == 0
    assert report['passes_internal'] is False


def test_changed_condition_nonfinite_threshold_rejected():
    choice = api().choose_condition(**population())
    choice['parameters']['minimum_gain'] = float('nan')
    with pytest.raises(ValueError):
        api().assess_condition(choice, **population('dev'))


def test_invalid_scores_rejected_even_when_no_candidates_available():
    p = population(); p['available'][:] = False; p['scores'][0,0] = float('nan')
    with pytest.raises(ValueError):
        api().choose_condition(**p)
