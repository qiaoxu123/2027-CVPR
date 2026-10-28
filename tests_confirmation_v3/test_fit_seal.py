import json
import pytest
from experiments.a2_waldo_landmark_belief.paths import sha
from experiments.a2_waldo_landmark_belief.relation_v3_calibrate import seal_fit_decision,restore_fit_seal
from tests_confirmation_v3.test_calibration_data import create_fit_cache


def test_fit_seal_cannot_be_replaced_by_another_choice_and_recovers_partial_commit(tmp_path):
    create_fit_cache(tmp_path);output=tmp_path/'calibration'
    analysis=dict(checkpoint_sha256='checkpoint',roles_sha256='roles')
    # No-candidate is a real terminal choice, not permission to use a runner-up.
    decision=seal_fit_decision(tmp_path,analysis,[],None);old=sha(output/'fit_decision.json')
    (output/'fit_decision_seal.json').unlink();restore_fit_seal(tmp_path,analysis,decision)
    assert sha(output/'fit_decision.json')==old and (output/'fit_decision_seal.json').exists()
    different=dict(index=0,parameters=dict(minimum_gain=.2,maximum_harm=.1,temperature=1.))
    with pytest.raises(ValueError):seal_fit_decision(tmp_path,analysis,[different],different)
