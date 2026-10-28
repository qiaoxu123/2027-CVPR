import pytest
import torch
from experiments.a2_waldo_landmark_belief.paths import A0_SHA
from experiments.a2_waldo_landmark_belief.relation_v3_provenance import checked_state,validate_gate


def test_wrong_baseline_or_untrained_checkpoint_cannot_enter_formal_runtime(tmp_path):
    path=tmp_path/'bad.pt'
    torch.save(dict(a0_sha256='wrong',config=dict(version='a2_relation_v3')),path)
    with pytest.raises(ValueError,match='lineage'):checked_state(path)
    torch.save(dict(a0_sha256=A0_SHA,config=dict(version='a2_relation_v3'),checkpoint_eligible=False,scope='TRAIN_functional_only'),path)
    with pytest.raises(ValueError,match='eligible'):checked_state(path)
    with pytest.raises(ValueError,match='functional'):checked_state(path,functional_only=True)


def test_formal_gate_cannot_be_forced_on_without_independent_verification():
    with pytest.raises(ValueError,match='[Ii]ndependent'):
        validate_gate(dict(enabled=True,minimum_gain=0.,maximum_harm=1.,temperature=1.),dict(roles_sha256='x',code_sha256={}))
    validate_gate(dict(enabled=False),dict(roles_sha256='x',code_sha256={}))


def test_formal_gate_rejects_smaller_actual_population_despite_declared_209():
    from experiments.a2_waldo_landmark_belief.calibration_protocol import verify_gate
    stats=dict(groups=208,supported_groups=50,a0_correct_groups=100,gain_groups=6,gain_ci95=[.001,.1],
        broken_group_wilson_upper95=.04,mean_ne_difference_m=-.1)
    parameters=dict(enabled=True,fit_groups=84,verify_groups=209,roles_sha256='x',training_code_sha256={},verification=stats,verify_gate=verify_gate(stats))
    with pytest.raises(ValueError,match='[Ii]ndependent'):validate_gate(parameters,dict(roles_sha256='x',code_sha256={}))
