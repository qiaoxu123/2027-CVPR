"""Single FIT choice and honest disabled outcome for the V4 branch."""
import copy
import importlib
import json
import pytest
from experiments.a2_waldo_landmark_belief.paths import sha,save
from experiments.a2_waldo_landmark_belief.calibration_protocol import verify_gate
from tests_confirmation_v4.test_calibration_data import source_fixture


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_calibrate')


def fitted_fixture(tmp_path,monkeypatch):
    data,root,state=source_fixture(tmp_path,monkeypatch);source=data.FreshConfirmation(root)
    keys=('code_sha256','roles_sha256','feature_manifest_sha256','feature_signature_sha256',
          'feature_build_status_sha256','initial_checkpoint_sha256')
    analysis=dict({key:state[key] for key in keys},checkpoint_sha256=source.checkpoint_sha256,
        implementation=data.calibration_code_hashes(),confirmation_manifest_sha256=sha(root/'confirmation/manifest.json'),
        sealed_verify_sha256=json.loads((root/'confirmation_population.json').read_text())['sealed_verify_sha256'],
        training_verification_sha256=sha(root/'training/artifact_verification.json'))
    save(analysis,root/'calibration/analysis_signature.json');source.packets('fit')
    return source,analysis


def test_no_candidate_seal_is_immutable_and_recovers_missing_seal_file(tmp_path,monkeypatch):
    source,analysis=fitted_fixture(tmp_path,monkeypatch);m=module();root=source.root
    decision=m.seal_fit_decision(root,analysis,[],None);original=sha(root/'calibration/fit_decision.json')
    (root/'calibration/fit_decision_seal.json').unlink();m.restore_fit_seal(root,analysis,decision)
    assert sha(root/'calibration/fit_decision.json')==original
    point=dict(index=0,parameters=dict(minimum_gain=.2,maximum_harm=.1,temperature=1.))
    with pytest.raises(ValueError):m.seal_fit_decision(root,analysis,[point],point)


def test_no_candidate_result_preserves_feature_binding_and_cannot_enable(tmp_path,monkeypatch):
    source,analysis=fitted_fixture(tmp_path,monkeypatch);m=module();monkeypatch.setattr(m,'ROOT',tmp_path)
    decision=m.seal_fit_decision(source.root,analysis,[],None)
    value=m.result_record(source,analysis,decision,None,seconds=0.)
    assert value['enabled'] is False and value['verification'] is None
    assert value['feature_manifest_sha256']==source.state['feature_manifest_sha256']
    assert not value['verification_performance_evaluated']
    assert not (source.root/'calibration/packets_fresh_verify').exists()
    m.validate_existing(source,analysis)
    for key,change in [('enabled',True),('verification_performance_evaluated',True),('minimum_gain',.1)]:
        changed=copy.deepcopy(value);changed[key]=change
        with pytest.raises(ValueError):m.validate_result_contract(changed,decision)
    path=source.root/'calibration/candidate_calibration.json';changed=copy.deepcopy(value)
    changed['feature_manifest_sha256']='9'*64;save(changed,path)
    with pytest.raises(ValueError):m.validate_existing(source,analysis)


def test_failed_independent_check_cannot_be_reported_as_success():
    selected=dict(minimum_gain=.9,maximum_harm=.1,temperature=1.)
    stats=dict(groups=209,supported_groups=30,a0_correct_groups=100,gain_groups=5,
        gain_ci95=[0.,.1],broken_group_wilson_upper95=.02,mean_ne_difference_m=-.1)
    result=dict(status='v4_VERIFY_rejected_no_forced_enable',enabled=False,gate_final=False,
        fixed_action=.25,temperature=1.,temperature_fitting=False,benchmark_used=False,no_verify_reselection=True,
        fit_groups=84,verify_groups=209,fit_selected_parameters=selected,verification=stats,
        verification_performance_evaluated=True,verify_gate=verify_gate(stats),minimum_gain=1.,maximum_harm=0.)
    decision=dict(selected_parameters=selected);m=module();m.validate_result_contract(result,decision)
    wrong=copy.deepcopy(result);wrong.update(enabled=True,minimum_gain=.9,maximum_harm=.1)
    with pytest.raises(ValueError):m.validate_result_contract(wrong,decision)
    wrong=copy.deepcopy(result);wrong['verification']['groups']=208
    with pytest.raises(ValueError):m.validate_result_contract(wrong,decision)


def test_analysis_signature_requires_current_cpu_proof_and_binds_all_new_features(tmp_path,monkeypatch):
    data,root,state=source_fixture(tmp_path,monkeypatch);source=data.FreshConfirmation(root)
    from experiments.a2_waldo_landmark_belief import reference_v4_confirmation_preparation as preparation
    monkeypatch.setattr(preparation,'BASE',root.parent)
    save(dict(status='original_gate_fixture'),root/'cpu_gates.json')
    log=root/'posttraining_cpu/cpu_tests.log';log.parent.mkdir();log.write_text('628 passed in 1s\n')
    proof=dict(preparation.cpu_signature(root),status='v4_posttraining_CPU_gates_passed',test_log_sha256=sha(log),
        unit_test_summary='628 passed in 1s',no_CUDA_used=True,no_new_confirmation_data_consumed=True)
    save(proof,root/'posttraining_cpu_gates.json')
    m=module();analysis=m.analysis_signature(source)
    for key in ('feature_manifest_sha256','feature_signature_sha256','feature_build_status_sha256','initial_checkpoint_sha256'):
        assert analysis[key]==state[key]
    assert analysis['checkpoint_sha256']==source.checkpoint_sha256
    assert analysis['posttraining_cpu_gate_sha256']==sha(root/'posttraining_cpu_gates.json')
    log.write_text('changed test log')
    with pytest.raises(ValueError):m.analysis_signature(source)


def test_passing_independent_check_requires_pilot_and_does_not_claim_final_gate():
    selected=dict(minimum_gain=.9,maximum_harm=.1,temperature=1.)
    stats=dict(groups=209,supported_groups=30,a0_correct_groups=100,gain_groups=5,
        gain_ci95=[.01,.1],broken_group_wilson_upper95=.02,mean_ne_difference_m=-.1)
    result=dict(status='v4_candidate_requires_real_closed_loop_pilot',enabled=True,gate_final=False,
        fixed_action=.25,temperature=1.,temperature_fitting=False,benchmark_used=False,no_verify_reselection=True,
        fit_groups=84,verify_groups=209,fit_selected_parameters=selected,verification=stats,
        verification_performance_evaluated=True,verify_gate=verify_gate(stats),minimum_gain=.9,maximum_harm=.1)
    decision=dict(selected_parameters=selected);m=module();m.validate_result_contract(result,decision)
    changed=copy.deepcopy(result);changed['gate_final']=True
    with pytest.raises(ValueError):m.validate_result_contract(changed,decision)
