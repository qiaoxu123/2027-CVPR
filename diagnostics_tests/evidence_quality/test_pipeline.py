"""Run control never treats waiting or a stale CPU proof as an experiment result."""
import importlib

import pytest


def api():
    return importlib.import_module('diagnostics.evidence_quality.pipeline')


def test_waiting_does_not_start_model_fitting(tmp_path,monkeypatch):
    module=api();monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(module,'export_packets',lambda root,directory:None)
    assert module.run_once(tmp_path/'root',tmp_path/'out') is None
    assert not (tmp_path/'out/fit').exists()


def test_cpu_proof_binds_actual_test_log_and_source(tmp_path):
    module=api();log=tmp_path/'cpu_tests.log';log.write_text('76 passed in 10.00s\n')
    signature=dict(implementation_sha256={'module.py':'a'*64})
    proof=dict(status='evidence_quality_actual_CPU_tests_passed',signature=signature,
               test_log_sha256=module.sha(log),unit_test_summary='76 passed in 10.00s',
               returncode=0,no_gpu_used=True)
    module.validate_cpu_proof(proof,signature,log)
    with pytest.raises(ValueError):module.validate_cpu_proof(proof,{},log)
    log.write_text('changed')
    with pytest.raises(ValueError):module.validate_cpu_proof(proof,signature,log)


def test_nonzero_test_return_cannot_be_admitted(tmp_path):
    module=api();log=tmp_path/'cpu_tests.log';log.write_text('1 passed, 1 failed\n')
    proof=dict(status='evidence_quality_actual_CPU_tests_passed',signature={},
               test_log_sha256=module.sha(log),unit_test_summary='1 passed',returncode=1,no_gpu_used=True)
    with pytest.raises(ValueError):module.validate_cpu_proof(proof,{},log)


def test_complete_internal_workflow_after_simulated_original_export(tmp_path,monkeypatch):
    from diagnostics_tests.evidence_quality.test_report import signal_packets
    module=api();monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    directory=tmp_path/'output'
    exported=signal_packets(directory,binding=module.source_binding(dict(checkpoint_sha256='c'*64)))
    # Original native completion/export is unavailable in unit tests; actual
    # fitting, immutable choice, DEV assessment, reports and source checks run.
    monkeypatch.setattr(module,'export_packets',lambda root,directory:exported)
    result=module.run_once(tmp_path/'root',directory)
    assert result['promising_for_formal_iteration'] is True
    assert result['objective_achieved'] is False
    assert result['no_runtime_enablement'] is True
