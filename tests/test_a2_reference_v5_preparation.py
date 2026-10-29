import json

import pytest

from experiments.a2_waldo_landmark_belief.paths import sha
from experiments.a2_waldo_landmark_belief.reference_v5_preparation import (
    assert_cpu_environment, proof_binding_sha, validate_cpu_proof,
    validate_pretraining_proofs,
)


def proof_files(directory, binding):
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "cpu_tests.log"
    log.write_text("17 passed in 1.00s\n")
    binding_sha = proof_binding_sha(binding)
    (directory / "cpu_gates.json").write_text(json.dumps(dict(
        status="v5_cpu_tests_and_weighting_verified", binding_sha256=binding_sha,
        all_cpu_tests_passed=True, no_gpu_used=True, returncode=0,
        unit_test_summary="17 passed in 1.00s", test_log_sha256=sha(log),
    )))
    (directory / "cuda_smoke.json").write_text(json.dumps(dict(
        status="v5_weighted_TRAIN_CUDA_backward_passed", binding_sha256=binding_sha,
        rank_backward_steps=3, refine_backward_steps=3, all_finite=True,
        frozen_parameters_unchanged=True, weighted_rows=True, benchmark_used=False,
    )))
    return binding_sha


def test_preflight_requires_cpu_and_actual_weighted_cuda_proofs(tmp_path):
    binding = {"source": "v5"}
    binding_sha = proof_files(tmp_path, binding)

    validate_pretraining_proofs(binding, tmp_path, expected_sha256=binding_sha)


def test_preflight_rejects_foreign_binding(tmp_path):
    binding_sha = proof_files(tmp_path, {"source": "v5"})

    with pytest.raises(ValueError, match="binding"):
        validate_pretraining_proofs({"source": "changed"}, tmp_path, expected_sha256=binding_sha)


def test_preflight_rejects_old_or_incomplete_cuda_smoke(tmp_path):
    binding = {"source": "v5"}
    binding_sha = proof_files(tmp_path, binding)
    path = tmp_path / "cuda_smoke.json"
    proof = json.loads(path.read_text())
    proof["rank_backward_steps"] = 1
    path.write_text(json.dumps(proof))

    with pytest.raises(ValueError, match="CUDA"):
        validate_pretraining_proofs(binding, tmp_path, expected_sha256=binding_sha)


def test_cuda_smoke_can_start_after_cpu_proof_without_existing_cuda_file(tmp_path):
    binding = {"source": "v5"}
    binding_sha = proof_files(tmp_path, binding)
    (tmp_path / "cuda_smoke.json").unlink()

    validate_cpu_proof(binding, tmp_path, expected_sha256=binding_sha)


@pytest.mark.parametrize("visible", [None, ""])
def test_cpu_preparation_accepts_unset_or_hidden_cuda(visible):
    assert_cpu_environment(visible_devices=visible, cuda_initialized=False)


@pytest.mark.parametrize("visible,initialized", [("0", False), (None, True)])
def test_cpu_preparation_rejects_visible_or_initialized_cuda(visible, initialized):
    with pytest.raises(ValueError, match="CPU"):
        assert_cpu_environment(visible_devices=visible, cuda_initialized=initialized)
