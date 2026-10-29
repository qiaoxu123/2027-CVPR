import json

from experiments.a2_waldo_landmark_belief.reference_v5_train import (
    attach_pretraining_hashes, build_training_binding,
)
from experiments.a2_waldo_landmark_belief.paths import sha
from experiments.a2_waldo_landmark_belief.reference_v5_preparation import proof_binding_sha
from experiments.a2_waldo_landmark_belief.reference_v5_weighting import target_balanced_weights


def sample_entries():
    return [
        dict(role="train", source_kind="revision_v1_A0_closedloop", map_name="m", object_id=1),
        dict(role="train", source_kind="revision_v1_A0_closedloop", map_name="m", object_id=1),
        dict(role="train", source_kind="teacher_snapshot", map_name="n", object_id=2),
    ]


def test_training_binding_records_parent_and_exact_balancing_inputs():
    parent = dict(
        feature_manifest_sha256="manifest",
        roles_sha256="roles",
        initial_weights_sha256="weights",
        initial_checkpoint_sha256="initial-checkpoint",
        code_sha256={"v4.py": "parent-code"},
    )
    entries = sample_entries()
    weights = target_balanced_weights(entries)

    binding = build_training_binding(parent, entries, weights, {"v5.py": "new-code"})

    assert binding["feature_manifest_sha256"] == "manifest"
    assert binding["roles_sha256"] == "roles"
    assert binding["initial_weights_sha256"] == "weights"
    assert binding["initial_checkpoint_sha256"] == "initial-checkpoint"
    assert binding["parent_code_sha256"] == {"v4.py": "parent-code"}
    assert binding["v5_code_sha256"] == {"v5.py": "new-code"}
    assert binding["weighting"]["weights_sha256"]
    assert binding["train_views"] == 3
    assert binding["benchmark_used"] is False


def test_training_binding_requires_complete_parent_proof_fields():
    try:
        build_training_binding({}, sample_entries(), target_balanced_weights(sample_entries()), {})
    except ValueError as error:
        assert "parent" in str(error).lower()
    else:
        raise AssertionError("Incomplete V4 parent must not bind V5")


def test_training_binding_rejects_weights_that_do_not_follow_the_v5_rule():
    parent = dict(feature_manifest_sha256="manifest", roles_sha256="roles",
                  initial_weights_sha256="weights", initial_checkpoint_sha256="initial",
                  code_sha256={"v4.py": "parent"})
    entries = sample_entries()
    weights = target_balanced_weights(entries)
    weights[0] *= 2

    try:
        build_training_binding(parent, entries, weights, {"v5.py": "new-code"})
    except ValueError as error:
        assert "weight" in str(error).lower()
    else:
        raise AssertionError("A foreign weight vector must not bind V5")


def test_training_signature_binds_the_exact_pretraining_proof_files(tmp_path):
    binding = {"source": "v5"}
    proof_sha = proof_binding_sha(binding)
    log = tmp_path / "cpu_tests.log"
    log.write_text("17 passed in 1.00s\n")
    (tmp_path / "cpu_gates.json").write_text(json.dumps(dict(
        status="v5_cpu_tests_and_weighting_verified", binding_sha256=proof_sha,
        all_cpu_tests_passed=True, no_gpu_used=True, returncode=0,
        unit_test_summary="17 passed in 1.00s", test_log_sha256=sha(log),
    )))
    (tmp_path / "cuda_smoke.json").write_text(json.dumps(dict(
        status="v5_weighted_TRAIN_CUDA_backward_passed", binding_sha256=proof_sha,
        rank_backward_steps=3, refine_backward_steps=3, all_finite=True,
        frozen_parameters_unchanged=True, weighted_rows=True, benchmark_used=False,
    )))
    signature = attach_pretraining_hashes(binding, tmp_path, expected_sha256=proof_sha)
    (tmp_path / "cuda_smoke.json").write_text('{"changed":true}')

    try:
        attach_pretraining_hashes(binding, tmp_path, expected_sha256=proof_sha)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed CUDA proof must not retain the same training signature")
    assert signature["cpu_gates_sha256"]
    assert signature["cuda_smoke_sha256"]
