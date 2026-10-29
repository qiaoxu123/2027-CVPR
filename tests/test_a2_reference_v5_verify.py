import pytest

from experiments.a2_waldo_landmark_belief.reference_v5_verify import validate_verified_header


def complete_proof(binding):
    proof = dict(binding)
    proof.update(
        status="v5_all25_artifacts_and_dev_verified",
        all_optimizer_steps_verified=True,
        all_dev_epochs_recomputed=True,
        selected_epoch=23,
        rank_selected_epoch=17,
        epochs=[dict(epoch=epoch, recomputed=True) for epoch in range(1, 26)],
    )
    return proof


def test_verifier_requires_all_25_recomputed_v5_checkpoints():
    binding = {"parent_signature_sha256": "parent", "v5_code_sha256": {"v5.py": "code"}}
    validate_verified_header(complete_proof(binding), binding)


@pytest.mark.parametrize("mutate", [
    lambda proof: proof.update(epochs=proof["epochs"][:-1]),
    lambda proof: proof.update(epochs=[dict(epoch=epoch, recomputed=True) for epoch in range(2, 27)]),
    lambda proof: proof["epochs"][4].update(recomputed=False),
    lambda proof: proof.update(selected_epoch=20),
    lambda proof: proof.update(status="v4_all25_artifacts_and_dev_verified"),
])
def test_verifier_rejects_partial_or_foreign_v5_artifact_proof(mutate):
    binding = {"parent_signature_sha256": "parent"}
    proof = complete_proof(binding)
    mutate(proof)
    with pytest.raises(ValueError):
        validate_verified_header(proof, binding)
