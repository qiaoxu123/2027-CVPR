import pytest

from experiments.a2_waldo_landmark_belief.reference_v4_verify import validate_verified_header


def example():
    binding = dict(code_sha256={'model':'a'},roles_sha256='r',feature_manifest_sha256='m',
                   feature_signature_sha256='s',feature_build_status_sha256='f')
    proof = dict(binding,status='v4_all25_artifacts_and_dev_verified',selected_epoch=23,rank_selected_epoch=20,
        all_optimizer_steps_verified=True,all_dev_epochs_recomputed=True,
        epochs=[dict(epoch=e,recomputed=True) for e in range(1,26)])
    return proof,binding


def test_only_full_same_feature_bound_verification_is_admitted():
    proof,binding = example();validate_verified_header(proof,binding)


@pytest.mark.parametrize('key,value', [('status','v3_all25_artifacts_and_dev_verified'),
    ('feature_manifest_sha256','other'),('feature_build_status_sha256','other'),
    ('all_optimizer_steps_verified',False),('all_dev_epochs_recomputed',False),
    ('selected_epoch',20),('rank_selected_epoch',21),
    ('epochs',[dict(epoch=e,recomputed=True) for e in range(1,25)]),
    ('epochs',[dict(epoch=e,recomputed=e!=12) for e in range(1,26)])])
def test_old_foreign_or_partial_artifact_verification_is_rejected(key,value):
    proof,binding = example();proof[key]=value
    with pytest.raises(ValueError): validate_verified_header(proof,binding)
