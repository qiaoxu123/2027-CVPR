import copy
import pytest

from experiments.a2_waldo_landmark_belief.reference_v4_provenance import validate_state_binding


def example():
    from experiments.a2_waldo_landmark_belief.paths import A0_SHA
    binding = dict(code_sha256={'model':'a'},roles_sha256='r',feature_manifest_sha256='m',
                   feature_signature_sha256='s',feature_build_status_sha256='f')
    state = dict(binding,a0_sha256=A0_SHA,config={'version':'a2_relation_v4_public_roads'},
                 epoch=0,scope='TRAIN_functional_only',checkpoint_eligible=False)
    return state,binding


def test_only_initial_state_is_accepted_for_functional_checks_and_never_final_use():
    state,binding = example()
    validate_state_binding(state,binding=binding,functional_only=True)
    with pytest.raises(ValueError): validate_state_binding(state,binding=binding,functional_only=False)
    forged = dict(state,epoch=23,checkpoint_eligible=True)
    with pytest.raises(ValueError): validate_state_binding(forged,binding=binding,functional_only=False)
    trained = dict(forged);trained.pop('scope')
    validate_state_binding(trained,binding=binding,functional_only=False)
    with pytest.raises(ValueError): validate_state_binding(trained,binding=binding,functional_only=True)


@pytest.mark.parametrize('key,value', [('feature_manifest_sha256','other'),('feature_signature_sha256','other'),
    ('feature_build_status_sha256','other'),('roles_sha256','other'),('code_sha256',{'model':'other'}),
    ('a0_sha256','other'),('config',{'version':'a2_relation_v3'}),('scope','benchmark')])
def test_wrong_feature_baseline_or_functional_scope_is_rejected(key,value):
    state,binding = example();state[key]=value
    with pytest.raises(ValueError): validate_state_binding(state,binding=binding,functional_only=True)
