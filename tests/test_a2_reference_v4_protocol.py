import copy
import pytest

from experiments.a2_waldo_landmark_belief.reference_v4_protocol import CONFIG,validate_training_config,validate_pretraining_proofs


def inputs():
    binding = dict(code_sha256={'model':'a'},roles_sha256='r',feature_manifest_sha256='m',
        feature_signature_sha256='s',feature_build_status_sha256='f',initial_checkpoint_sha256='i',cpu_gates_sha256='c')
    common = {k:v for k,v in binding.items() if k!='cpu_gates_sha256'}
    cpu = dict(common,status='v4_cpu_tests_and_initial_state_verified',all_cpu_tests_passed=True,initial_weights_sha256='w')
    cuda = dict(common,status='v4_real_TRAIN_CUDA_backward_passed',cpu_gates_sha256='c',initial_weights_sha256='w',
                rank_backward_steps=3,refine_backward_steps=3,all_finite=True,frozen_parameters_unchanged=True)
    runtime = dict(common,status='v4_real_TRAIN_runtime_preflight_passed',cpu_gates_sha256='c',
        arm_counts={'A0':32,'OFF':32,'ON':32},exact_a0_off_parity=True,original_controller_verified=True,
        benchmark_used=False,functional_train_only=True)
    return dict(cpu=cpu,cuda=cuda,runtime=runtime),binding


def test_same_feature_and_source_bound_complete_proofs_are_required():
    proofs,binding = inputs()
    validate_pretraining_proofs(proofs,binding=binding)
    before = copy.deepcopy(proofs)
    assert proofs == before


def test_metadata_cannot_claim_different_batch_size_than_reused_training_engine(monkeypatch):
    validate_training_config()
    monkeypatch.setitem(CONFIG,'batch_size',32)
    with pytest.raises(ValueError): validate_training_config()


@pytest.mark.parametrize('kind,key,value', [
    ('cpu','status','v3_cpu_tests_and_initial_state_verified'),
    ('cuda','feature_manifest_sha256','other_cache'),
    ('runtime','roles_sha256','foreign_roles'),
    ('runtime','feature_build_status_sha256','other_build'),
    ('cuda','code_sha256',{'model':'changed'}),
    ('cpu','all_cpu_tests_passed',False),
    ('cuda','rank_backward_steps',1),
    ('cuda','refine_backward_steps',0),
    ('cuda','all_finite',False),
    ('cuda','frozen_parameters_unchanged',False),
    ('cuda','initial_weights_sha256','other_weights'),
    ('runtime','cpu_gates_sha256','old_proof'),
    ('runtime','arm_counts',{'A0':32,'OFF':32,'ON':1}),
    ('runtime','exact_a0_off_parity',False),
    ('runtime','original_controller_verified',False),
    ('runtime','benchmark_used',True),
    ('runtime','functional_train_only',False),
])
def test_old_partial_or_foreign_proofs_cannot_admit_new_training(kind,key,value):
    proofs,binding = inputs();proofs[kind][key] = value
    with pytest.raises(ValueError): validate_pretraining_proofs(proofs,binding=binding)
