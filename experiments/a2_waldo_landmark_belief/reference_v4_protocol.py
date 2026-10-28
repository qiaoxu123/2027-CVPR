"""New-feature training admission; never reuse V3 proofs for rebuilt inputs."""
from copy import deepcopy

from .relation_v3_protocol import CONFIG as V3_CONFIG, choose_stage, verify_history


CONFIG = deepcopy(V3_CONFIG)
CONFIG.update(version='a2_relation_v4_public_roads',
              reference_features='connected public road regions + occurrence mentions; missing new-reference heights')


def validate_training_config():
    # Reused V3 numeric helpers intentionally retain their original CONFIG.
    # Refuse any metadata-only change that would misdescribe actual updates.
    if any(CONFIG.get(key) != value for key,value in V3_CONFIG.items() if key != 'version'):
        raise ValueError('V4 public-input revision must preserve reused training hyperparameters')


def validate_pretraining_proofs(proofs, *, binding):
    validate_training_config()
    statuses = dict(cpu='v4_cpu_tests_and_initial_state_verified',
                    cuda='v4_real_TRAIN_CUDA_backward_passed',runtime='v4_real_TRAIN_runtime_preflight_passed')
    if set(proofs) != set(statuses):
        raise ValueError('All new-version CPU/CUDA/runtime proofs are required')
    for kind,status in statuses.items():
        proof = proofs[kind]
        if proof.get('status') != status:
            raise ValueError('Old or incomplete reference V4 pretraining proof')
        for key in ('code_sha256','roles_sha256','feature_manifest_sha256','feature_signature_sha256','feature_build_status_sha256','initial_checkpoint_sha256'):
            if proof.get(key) != binding[key]:
                raise ValueError('Reference V4 proof belongs to other code/data/initial state')
    cpu,cuda,runtime = proofs['cpu'],proofs['cuda'],proofs['runtime']
    if cpu.get('all_cpu_tests_passed') is not True:
        raise ValueError('Complete CPU test proof required')
    if (not cpu.get('initial_weights_sha256') or cuda.get('initial_weights_sha256') != cpu['initial_weights_sha256']
        or cuda.get('cpu_gates_sha256') != binding['cpu_gates_sha256']
        or runtime.get('cpu_gates_sha256') != binding['cpu_gates_sha256']):
        raise ValueError('Reference V4 CPU/initial-state proof chain changed')
    if (cuda.get('rank_backward_steps') != 3 or cuda.get('refine_backward_steps') != 3
        or cuda.get('all_finite') is not True or cuda.get('frozen_parameters_unchanged') is not True):
        raise ValueError('Actual finite three-ranking/three-refinement backward steps required')
    if (runtime.get('arm_counts') != {'A0':32,'OFF':32,'ON':32}
        or runtime.get('exact_a0_off_parity') is not True or runtime.get('original_controller_verified') is not True
        or runtime.get('functional_train_only') is not True or runtime.get('benchmark_used') is not False):
        raise ValueError('Complete TRAIN-only runtime parity/controller proof required')
