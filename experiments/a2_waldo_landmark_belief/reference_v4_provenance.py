"""V4 checkpoint admission and independent gate lineage; no runtime bootstrap."""
import json
from pathlib import Path

import torch

from .paths import A0_SHA,sha
from .reference_overlay_context import BASE


ROOT = BASE/'revision_v4'


def validate_state_binding(state, *, binding, functional_only):
    if state.get('a0_sha256') != A0_SHA or state.get('config',{}).get('version') != 'a2_relation_v4_public_roads':
        raise ValueError('Wrong reference V4 baseline or version')
    for key in ('code_sha256','roles_sha256','feature_manifest_sha256','feature_signature_sha256','feature_build_status_sha256'):
        if state.get(key) != binding[key]:
            raise ValueError('Reference V4 checkpoint belongs to other data/code')
    if functional_only:
        if state.get('scope') != 'TRAIN_functional_only' or state.get('epoch') != 0 or state.get('checkpoint_eligible') is not False:
            raise ValueError('Only pinned untrained state may be used for functional TRAIN checks')
    elif (state.get('scope') == 'TRAIN_functional_only' or state.get('epoch') not in range(21,26)
          or state.get('checkpoint_eligible') is not True):
        raise ValueError('Completed eligible V4 refinement checkpoint required')


def checked_state(checkpoint, *, functional_only=False):
    from .reference_v4_preparation import current_binding
    binding = current_binding()
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    validate_state_binding(state,binding=binding,functional_only=functional_only)
    if functional_only:
        initial = ROOT/'pretraining/initial.pt'
        cpu = json.loads((ROOT/'cpu_gates.json').read_text())
        if (checkpoint.resolve() != initial.resolve() or cpu['initial_checkpoint_sha256'] != sha(checkpoint)
            or any(cpu.get(key) != value for key,value in binding.items())):
            raise ValueError('Pinned V4 initial checkpoint and current CPU proof required')
    else:
        directory = checkpoint.parent
        marker = json.loads((directory/'training_complete.json').read_text())
        from .reference_v4_verify import validate_verified_proof
        proof = validate_verified_proof(directory,binding)
        if (marker['rank_epochs'] != 20 or marker['refine_epochs'] != 5
            or marker['checkpoint_sha256'] != sha(checkpoint) or marker['selected_epoch'] != state['epoch']
            or marker['checkpoint_eligible'] is not True or marker['history_sha256'] != sha(directory/'history.json')
            or proof['checkpoint_sha256'] != sha(checkpoint)):
            raise ValueError('Complete independently verified V4 training required')
    return state


def validate_gate(parameters,state):
    if not parameters.get('enabled',False):
        return
    from .calibration_protocol import verify_gate
    stats = parameters.get('verification')
    if (not isinstance(stats,dict) or stats.get('groups') != 209 or parameters.get('fit_groups') != 84
        or parameters.get('verify_groups') != 209 or parameters.get('roles_sha256') != state['roles_sha256']
        or parameters.get('training_code_sha256') != state['code_sha256']
        or parameters.get('feature_manifest_sha256') != state['feature_manifest_sha256']):
        raise ValueError('Independent feature-bound84-FIT/209-VERIFY proof required')
    risk = verify_gate(stats)
    if not risk['passed'] or parameters.get('verify_gate') != risk:
        raise ValueError('Independent V4 statistical verification failed')
    selected = {key:parameters[key] for key in ('minimum_gain','maximum_harm','temperature')}
    if selected != parameters.get('fit_selected_parameters'):
        raise ValueError('V4 gate differs from frozen independent FIT choice')
    decision = ROOT/'calibration/fit_decision.json'
    analysis = ROOT/'calibration/analysis_signature.json'
    if parameters.get('fit_decision_sha256') != sha(decision) or parameters.get('analysis_signature_sha256') != sha(analysis):
        raise ValueError('Independent V4 calibration files changed')
    fit = json.loads(decision.read_text());source = json.loads(analysis.read_text())
    if (fit['selected_parameters'] != selected or fit['checkpoint_sha256'] != parameters['checkpoint_sha256']
        or fit['roles_sha256'] != state['roles_sha256'] or fit['fit_groups'] != 84
        or fit['analysis_signature_sha256'] != sha(analysis)
        or source['checkpoint_sha256'] != parameters['checkpoint_sha256']
        or source['roles_sha256'] != state['roles_sha256']
        or source['feature_manifest_sha256'] != state['feature_manifest_sha256']):
        raise ValueError('Independent V4 FIT/analysis lineage changed')
