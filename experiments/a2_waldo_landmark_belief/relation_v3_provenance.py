"""Pure V3 checkpoint/gate provenance checks; no legacy runtime bootstrap."""
import json
from pathlib import Path
import torch
from .paths import A0_SHA,sha
from .collected_snapshot import BASE


def validate_gate(parameters,state):
    if not parameters.get('enabled',False):return
    from .calibration_protocol import verify_gate
    stats=parameters.get('verification')
    if (not isinstance(stats,dict) or stats.get('groups')!=209 or parameters.get('fit_groups')!=84 or parameters.get('verify_groups')!=209
        or parameters.get('roles_sha256')!=state['roles_sha256'] or parameters.get('training_code_sha256')!=state['code_sha256']):
        raise ValueError('Independent84-FIT/209-VERIFY gate proof required')
    risk=verify_gate(stats)
    if not risk['passed'] or parameters.get('verify_gate')!=risk:raise ValueError('Independent statistical verification failed')
    selected={k:parameters[k] for k in ('minimum_gain','maximum_harm','temperature')}
    if selected!=parameters.get('fit_selected_parameters'):raise ValueError('Gate differs from frozen FIT choice')
    decision=BASE/'revision_v3/calibration/fit_decision.json'
    if parameters.get('fit_decision_sha256')!=sha(decision):raise ValueError('Independent FIT decision changed')
    fit=json.loads(decision.read_text());analysis=BASE/'revision_v3/calibration/analysis_signature.json'
    if (fit['selected_parameters']!=selected or fit['checkpoint_sha256']!=parameters['checkpoint_sha256']
        or fit['roles_sha256']!=state['roles_sha256'] or fit['fit_groups']!=84
        or fit['analysis_signature_sha256']!=sha(analysis) or parameters.get('analysis_signature_sha256')!=sha(analysis)):
        raise ValueError('Independent FIT decision changed')
    source=json.loads(analysis.read_text())
    if source['checkpoint_sha256']!=parameters['checkpoint_sha256'] or source['roles_sha256']!=state['roles_sha256']:
        raise ValueError('Independent calibration source lineage changed')


def checked_state(checkpoint,*,functional_only=False):
    checkpoint=Path(checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['a0_sha256']!=A0_SHA or state['config']['version']!='a2_relation_v3':raise ValueError('Wrong V3 lineage')
    if not functional_only and (not state.get('checkpoint_eligible') or state.get('epoch',0)<=20):raise ValueError('Completed eligible refinement checkpoint required')
    initial=BASE/'revision_v3/pretraining/initial.pt'
    if functional_only and (state.get('scope')!='TRAIN_functional_only' or checkpoint.resolve()!=initial.resolve()):
        raise ValueError('Only the pinned pretraining checkpoint is allowed for functional TRAIN checks')
    from .relation_v3_train import code_hashes
    if state['code_sha256']!=code_hashes() or state['roles_sha256']!=sha(BASE/'revision_v3/roles.json'):
        raise ValueError('V3 frozen implementation or roles changed')
    if functional_only:
        proof=json.loads((BASE/'revision_v3/cpu_gates.json').read_text())
        if proof['initial_checkpoint_sha256']!=sha(checkpoint) or proof['code_sha256']!=state['code_sha256']:
            raise ValueError('Current pretraining CPU proof required')
    else:
        directory=checkpoint.parent;marker=json.loads((directory/'training_complete.json').read_text())
        proof=json.loads((directory/'artifact_verification.json').read_text())
        if (marker['rank_epochs']!=20 or marker['refine_epochs']!=5 or marker['checkpoint_sha256']!=sha(checkpoint)
            or marker['selected_epoch']!=state['epoch'] or not marker['checkpoint_eligible']
            or marker['history_sha256']!=sha(directory/'history.json')
            or proof['status']!='v3_all25_artifacts_and_dev_verified' or proof['checkpoint_sha256']!=sha(checkpoint)
            or proof['code_sha256']!=state['code_sha256'] or proof['roles_sha256']!=state['roles_sha256']
            or not proof['all_optimizer_steps_verified'] or not proof['all_dev_epochs_recomputed']):
            raise ValueError('Complete verified V3 training artifacts required')
        from .relation_v3_verify import validate_verified_proof
        validate_verified_proof(directory,state['code_sha256'],state['roles_sha256'])
    return state


