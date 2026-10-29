"""Separate full20+5 trainer using source/target-balanced TRAIN weights."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import torch

from .paths import ROOT,A0_SHA,sha,save
from .reference_overlay_context import BASE
from .reference_overlay_data import ReferenceOverlayViews
from .reference_v4_train import (training_signature as v4_training_signature,
    validate_complete_feature_cache, CODE_FILES as V4_CODE_FILES)
from .reference_v5_data import TargetBalancedReferenceViews
from .reference_v5_protocol import CONFIG,choose_stage,verify_history,validate_training_config
from .reference_v5_weighting import target_balanced_weights, weighting_manifest
from .relation_v3_epoch import run_epoch
from .relation_v3_train import (PARENT_SHA,initial_model,fingerprint,require_idle_gpu,optimizer_for,
    start_refinement,verify_optimizer_steps,loader,save_packets,reconcile,validate_resume)
from .train import atomic_checkpoint,atomic_copy,write_record_scalars,restore_training_logs


DIRECTORY = BASE/'revision_v5/training'
CODE_FILES = tuple(sorted(set(V4_CODE_FILES) | {
    'reference_v5_protocol.py','reference_v5_weighting.py','reference_v5_data.py',
    'reference_v5_train.py','reference_v5_preparation.py','reference_v5_verify.py',
}))


def code_hashes():
    return {name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in CODE_FILES}


def _json_sha(value):
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_training_binding(parent_signature,entries,weights,v5_code_sha256):
    required=('feature_manifest_sha256','roles_sha256','initial_weights_sha256',
              'initial_checkpoint_sha256','code_sha256')
    if any(not parent_signature.get(key) for key in required):
        raise ValueError('Complete V4 parent training signature required')
    if not isinstance(v5_code_sha256,dict) or not v5_code_sha256:
        raise ValueError('V5 source hashes required')
    validate_training_config()
    entries=list(entries);weights=list(weights)
    if weights != target_balanced_weights(entries):
        raise ValueError('V5 training weights differ from the declared group-balancing rule')
    weighting=weighting_manifest(entries,weights)
    return dict(status='v5_target_group_balanced_training_binding',
        parent_signature_sha256=_json_sha(parent_signature),
        parent_code_sha256=parent_signature['code_sha256'],v5_code_sha256=v5_code_sha256,
        feature_manifest_sha256=parent_signature['feature_manifest_sha256'],
        roles_sha256=parent_signature['roles_sha256'],
        initial_weights_sha256=parent_signature['initial_weights_sha256'],
        initial_checkpoint_sha256=parent_signature.get('initial_checkpoint_sha256'),
        a0_sha256=A0_SHA,parent_sha256=PARENT_SHA,config=CONFIG,weighting=weighting,
        train_views=len(entries),dev_views=5647,benchmark_used=False)


def source_binding():
    parent=v4_training_signature()
    train=TargetBalancedReferenceViews('train')
    return build_training_binding(parent,train.entries,train.weights,code_hashes())


def attach_pretraining_hashes(binding,directory=BASE/'revision_v5',*,expected_sha256=None):
    from .reference_v5_preparation import validate_pretraining_proofs
    directory=Path(directory)
    validate_pretraining_proofs(binding,directory,expected_sha256=expected_sha256)
    return dict(binding,cpu_gates_sha256=sha(directory/'cpu_gates.json'),
                cuda_smoke_sha256=sha(directory/'cuda_smoke.json'))


def training_signature():
    binding=source_binding()
    return attach_pretraining_hashes(binding,BASE/'revision_v5')


def run_training(directory=DIRECTORY, *, resume=False):
    signature = training_signature()
    require_idle_gpu()
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'training_complete.json').exists():
        raise ValueError('V5 already completed; verify instead of rerunning')
    train,dev = TargetBalancedReferenceViews('train'),ReferenceOverlayViews('dev')
    if train.base.manifest_sha256 != dev.manifest_sha256 or train.base.manifest_sha256 != signature['feature_manifest_sha256']:
        raise ValueError('TRAIN/DEV reference manifests differ')
    roles_sha = train.base.parent.roles_sha256
    if roles_sha != dev.parent.roles_sha256 or roles_sha != signature['roles_sha256']:
        raise ValueError('TRAIN/DEV target roles differ')
    torch.set_num_threads(2);torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model = initial_model()
    if fingerprint(model.state_dict()) != signature['initial_weights_sha256']:
        raise ValueError('V5 initial model differs from actual pretraining proofs')
    require_idle_gpu();model=model.cuda()
    optimizer = optimizer_for(model);history=[];start=1;transition=None
    if (directory/'last.pt').exists():
        if not resume: raise ValueError('Explicit V5 --resume required')
        state=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        validate_resume(state,directory,signature)
        history=state['history'];start=state['epoch']+1;transition=state.get('stage_initialization')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_state'])
        reconcile(history,directory,signature)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(directory/'tensorboard'),purge_step=1)
    restore_training_logs(history,directory,writer)
    save(dict(signature,model_config=model.config,train_views=len(train),dev_views=len(dev),benchmark_used=False,
              confirmation_used=False),directory/'protocol.json')
    save(dev.entries,directory/'dev_observations.json')
    for epoch in range(start,26):
        if training_signature() != signature:
            raise ValueError('V5 source/data/pretraining proofs changed during training')
        require_idle_gpu(allow_self=True);refine=epoch>20
        if epoch==21:
            rank=choose_stage(history,refine=False);path=directory/f'epoch_{rank["epoch"]:02d}.pt'
            state=torch.load(path,map_location='cpu',weights_only=False);optimizer=start_refinement(model,state)
            transition=dict(source_epoch=rank['epoch'],source_checkpoint_sha256=sha(path),optimizer_reset=True,
                initial_optimizer_state_entries=len(optimizer.state),initial_weights_sha256=fingerprint(model.state_dict()))
            if transition['initial_weights_sha256'] != fingerprint(state['model']):
                raise ValueError('V5 refinement did not restore selected rank weights')
        trained,_=run_epoch(model,loader(train,epoch,shuffle=True),optimizer=optimizer,refine=refine)
        validation,packets=run_epoch(model,loader(dev,0,shuffle=False),refine=refine)
        weights={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        weight_sha=fingerprint(weights)
        packet_sha=save_packets(packets,directory/f'dev_epoch_{epoch:02d}.npz',epoch=epoch,weights_sha256=weight_sha,roles_sha256=roles_sha)
        record=dict(epoch=epoch,stage='refinement' if refine else 'ranking',stage_epoch=epoch-20 if refine else epoch,
                    train=trained,dev=validation,dev_packets_sha256=packet_sha,weights_sha256=weight_sha)
        history.append(record)
        state=dict(signature,epoch=epoch,model=weights,model_config=model.config,optimizer=optimizer.state_dict(),history=history,
            trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],stage_initialization=transition,
            rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),checkpoint_eligible=bool(refine and validation['eligible']))
        verify_optimizer_steps(state);path=directory/f'epoch_{epoch:02d}.pt';atomic_checkpoint(state,path)
        atomic_copy(path,directory/'last.pt');reconcile(history,directory,signature)
        save(history,directory/'history.json');write_record_scalars(record,writer);writer.flush();print(json.dumps(record),flush=True)
    writer.close();verify_history(history)
    chosen=choose_stage(history,refine=True);best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    if best['epoch'] != chosen['epoch'] or sha(directory/'best.pt') != sha(directory/f'epoch_{chosen["epoch"]:02d}.pt'):
            raise ValueError('V5 final checkpoint differs from selected refinement epoch')
    for epoch in range(1,26):
        state=torch.load(directory/f'epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False)
        verify_optimizer_steps(state)
        if any(state[key] != value for key,value in signature.items()) or fingerprint(state['model']) != history[epoch-1]['weights_sha256']:
            raise ValueError('V5 saved epoch lineage changed')
        if sha(directory/f'dev_epoch_{epoch:02d}.npz') != history[epoch-1]['dev_packets_sha256']:
            raise ValueError('V5 saved validation packet changed')
    save(dict(signature,status='v5_full20plus5_complete_artifact_verification_pending',rank_epochs=20,refine_epochs=5,
        selected_epoch=best['epoch'],checkpoint_sha256=sha(directory/'best.pt'),checkpoint_eligible=best['checkpoint_eligible'],
        history_sha256=sha(directory/'history.json'),no_benchmark_used=True,independent_confirmation_pending=True),
        directory/'training_complete.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_training(resume=args.resume)


if __name__=='__main__': main()
