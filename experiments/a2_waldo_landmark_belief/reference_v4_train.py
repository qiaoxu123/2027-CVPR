"""Separate full20+5 trainer for repaired public reference inputs.

Unchanged V3 optimizer/epoch/selection primitives are reused explicitly; neither
their globals nor the original experiment directory are monkeypatched.
"""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from .paths import ROOT,A0_SHA,sha,save
from .reference_overlay_context import BASE,DEFAULT_OUTPUT,CODE_FILES as FEATURE_CODE_FILES,feature_signature
from .reference_overlay import verify_overlay_population
from .reference_overlay_data import ReferenceOverlayViews
from .reference_v4_protocol import CONFIG,choose_stage,verify_history,validate_pretraining_proofs
from .relation_v3_epoch import run_epoch
from .relation_v3_train import (PARENT_SHA,initial_model,fingerprint,require_idle_gpu,optimizer_for,
    start_refinement,verify_optimizer_steps,loader,save_packets,reconcile,validate_resume)
from .train import atomic_checkpoint,atomic_copy,write_record_scalars,restore_training_logs


DIRECTORY = BASE/'revision_v4/training'
CODE_FILES = tuple(sorted(set(FEATURE_CODE_FILES) | {
    'reference_perception.py','reference_v4_protocol.py','reference_v4_train.py',
    'reference_v4_preparation.py','reference_v4_policy.py','reference_v4_verify.py',
    'relation_v3_train.py','relation_v3_epoch.py','relation_v3_protocol.py','relation_v3_model.py',
    'relation_v3_loss.py','relation_v3_fusion.py','relation_v3_runtime.py','relation_v3_policy.py',
    'model.py','risk_probe.py','revision_fusion.py','batching.py','train.py','training_protocol.py',
    'policy.py','perception.py','observed_geometry.py','collected_snapshot.py',
    'direct_risk_protocol.py','gpu_execution_policy.py','gpu_execution_policy.json'}))


def code_hashes():
    return {name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in CODE_FILES}


def validate_complete_feature_cache(directory=DEFAULT_OUTPUT):
    directory = Path(directory);path = directory/'manifest.json'
    if not path.is_file():
        raise ValueError('Complete reference feature cache required before training')
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'all_train_dev_reference_overlays_complete':
        raise ValueError('Complete reference feature cache required before training')
    if (manifest.get('counts',{}).get('train',{}).get('observations') != 33440
        or manifest.get('counts',{}).get('dev',{}).get('observations') != 5647):
        raise ValueError('Incomplete reference feature population')
    if (manifest.get('parent_cache_unchanged') is not True or manifest.get('no_gpu_used') is not True
        or manifest.get('calibration_tensors_opened') is not False or manifest.get('benchmark_used') is not False):
        raise ValueError('Invalid reference feature audit provenance')
    build_status_path = directory/'status.json'
    if not build_status_path.is_file():
        raise ValueError('A fresh full feature build with bound completion status is required')
    built = json.loads(build_status_path.read_text())
    # The v1 builder has no independently committed per-file resume journal.
    # Admit this currently fresh run only; never bless resumed tensors by
    # accepting hashes computed after a possible interrupted-file change.
    if (built.get('state') != 'complete' or built.get('processed') != 39087 or built.get('total') != 39087
        or built.get('resumed') != 0 or built.get('manifest_sha256') != sha(path)):
        raise ValueError('A fresh full feature build is required; resumed v1 caches are not admitted')
    signature_path = directory/'signature.json'
    if manifest['signature_sha256'] != sha(signature_path) or json.loads(signature_path.read_text()) != feature_signature():
        raise ValueError('Reference feature signature changed')
    roles_path = BASE/'revision_v3/roles.json'
    roles = json.loads(roles_path.read_text())
    verify_overlay_population(manifest['entries'],[e for e in roles['rows'] if e['role'] in ('train','dev')])
    return dict(feature_manifest_sha256=sha(path),feature_signature_sha256=sha(signature_path),
                feature_build_status_sha256=sha(build_status_path),roles_sha256=sha(roles_path))


def training_signature():
    feature = validate_complete_feature_cache()
    binding = dict(feature,code_sha256=code_hashes(),initial_checkpoint_sha256=sha(BASE/'revision_v4/pretraining/initial.pt'),
                   cpu_gates_sha256=sha(BASE/'revision_v4/cpu_gates.json'))
    paths = {name:BASE/'revision_v4'/filename for name,filename in (
        ('cpu','cpu_gates.json'),('cuda','cuda_smoke.json'),('runtime','runtime_preflight.json'))}
    proofs = {name:json.loads(path.read_text()) for name,path in paths.items()}
    validate_pretraining_proofs(proofs,binding=binding)
    from .reference_v4_preparation import test_hashes
    if proofs['cpu']['test_source_sha256'] != test_hashes():
        raise ValueError('CPU test sources changed after V4 gate')
    return dict(binding,config=CONFIG,a0_sha256=A0_SHA,parent_sha256=PARENT_SHA,
                initial_weights_sha256=proofs['cpu']['initial_weights_sha256'],
                cuda_smoke_sha256=sha(paths['cuda']),runtime_preflight_sha256=sha(paths['runtime']))


def run_training(directory=DIRECTORY, *, resume=False):
    signature = training_signature()
    require_idle_gpu()
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'training_complete.json').exists():
        raise ValueError('V4 already completed; verify instead of rerunning')
    train,dev = ReferenceOverlayViews('train'),ReferenceOverlayViews('dev')
    if train.manifest_sha256 != dev.manifest_sha256 or train.manifest_sha256 != signature['feature_manifest_sha256']:
        raise ValueError('TRAIN/DEV reference manifests differ')
    roles_sha = train.parent.roles_sha256
    if roles_sha != dev.parent.roles_sha256 or roles_sha != signature['roles_sha256']:
        raise ValueError('TRAIN/DEV target roles differ')
    torch.set_num_threads(2);torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model = initial_model()
    if fingerprint(model.state_dict()) != signature['initial_weights_sha256']:
        raise ValueError('V4 initial model differs from actual pretraining proofs')
    require_idle_gpu();model=model.cuda()
    optimizer = optimizer_for(model);history=[];start=1;transition=None
    if (directory/'last.pt').exists():
        if not resume: raise ValueError('Explicit V4 --resume required')
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
            raise ValueError('V4 source/data/pretraining proofs changed during training')
        require_idle_gpu(allow_self=True);refine=epoch>20
        if epoch==21:
            rank=choose_stage(history,refine=False);path=directory/f'epoch_{rank["epoch"]:02d}.pt'
            state=torch.load(path,map_location='cpu',weights_only=False);optimizer=start_refinement(model,state)
            transition=dict(source_epoch=rank['epoch'],source_checkpoint_sha256=sha(path),optimizer_reset=True,
                initial_optimizer_state_entries=len(optimizer.state),initial_weights_sha256=fingerprint(model.state_dict()))
            if transition['initial_weights_sha256'] != fingerprint(state['model']):
                raise ValueError('V4 refinement did not restore selected rank weights')
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
        raise ValueError('V4 final checkpoint differs from selected refinement epoch')
    for epoch in range(1,26):
        state=torch.load(directory/f'epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False)
        verify_optimizer_steps(state)
        if any(state[key] != value for key,value in signature.items()) or fingerprint(state['model']) != history[epoch-1]['weights_sha256']:
            raise ValueError('V4 saved epoch lineage changed')
        if sha(directory/f'dev_epoch_{epoch:02d}.npz') != history[epoch-1]['dev_packets_sha256']:
            raise ValueError('V4 saved validation packet changed')
    save(dict(signature,status='v4_full20plus5_complete_artifact_verification_pending',rank_epochs=20,refine_epochs=5,
        selected_epoch=best['epoch'],checkpoint_sha256=sha(directory/'best.pt'),checkpoint_eligible=best['checkpoint_eligible'],
        history_sha256=sha(directory/'history.json'),no_benchmark_used=True,independent_confirmation_pending=True),
        directory/'training_complete.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_training(resume=args.resume)


if __name__=='__main__': main()
