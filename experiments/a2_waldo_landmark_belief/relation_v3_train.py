"""Full20+5 V3 trainer, gated on current CPU/CUDA and runtime preflight proofs."""
import argparse,fcntl,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .paths import ROOT,A0_SHA,sha,save
from .collected_snapshot import BASE
from .batching import collate_views
from .relation_v3_data import RelationV3Views
from .relation_v3_model import RelationV3Network
from .relation_v3_epoch import run_epoch
from .relation_v3_protocol import CONFIG,choose_stage,verify_history
from .train import atomic_checkpoint,atomic_copy,write_record_scalars,restore_training_logs

DIRECTORY=BASE/'revision_v3/training'
PARENT=BASE/'revision_v1/training/best.pt'
PARENT_SHA='b5f7ae68659d1550c77e6d1cb4c32447433e9c2d459b2e4b2ac221ef45a2fc9d'
CODE_FILES=('relation_v3_train.py','relation_v3_epoch.py','relation_v3_protocol.py','relation_v3_runtime.py',
    'relation_v3_policy.py','relation_v3_preparation.py','relation_v3_preflight.py','relation_v3_data.py',
    'relation_v3_model.py','relation_v3_fusion.py','relation_v3_loss.py','model.py','core.py','risk_probe.py',
    'revision_fusion.py','batching.py','paths.py','train.py','training_protocol.py','collected_snapshot.py',
    'data_protocol.py','perception.py','observed_geometry.py','direct_risk_protocol.py','policy.py','mentions.py',
    'prepare.py','revision_confirmation.py','calibration_protocol.py','relation_v3_verify.py','relation_v3_pipeline.py','relation_v3_provenance.py','artifact_protocol.py',
    'gpu_execution_policy.py','gpu_execution_policy.json')


def code_hashes():return {n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in CODE_FILES}


def fingerprint(weights):
    h=hashlib.sha256()
    for name,value in weights.items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def require_idle_gpu(*,allow_self=False):
    from .gpu_execution_policy import blocking_gpu_pids
    if blocking_gpu_pids(allow_self=allow_self):raise ValueError('GPU occupied; no parallel A2 GPU work allowed')


def optimizer_for(model):
    return torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])


def start_refinement(model,rank_state):
    model.load_state_dict(rank_state['model'],strict=True)
    optimizer=optimizer_for(model)
    if optimizer.state:raise ValueError('Refinement optimizer was not reset')
    return optimizer


def verify_optimizer_steps(state):
    names=state['trainable_parameter_names'];groups=state['optimizer']['param_groups'];ids=[i for g in groups for i in g['params']]
    if len(ids)!=len(names) or len(set(ids))!=len(ids):raise ValueError('Optimizer parameter mapping changed')
    epoch=state['epoch'];expected=(epoch if epoch<=20 else epoch-20)*math.ceil(33440/CONFIG['batch_size'])
    expected_ids={i for i,n in zip(ids,names) if epoch>20 or not n.startswith('risk.')}
    actual=state['optimizer']['state']
    if set(actual)!=expected_ids or any(float(s['step'])!=expected for s in actual.values()):raise ValueError('Optimizer steps or stage-active parameters incomplete')


def loader(dataset,epoch,*,shuffle):
    return DataLoader(dataset,batch_size=CONFIG['batch_size'],num_workers=CONFIG['workers'],shuffle=shuffle,
        collate_fn=collate_views,pin_memory=True,generator=torch.Generator().manual_seed(CONFIG['seed']+epoch))


def validate_audit():
    audit=BASE/'revision_v3/data_audit.json';data=json.loads(audit.read_text())
    if data['status']!='all_v3_TRAIN_dev_rows_sha_and_labels_audited' or data['roles_sha256']!=sha(BASE/'revision_v3/roles.json'):
        raise ValueError('Current all-row V3 audit required')
    for n,digest in data['label_code_sha256'].items():
        if sha(ROOT/'experiments/a2_waldo_landmark_belief'/n)!=digest:raise ValueError('V3 labels changed after audit')
    if data['counts']['train']['counts']['views']!=33440 or data['counts']['dev']['counts']['views']!=5647:raise ValueError('Partial V3 audit')
    return sha(audit)


def initial_model():
    torch.manual_seed(0)
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Frozen parent checkpoint changed')
    model=RelationV3Network();model.warmstart(torch.load(PARENT,map_location='cpu',weights_only=False))
    return model


def save_packets(packets,path,*,epoch,weights_sha256,roles_sha256):
    arrays={k:v for k,v in packets.items() if k!='identities'}
    arrays.update(epoch=np.asarray(epoch),weights_sha256=np.asarray(weights_sha256),roles_sha256=np.asarray(roles_sha256))
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temp.replace(path)
    return sha(path)


def reconcile(history,directory,signature):
    for refine,alias in ((False,'best_rank.pt'),(True,'best.pt')):
        if not any((r['epoch']>20)==refine for r in history):continue
        chosen=choose_stage(history,refine=refine);path=directory/f'epoch_{chosen["epoch"]:02d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        if (any(state[k]!=v for k,v in signature.items()) or state['epoch']!=chosen['epoch']
            or fingerprint(state['model'])!=chosen['weights_sha256']):raise ValueError('Best checkpoint lineage or weights changed')
        atomic_copy(path,directory/alias)


def validate_resume(state,directory,signature):
    if any(state[k]!=v for k,v in signature.items()):raise ValueError('V3 resume signature changed')
    epoch=state['epoch'];history=state['history']
    if len(history)!=epoch or [r['epoch'] for r in history]!=list(range(1,epoch+1)):raise ValueError('V3 resume history incomplete')
    if sha(directory/'last.pt')!=sha(directory/f'epoch_{epoch:02d}.pt'):raise ValueError('V3 last checkpoint differs from saved epoch')
    if fingerprint(state['model'])!=history[-1]['weights_sha256']:raise ValueError('V3 resume tensors differ from completed history')
    verify_optimizer_steps(state)


def training_signature():
    hashes=code_hashes();audit_sha=validate_audit()
    proofs={}
    for name,status in (('cpu_gates','v3_cpu_tests_and_initial_state_verified'),
                        ('cuda_smoke','v3_real_TRAIN_CUDA_backward_passed'),
                        ('runtime_preflight','v3_real_TRAIN_runtime_preflight_passed')):
        path=BASE/'revision_v3'/f'{name}.json';proof=json.loads(path.read_text())
        if proof['status']!=status or proof['code_sha256']!=hashes or proof['roles_sha256']!=sha(BASE/'revision_v3/roles.json'):
            raise ValueError(f'Current V3 {name} proof required')
        proofs[name+'_sha256']=sha(path)
    from .relation_v3_preparation import test_hashes
    cpu=json.loads((BASE/'revision_v3/cpu_gates.json').read_text())
    smoke=json.loads((BASE/'revision_v3/cuda_smoke.json').read_text())
    preflight=json.loads((BASE/'revision_v3/runtime_preflight.json').read_text())
    if (cpu['audit_sha256']!=audit_sha or cpu['test_source_sha256']!=test_hashes()
        or smoke['cpu_gates_sha256']!=proofs['cpu_gates_sha256'] or preflight['cpu_gates_sha256']!=proofs['cpu_gates_sha256']
        or smoke['initial_weights_sha256']!=cpu['initial_weights_sha256']
        or preflight['initial_checkpoint_sha256']!=cpu['initial_checkpoint_sha256']
        or cpu['initial_checkpoint_sha256']!=sha(BASE/'revision_v3/pretraining/initial.pt')):raise ValueError('Pretraining proof chain changed')
    return dict(code_sha256=hashes,config=CONFIG,a0_sha256=A0_SHA,parent_sha256=PARENT_SHA,
        roles_sha256=sha(BASE/'revision_v3/roles.json'),audit_sha256=audit_sha,**proofs)


def run_training(directory=DIRECTORY,*,resume=False):
    # Called only while the shared lock is held and before a CUDA context exists.
    require_idle_gpu();signature=training_signature();hashes=signature['code_sha256']
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'training_complete.json').exists():raise ValueError('V3 already completed; verify rather than rerun')
    train=RelationV3Views('train');dev=RelationV3Views('dev')
    if train.roles_sha256!=dev.roles_sha256 or train.roles_sha256!=signature['roles_sha256']:raise ValueError('TRAIN/dev roles differ')
    torch.set_num_threads(2);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=initial_model();require_idle_gpu();model=model.cuda();optimizer=optimizer_for(model);history=[];start=1;transition=None
    if (directory/'last.pt').exists():
        if not resume:raise ValueError('Explicit --resume required')
        state=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        validate_resume(state,directory,signature);history=state['history'];start=state['epoch']+1
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);transition=state.get('stage_initialization')
        torch.set_rng_state(state['rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_state'])
        reconcile(history,directory,signature)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(directory/'tensorboard'),purge_step=1);restore_training_logs(history,directory,writer)
    save(dict(signature,model_config=model.config,train_views=len(train),dev_views=len(dev),benchmark_used=False,
              confirmation_used=False),directory/'protocol.json')
    save(dev.entries,directory/'dev_observations.json')
    for epoch in range(start,26):
        require_idle_gpu(allow_self=True)
        refine=epoch>20
        if epoch==21:
            rank=choose_stage(history,refine=False);path=directory/f'epoch_{rank["epoch"]:02d}.pt'
            state=torch.load(path,map_location='cpu',weights_only=False);optimizer=start_refinement(model,state)
            transition=dict(source_epoch=rank['epoch'],source_checkpoint_sha256=sha(path),optimizer_reset=True,
                initial_optimizer_state_entries=len(optimizer.state),initial_weights_sha256=fingerprint(model.state_dict()))
            if transition['initial_weights_sha256']!=fingerprint(state['model']):raise ValueError('Stage21 failed to restore best rank weights')
        trained,_=run_epoch(model,loader(train,epoch,shuffle=True),optimizer=optimizer,refine=refine)
        validation,packets=run_epoch(model,loader(dev,0,shuffle=False),refine=refine)
        weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};weight_sha=fingerprint(weights)
        packet_sha=save_packets(packets,directory/f'dev_epoch_{epoch:02d}.npz',epoch=epoch,weights_sha256=weight_sha,roles_sha256=train.roles_sha256)
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
    if best['epoch']!=chosen['epoch'] or sha(directory/'best.pt')!=sha(directory/f'epoch_{chosen["epoch"]:02d}.pt'):
        raise ValueError('Final best alias is not the selected refinement checkpoint')
    for epoch in range(1,26):
        state=torch.load(directory/f'epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False);verify_optimizer_steps(state)
        if any(state[k]!=v for k,v in signature.items()) or fingerprint(state['model'])!=history[epoch-1]['weights_sha256']:
            raise ValueError('Saved V3 epoch lineage or weights changed')
        if sha(directory/f'dev_epoch_{epoch:02d}.npz')!=history[epoch-1]['dev_packets_sha256']:raise ValueError('Saved DEV packet changed')
    save(dict(signature,status='v3_full20plus5_complete_artifact_verification_pending',rank_epochs=20,refine_epochs=5,
        selected_epoch=best['epoch'],checkpoint_sha256=sha(directory/'best.pt'),checkpoint_eligible=best['checkpoint_eligible'],
        code_sha256=hashes,roles_sha256=train.roles_sha256,history_sha256=sha(directory/'history.json'),
        no_benchmark_used=True,independent_confirmation_pending=True),directory/'training_complete.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_training(resume=args.resume)


if __name__=='__main__':main()
