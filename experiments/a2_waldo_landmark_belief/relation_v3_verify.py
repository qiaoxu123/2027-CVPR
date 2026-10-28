"""Recompute all25 DEV checkpoints, verify artifacts, export actual curves."""
import argparse,fcntl,json,math
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .relation_v3_train import DIRECTORY,code_hashes,validate_audit,fingerprint,require_idle_gpu,verify_optimizer_steps,loader,training_signature
from .relation_v3_model import RelationV3Network
from .relation_v3_data import RelationV3Views
from .relation_v3_epoch import run_epoch
from .relation_v3_protocol import verify_history,choose_stage
from .artifact_protocol import verify_curves

PACKET_KEYS=('scores','available','before_ne','after_ne','proposal_ne','is_real')


def compare_packets(saved,actual):
    for key in PACKET_KEYS:
        a=np.asarray(saved[key]);b=np.asarray(actual[key])
        if a.shape!=b.shape:raise ValueError('DEV packet shape changed: '+key)
        if a.dtype==np.bool_:
            if not np.array_equal(a,b):raise ValueError('DEV boolean packet changed: '+key)
        elif not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=2e-5,rtol=1e-6):
            raise ValueError('Recomputed DEV packet differs: '+key)
        if key in ('before_ne','after_ne','proposal_ne') and not np.array_equal(a<=20,b<=20):
            raise ValueError('Recomputed localization success boundary changed: '+key)


def compare_metrics(saved,actual,path='dev'):
    if isinstance(saved,dict):
        if not isinstance(actual,dict) or set(saved)!=set(actual):raise ValueError('Metric schema changed: '+path)
        for key,value in saved.items():
            if key!='seconds':compare_metrics(value,actual[key],path+'.'+key)
    elif isinstance(saved,list):
        if not isinstance(actual,list) or len(saved)!=len(actual):raise ValueError('Metric list changed: '+path)
        for i,(a,b) in enumerate(zip(saved,actual)):compare_metrics(a,b,path+f'[{i}]')
    elif isinstance(saved,(int,bool)) or saved is None or isinstance(saved,str):
        if saved!=actual:raise ValueError('Metric changed: '+path)
    elif isinstance(saved,float):
        if not isinstance(actual,(float,int)) or not math.isfinite(saved) or not math.isfinite(actual) or not math.isclose(saved,actual,rel_tol=1e-6,abs_tol=2e-5):
            raise ValueError('Metric changed: '+path)
    else:raise ValueError('Unsupported verification metric: '+path)


def validate_epoch_state(state,record,history,signature,directory,names,model_config):
    epoch=record['epoch']
    if (any(state.get(k)!=v for k,v in signature.items()) or state['epoch']!=epoch or state['history']!=history[:epoch]
        or state['model_config']!=model_config or state['trainable_parameter_names']!=names
        or fingerprint(state['model'])!=record['weights_sha256']
        or state['checkpoint_eligible']!=bool(epoch>20 and record['dev']['eligible'])):
        raise ValueError('Epoch recipe/history/weights/eligibility provenance changed')
    if epoch<=20:
        if state['stage_initialization'] is not None:raise ValueError('Unexpected ranking stage transition')
    else:
        rank=choose_stage(history,refine=False)
        expected=dict(source_epoch=rank['epoch'],source_checkpoint_sha256=sha(directory/f'epoch_{rank["epoch"]:02d}.pt'),
            optimizer_reset=True,initial_optimizer_state_entries=0,initial_weights_sha256=rank['weights_sha256'])
        if state['stage_initialization']!=expected:raise ValueError('Refinement did not restore selected rank/reset optimizer')
    verify_optimizer_steps(state)


def protected_paths(directory):
    names=['protocol.json','history.json','training_complete.json','last.pt','best.pt','best_rank.pt','dev_observations.json']
    names += [f'epoch_{e:02d}.pt' for e in range(1,26)]+[f'dev_epoch_{e:02d}.npz' for e in range(1,26)]
    events=sorted(directory.joinpath('tensorboard').glob('events.out.tfevents.*'))
    if not events:raise ValueError('TensorBoard event artifacts required')
    return names+[str(p.relative_to(directory)) for p in events]


def validate_verified_proof(directory,hashes,roles):
    directory=Path(directory);proof=json.loads((directory/'artifact_verification.json').read_text())
    if (proof['status']!='v3_all25_artifacts_and_dev_verified' or proof['code_sha256']!=hashes or proof['roles_sha256']!=roles
        or not proof['all_optimizer_steps_verified'] or not proof['all_dev_epochs_recomputed']
        or [e['epoch'] for e in proof['epochs']]!=list(range(1,26))):raise ValueError('Incomplete verified V3 proof')
    bindings=proof['artifact_sha256']
    if set(bindings)!=set(protected_paths(directory)) or any(sha(directory/name)!=digest for name,digest in bindings.items()):
        raise ValueError('Previously verified V3 artifact changed')
    for e in proof['epochs']:
        if e['checkpoint_sha256']!=bindings[f'epoch_{e["epoch"]:02d}.pt'] or e['packet_sha256']!=bindings[f'dev_epoch_{e["epoch"]:02d}.npz']:
            raise ValueError('Verified epoch manifest changed')
    if (proof['selected_epoch'] not in range(21,26) or proof['rank_selected_epoch'] not in range(1,21)
        or bindings['best.pt']!=bindings[f'epoch_{proof["selected_epoch"]:02d}.pt']
        or bindings['best_rank.pt']!=bindings[f'epoch_{proof["rank_selected_epoch"]:02d}.pt']
        or bindings['last.pt']!=bindings['epoch_25.pt']):raise ValueError('Verified checkpoint alias changed')
    return proof


def verify_artifacts(directory=DIRECTORY):
    require_idle_gpu();signature=training_signature();hashes=signature['code_sha256'];directory=Path(directory)
    marker=json.loads((directory/'training_complete.json').read_text());history=json.loads((directory/'history.json').read_text());verify_history(history)
    if (any(marker.get(k)!=v for k,v in signature.items()) or marker['history_sha256']!=sha(directory/'history.json')
        or marker['rank_epochs']!=20 or marker['refine_epochs']!=5):
        raise ValueError('Completed V3 training source changed')
    protocol=json.loads((directory/'protocol.json').read_text())
    if (any(protocol.get(k)!=v for k,v in signature.items()) or protocol['train_views']!=33440 or protocol['dev_views']!=5647
        or protocol['benchmark_used'] or protocol['confirmation_used']):raise ValueError('Training protocol changed')
    chosen=choose_stage(history,refine=True);best=directory/'best.pt'
    rank=choose_stage(history,refine=False)
    if (marker['selected_epoch']!=chosen['epoch'] or marker['checkpoint_sha256']!=sha(best)
        or marker['checkpoint_eligible']!=chosen['dev']['eligible'] or sha(best)!=sha(directory/f'epoch_{chosen["epoch"]:02d}.pt')
        or sha(directory/'best_rank.pt')!=sha(directory/f'epoch_{rank["epoch"]:02d}.pt')
        or sha(directory/'last.pt')!=sha(directory/'epoch_25.pt')):raise ValueError('Best/last alias/selection changed')
    dev=RelationV3Views('dev')
    if json.loads((directory/'dev_observations.json').read_text())!=dev.entries:raise ValueError('Saved DEV identities changed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require_idle_gpu();model=RelationV3Network().cuda();reports=[]
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    for record in history:
        require_idle_gpu(allow_self=True);epoch=record['epoch'];path=directory/f'epoch_{epoch:02d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        validate_epoch_state(state,record,history,signature,directory,names,model.config)
        if state['model_config']!=protocol['model_config']:raise ValueError('Model configuration changed')
        packet_path=directory/f'dev_epoch_{epoch:02d}.npz'
        if sha(packet_path)!=record['dev_packets_sha256']:raise ValueError('Epoch packet changed')
        with np.load(packet_path,allow_pickle=False) as packet:
            if (int(packet['epoch'])!=epoch or str(packet['weights_sha256'])!=record['weights_sha256']
                or str(packet['roles_sha256'])!=marker['roles_sha256']):raise ValueError('Packet is bound to another epoch/role')
            saved={k:packet[k].copy() for k in PACKET_KEYS}
        model.load_state_dict(state['model'],strict=True)
        actual,packets=run_epoch(model,loader(dev,0,shuffle=False),refine=epoch>20)
        compare_packets(saved,packets);compare_metrics(record['dev'],actual)
        reports.append(dict(epoch=epoch,checkpoint_sha256=sha(path),packet_sha256=sha(packet_path),recomputed=True))
        print(json.dumps(dict(epoch=epoch,status='DEV_and_optimizer_verified')),flush=True)
    if hashes!=code_hashes():raise ValueError('Code changed during verification')
    curves=verify_curves(history,directory)
    proof=dict(status='v3_all25_artifacts_and_dev_verified',checkpoint_sha256=sha(best),code_sha256=hashes,
        roles_sha256=marker['roles_sha256'],history_sha256=sha(directory/'history.json'),selected_epoch=chosen['epoch'],
        rank_selected_epoch=rank['epoch'],curves=curves,artifact_sha256={name:sha(directory/name) for name in protected_paths(directory)},
        checkpoint_eligible=chosen['dev']['eligible'],all_optimizer_steps_verified=True,all_dev_epochs_recomputed=True,epochs=reports,
        no_benchmark_used=True,localization_only_not_navigation_gain=True)
    save(proof,directory/'artifact_verification.json')
    save(proof,ROOT/'reports/a2_execution_20261027/RELATION_V3_TRAINING_VERIFIED.json')


def export_curves(directory=DIRECTORY):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((directory/'history.json').read_text());verify_history(history)
    proof=validate_verified_proof(directory,code_hashes(),sha(BASE/'revision_v3/roles.json'))
    if proof['history_sha256']!=sha(directory/'history.json'):raise ValueError('Curves require verified actual history')
    x=[r['epoch'] for r in history];fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for source in ('real','teacher'):
        for role in ('train','dev'):
            axes[0,0].plot(x,[r[role][source]['loss'] for r in history],label=f'{role}/{source}')
        axes[0,1].plot(x,[r['dev'][source]['proposal_r1'] for r in history],label=f'proposal/{source}')
        axes[0,1].plot(x,[r['dev'][source]['a0_r1'] for r in history],linestyle='--',label=f'A0/{source}')
    for key in ('corrected','broken'):
        axes[1,0].plot(x,[r['dev']['real'][key] for r in history],label=key)
    y=[r['dev']['selected_point']['summary']['mean_r1_difference'] if r['epoch']>20 and r['dev']['selected_point'] else np.nan for r in history]
    axes[1,1].plot(x,y,marker='o',label='eligible realDEV gate')
    if not np.isfinite(y).any():axes[1,1].text(.5,.5,'No eligible refinement gate',ha='center',transform=axes[1,1].transAxes)
    for ax,title in zip(axes.flat,('Source-weighted loss','DEV proposal localization R1 (not SR)',
                                  'RealDEV ungated fixed .25 diagnostic','Selected DEV gate localization R1 difference')):
        ax.set_title(title);ax.set_xlabel('epoch');ax.axvline(20.5,color='grey',linestyle=':');ax.legend();ax.grid(alpha=.2)
    directory.mkdir(parents=True,exist_ok=True);fig.savefig(directory/'training_curves.png',dpi=180);fig.savefig(directory/'training_curves.svg');plt.close(fig)
    save(dict(history_sha256=sha(directory/'history.json'),png_sha256=sha(directory/'training_curves.png'),svg_sha256=sha(directory/'training_curves.svg'),
              scope='Internal localization and loss only, not final navigation metrics'),directory/'curve_manifest.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--curves',action='store_true');args=parser.parse_args()
    if args.curves:export_curves();return
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_artifacts()


if __name__=='__main__':main()
