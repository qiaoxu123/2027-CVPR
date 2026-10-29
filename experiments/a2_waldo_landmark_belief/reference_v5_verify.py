"""Recompute every V5 DEV checkpoint and bind actual saved artifacts/curves."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from .paths import sha,save
from .reference_overlay_context import BASE
from .reference_overlay_data import ReferenceOverlayViews
from .reference_v5_train import DIRECTORY,code_hashes,training_signature,validate_complete_feature_cache
from .reference_v5_protocol import choose_stage,verify_history
from .relation_v3_train import require_idle_gpu,loader
from .relation_v3_model import RelationV3Network
from .relation_v3_epoch import run_epoch
from .relation_v3_verify import PACKET_KEYS,compare_packets,compare_metrics,validate_epoch_state,protected_paths
from .artifact_protocol import verify_curves


def validate_verified_header(proof,binding):
    if (proof.get('status') != 'v5_all25_artifacts_and_dev_verified'
        or any(proof.get(key) != value for key,value in binding.items())
        or proof.get('all_optimizer_steps_verified') is not True
        or proof.get('all_dev_epochs_recomputed') is not True
        or proof.get('selected_epoch') not in range(21,26) or proof.get('rank_selected_epoch') not in range(1,21)
        or [r.get('epoch') for r in proof.get('epochs',[])] != list(range(1,26))
        or not all(r.get('recomputed') is True for r in proof['epochs'])):
        raise ValueError('Incomplete or foreign V5 full artifact verification')


def validate_verified_proof(directory,binding):
    directory = Path(directory)
    proof = json.loads((directory/'artifact_verification.json').read_text())
    validate_verified_header(proof,binding)
    artifacts = proof['artifact_sha256']
    if set(artifacts) != set(protected_paths(directory)) or any(sha(directory/name) != digest for name,digest in artifacts.items()):
        raise ValueError('Verified V5 artifact bytes changed')
    for record in proof['epochs']:
        if (record['checkpoint_sha256'] != artifacts[f'epoch_{record["epoch"]:02d}.pt']
            or record['packet_sha256'] != artifacts[f'dev_epoch_{record["epoch"]:02d}.npz']):
            raise ValueError('V5 verified epoch manifest changed')
    if (artifacts['best.pt'] != artifacts[f'epoch_{proof["selected_epoch"]:02d}.pt']
        or artifacts['best_rank.pt'] != artifacts[f'epoch_{proof["rank_selected_epoch"]:02d}.pt']
        or artifacts['last.pt'] != artifacts['epoch_25.pt']):
        raise ValueError('V5 best/rank/last aliases changed')
    return proof


def verify_artifacts(directory=DIRECTORY):
    signature = training_signature();require_idle_gpu()
    directory = Path(directory)
    marker = json.loads((directory/'training_complete.json').read_text())
    history = json.loads((directory/'history.json').read_text());verify_history(history)
    if (marker.get('status') != 'v5_full20plus5_complete_artifact_verification_pending'
        or any(marker.get(key) != value for key,value in signature.items())
        or marker['rank_epochs'] != 20 or marker['refine_epochs'] != 5
        or marker['history_sha256'] != sha(directory/'history.json')):
        raise ValueError('V5 completed training provenance changed')
    protocol = json.loads((directory/'protocol.json').read_text())
    if (any(protocol.get(key) != value for key,value in signature.items())
        or protocol['train_views'] != 33440 or protocol['dev_views'] != 5647
        or protocol['benchmark_used'] or protocol['confirmation_used']):
        raise ValueError('V5 training protocol changed')
    chosen,rank = choose_stage(history,refine=True),choose_stage(history,refine=False)
    best = directory/'best.pt'
    if (marker['selected_epoch'] != chosen['epoch'] or marker['checkpoint_sha256'] != sha(best)
        or marker['checkpoint_eligible'] != chosen['dev']['eligible']
        or sha(best) != sha(directory/f'epoch_{chosen["epoch"]:02d}.pt')
        or sha(directory/'best_rank.pt') != sha(directory/f'epoch_{rank["epoch"]:02d}.pt')
        or sha(directory/'last.pt') != sha(directory/'epoch_25.pt')):
        raise ValueError('V5 checkpoint selection or aliases changed')
    dev = ReferenceOverlayViews('dev')
    if json.loads((directory/'dev_observations.json').read_text()) != dev.entries:
        raise ValueError('V5 DEV identities or overlay sources changed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require_idle_gpu();model=RelationV3Network().cuda()
    names=[name for name,param in model.named_parameters() if param.requires_grad];reports=[]
    for record in history:
        if training_signature() != signature:
            raise ValueError('V5 sources/proofs changed during artifact verification')
        require_idle_gpu(allow_self=True)
        epoch=record['epoch'];path=directory/f'epoch_{epoch:02d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        validate_epoch_state(state,record,history,signature,directory,names,model.config)
        if state['model_config'] != protocol['model_config']:
            raise ValueError('V5 model configuration changed')
        packet_path=directory/f'dev_epoch_{epoch:02d}.npz'
        if sha(packet_path) != record['dev_packets_sha256']:
            raise ValueError('V5 DEV packet changed')
        with np.load(packet_path,allow_pickle=False) as packet:
            if (int(packet['epoch']) != epoch or str(packet['weights_sha256']) != record['weights_sha256']
                or str(packet['roles_sha256']) != signature['roles_sha256']):
                raise ValueError('V5 DEV packet epoch/weights/roles changed')
            saved={key:packet[key].copy() for key in PACKET_KEYS}
        model.load_state_dict(state['model'],strict=True)
        actual,packets=run_epoch(model,loader(dev,0,shuffle=False),refine=epoch>20)
        compare_packets(saved,packets);compare_metrics(record['dev'],actual)
        reports.append(dict(epoch=epoch,checkpoint_sha256=sha(path),packet_sha256=sha(packet_path),recomputed=True))
        print(json.dumps(dict(epoch=epoch,status='V5_DEV_and_optimizer_verified')),flush=True)
    if training_signature() != signature:
        raise ValueError('V5 source changed at verification completion')
    proof=dict(signature,status='v5_all25_artifacts_and_dev_verified',checkpoint_sha256=sha(best),
        history_sha256=sha(directory/'history.json'),selected_epoch=chosen['epoch'],rank_selected_epoch=rank['epoch'],
        checkpoint_eligible=chosen['dev']['eligible'],curves=verify_curves(history,directory),
        artifact_sha256={name:sha(directory/name) for name in protected_paths(directory)},
        all_optimizer_steps_verified=True,all_dev_epochs_recomputed=True,epochs=reports,
        no_benchmark_used=True,localization_only_not_navigation_gain=True)
    validate_verified_header(proof,signature)
    save(proof,directory/'artifact_verification.json')


def export_curves(directory=DIRECTORY):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    binding=training_signature()
    proof=validate_verified_proof(directory,binding)
    history=json.loads((directory/'history.json').read_text());verify_history(history)
    if proof['history_sha256'] != sha(directory/'history.json'):
        raise ValueError('V5 curves require verified actual history')
    x=[r['epoch'] for r in history];fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for source in ('real','teacher'):
        for role in ('train','dev'):
            axes[0,0].plot(x,[r[role][source]['loss'] for r in history],label=f'{role}/{source}')
        axes[0,1].plot(x,[r['dev'][source]['proposal_r1'] for r in history],label=f'proposal/{source}')
        axes[0,1].plot(x,[r['dev'][source]['a0_r1'] for r in history],linestyle='--',label=f'A0/{source}')
    for key in ('corrected','broken'):
        axes[1,0].plot(x,[r['dev']['real'][key] for r in history],label=key)
    selected=[r['dev']['selected_point']['summary']['mean_r1_difference'] if r['epoch']>20 and r['dev']['selected_point'] else np.nan for r in history]
    axes[1,1].plot(x,selected,marker='o',label='eligible realDEV gate')
    if not np.isfinite(selected).any():
        axes[1,1].text(.5,.5,'No eligible refinement gate',ha='center',transform=axes[1,1].transAxes)
    titles=('Source-weighted loss','DEV proposal localization R1 (not SR)',
            'RealDEV ungated fixed .25 diagnostic','Selected DEV gate localization R1 difference')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title);ax.set_xlabel('epoch');ax.axvline(20.5,color='grey',linestyle=':');ax.legend();ax.grid(alpha=.2)
    fig.savefig(directory/'training_curves.png',dpi=180);fig.savefig(directory/'training_curves.svg');plt.close(fig)
    save(dict(history_sha256=sha(directory/'history.json'),png_sha256=sha(directory/'training_curves.png'),
              svg_sha256=sha(directory/'training_curves.svg'),scope='Internal localization/loss, not final navigation'),directory/'curve_manifest.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--curves',action='store_true');args=parser.parse_args()
    if args.curves: export_curves();return
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_artifacts()


if __name__=='__main__': main()
