"""Independently rescore all saved A0/A1 traces after frozen full evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import aggregate,score_trace


def equal_metrics(actual,expected):
    for key,value in actual.items():
        if isinstance(value,(int,float)):
            if not math.isclose(value,expected[key],rel_tol=1e-9,abs_tol=1e-8):raise ValueError(f'metric mismatch {key}: {value} vs {expected[key]}')
        elif value is None and expected[key]is not None:raise ValueError('undefined metric mismatch: '+key)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);reference=Path(a.reference);frozen=json.loads((root/'frozen_pair.json').read_text());arm=frozen['arms']['A1_joint_rgb_geometry']
    assert file_sha256(arm['checkpoint'])==arm['checkpoint_sha256']
    records={};selected_epoch=None
    import torch
    for stage,count in [('A',3),('B',3),('C',5)]:
        folder=root/'A1_joint_rgb_geometry/stages'/stage;history=json.loads((folder/'history.json').read_text());complete=json.loads((folder/'complete.json').read_text())
        assert len(history)==count and complete['epochs']==count and not complete['smoke']
        assert all(r['train']['samples']==83140 and r['dev']['samples']==4372 for r in history)
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        assert state['epoch']==count and state['validation_complete'] and state['args']['seed']==0
        assert state['a3_args']['grid_size']==28 and state['protocol']['freeze_bert'] and state['protocol']['freeze_siglip']
        records[stage]=dict(epochs=count,train_states_per_epoch=83140,dev_states_per_epoch=4372,last_checkpoint_sha256=file_sha256(folder/'last.pt'),history_sha256=file_sha256(folder/'history.json'))
    state=torch.load(arm['checkpoint'],map_location='cpu',weights_only=True);selected_epoch=state['epoch'];del state
    audits={}
    for split,count in [('val_seen',2470),('val_unseen',2697)]:
        audits[split]={};paired_manifests=[];paired_data=[]
        for name,folder in [('A0',reference/('full_seen'if split=='val_seen'else 'full_unseen')),('A1',root/'A1_joint_rgb_geometry'/('full_'+split))]:
            report=json.loads((folder/'metrics.json').read_text());settings=report['args'];sys.path.insert(0,settings['sbf_source']if Path(settings['sbf_source']).exists()else '/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source')
            from sbfnav.dataset import CityNavDataset
            dataset=CityNavDataset('/home/20261024_1/Workspace/DATA/sbfnav-data',split,version=report['config']['data']['version'])
            assert len(dataset)==count and report['metrics']['episodes']==count
            live=[file_sha256(dataset.annotation_path),file_sha256(Path(dataset.root)/'cityrefer/objects.json'),file_sha256(Path(dataset.root)/'cityrefer/processed_descriptions.json')] if hasattr(dataset,'root')else [file_sha256(dataset.annotation_path),file_sha256('/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer/objects.json'),file_sha256('/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer/processed_descriptions.json')]
            assert sorted(live)==sorted(report['input_sha256'].values())
            manifest=json.loads((folder/'episodes.json').read_text());assert len(manifest)==count
            assert file_sha256(folder/'episodes.json')==report['episode_manifest_sha256']
            paired_manifests.append(report['episode_manifest_sha256']);paired_data.append(sorted(live))
            rows=[];maps=defaultdict(list);ids=set();max_difference=0.
            with (folder/'predictions.jsonl').open()as stream:
                for position,line in enumerate(stream):
                    row=json.loads(line);index=int(row['index']);truth=dataset[index]
                    assert index not in ids and row['episode_id']==list(truth.episode_id) and row['map_name']==truth.map_name
                    assert index==manifest[position]['index'] and row['episode_id']==manifest[position]['episode_id'];ids.add(index)
                    rescored=score_trace(np.asarray(row['poses']),truth.target_position,truth.trajectory,row['termination'],radius=report['protocol']['success_radius_m'],decision_seconds=row['decision_seconds'])
                    equal_metrics(rescored,row['metrics']);rows.append(rescored);maps[row['map_name']].append(rescored)
                    max_difference=max(max_difference,max(abs(rescored[k]-row['metrics'][k])for k in rescored))
            assert len(rows)==count and len(ids)==count
            equal_metrics(aggregate(rows),report['metrics'])
            for map_name,map_rows in maps.items():equal_metrics(aggregate(map_rows),report['by_map'][map_name])
            if name=='A1':assert report['checkpoint_sha256']==arm['checkpoint_sha256']
            audits[split][name]=dict(episodes=count,all_episode_metrics_rescored=True,all_map_aggregates_recomputed=True,max_metric_difference=max_difference,episode_manifest_sha256=paired_manifests[-1],predictions_sha256=file_sha256(folder/'predictions.jsonl'),metrics_sha256=file_sha256(folder/'metrics.json'))
            print(json.dumps(dict(verified=name,split=split,episodes=count,max_metric_difference=max_difference)),flush=True)
        assert paired_manifests[0]==paired_manifests[1] and paired_data[0]==paired_data[1]
    curves=json.loads(Path('reports/core_pair_20261027/training_curve_verification.json').read_text());assert curves['tensorboard_matches_history']
    atomic_json(dict(complete=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),training=records,selected_stage=arm['selected_stage'],selected_epoch=selected_epoch,selected_checkpoint_sha256=arm['checkpoint_sha256'],frozen_at_utc=frozen['frozen_at_utc'],audits=audits,tensorboard_curves_verified=True,scope='Complete A1 and archived formal A0 reference. All 10334 saved episode traces independently rescored; no checkpoint or inference setting changed.'),a.output)

if __name__=='__main__':main()
