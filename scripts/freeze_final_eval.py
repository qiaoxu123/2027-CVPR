"""Freeze all final candidates using Seen only, before any new Unseen rollout."""
import argparse
import copy
import json
from pathlib import Path
from graphnav.checkpointing import file_sha256,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing final plan overwrite')
    for queue in ('phase20','phase21b','phase22','phase22b'):
        if json.loads(Path('runs/'+queue+'_queue.status.json').read_text())['status']!='complete':
            raise ValueError('required training/Seen controls incomplete: '+queue)
    # Preserve Scratch unless the final P1 variant has a real independent Seen gain.
    def best_r1(path):
        return max(r['seen']['R@1/20m'] for r in json.loads(Path(path).read_text())['history'])
    if best_r1('runs/a3_p1_v3_hard_seed0/report.json')>best_r1('runs/a3_replicate_seed0/report.json'):
        raise ValueError('P1 v3 improves Seen R1: first test its downstream integration before freezing Scratch final systems')
    old=json.loads(Path('runs/phase19b_queue.json').read_text())
    template=old['jobs'][0]['command']
    python=template[0];jobs=[];candidates={}
    def rollout(name,command,seen=None):
        command=copy.copy(command)
        command+=['--evaluation-concurrency','2']
        if '--max-episodes' in command:
            index=command.index('--max-episodes');del command[index:index+2]
        command[command.index('--split')+1]='val_unseen'
        target='runs/final_closedloop_unseen/'+name
        command[command.index('--output')+1]=target
        jobs.append(dict(name='final_unseen_'+name,command=command,log=target+'.log',complete_artifact=target+'/metrics.json'))
        hashes={}
        for flag in ('--checkpoint','--head-checkpoint','--dynamic-checkpoint','--visual-checkpoint'):
            if flag in command:
                path=command[command.index(flag)+1];hashes[flag]=dict(path=path,sha256=file_sha256(path))
        candidates[name]=dict(checkpoints=hashes,seen_run=seen)
    # Geometry and pretraining controls run first; no final candidate is changed
    # after any of these Unseen results become available.
    for name,ck in [('a3_basic',template[template.index('--checkpoint')+1]),('g0_seed0','runs/a3_replicate_seed0/best.pt'),('g1_p1_v2','runs/a3_p1_v2_seed0/best.pt'),('g1_p1_v3_hard','runs/a3_p1_v3_hard_seed0/best.pt')]:
        c=template[:]
        for flag in ('--dynamic-checkpoint','--visual-checkpoint','--observed-cache','--head-checkpoint','--head-execution','--stop-threshold','--visual-control'):
            if flag in c:
                i=c.index(flag);del c[i:i+2]
        c[c.index('--model')+1]='a3';c[c.index('--checkpoint')+1]=ck
        rollout(name,c)
    # Matched G0/G1 trajectory heads are trained with identical supervision and
    # split/seed/epochs. Pretraining rows must not silently use another A3 head.
    for name,ck,head in [('g0_trajectory','runs/a3_replicate_seed0/best.pt','runs/full_seed0_trajectory/best.pt'),('g1_p1_v2_trajectory','runs/a3_p1_v2_seed0/best.pt','runs/p1_v2_matched_trajectory/best.pt'),('g1_p1_v3_trajectory','runs/a3_p1_v3_hard_seed0/best.pt','runs/p1_v3_matched_trajectory/best.pt')]:
        c=template[:]
        c[c.index('--model')+1]='a3-trajectory'
        for flag in ('--dynamic-checkpoint','--visual-checkpoint','--observed-cache','--visual-control'):
            i=c.index(flag);del c[i:i+2]
        c[c.index('--checkpoint')+1]=ck;c[c.index('--head-checkpoint')+1]=head
        rollout(name,c,'runs/p1_matched_trajectory_seen/'+name)
    # Five exact legacy backbone/trajectory module ablations; same old head.
    base=template[:]
    for name,variant,history in [('a3_trajectory',None,False),('gat','graph',False),('visual','visual',False),('joint_current','graph_visual_history',False),('joint_history','graph_visual_history',True)]:
        c=base[:]
        if variant is None:
            c[c.index('--model')+1]='a3-trajectory'
            for flag in ('--dynamic-checkpoint','--visual-checkpoint','--observed-cache','--visual-control'):
                i=c.index(flag);del c[i:i+2]
        else:
            c[c.index('--dynamic-checkpoint')+1]=f'runs/dynamic_belief_v2_{variant}_seed0/best.pt'
            if variant=='graph':c+=['--disable-visual']
            if history:c+=['--visual-history']
        rollout(name,c)
    legacy_seen='runs/closedloop_seen_altitude_guard/joint_current'
    legacy_metrics=json.loads(Path(legacy_seen,'metrics.json').read_text())['metrics']
    seen_scores={'joint_current':{k:legacy_metrics[k] for k in ('SR','SPL','NE')}}
    candidates['joint_current']['seen_run']=legacy_seen
    for seed in range(3):
        name=f'full_seed{seed}';c=base[:]
        for flag,value in [('--checkpoint',f'runs/a3_replicate_seed{seed}/best.pt'),('--head-checkpoint',f'runs/full_seed{seed}_trajectory/best.pt'),('--dynamic-checkpoint',f'runs/full_seed{seed}_dynamic/best.pt')]:c[c.index(flag)+1]=value
        seen='runs/closedloop_seen_altitude_guard/full_seed0' if seed==0 else f'runs/full_seed{seed}_seen'
        m=json.loads(Path(seen,'metrics.json').read_text())['metrics'];seen_scores[name]={k:m[k] for k in ('SR','SPL','NE')}
        rollout(name,c,seen)
    primary=max(seen_scores,key=lambda key:(seen_scores[key]['SR'],seen_scores[key]['SPL'],-seen_scores[key]['NE']))
    static_checkpoints=[('legacy',base[base.index('--checkpoint')+1]),*[(f'scratch_seed{i}',f'runs/a3_replicate_seed{i}/best.pt') for i in range(3)],('p1_v2','runs/a3_p1_v2_seed0/best.pt'),('p1_v3_hard','runs/a3_p1_v3_hard_seed0/best.pt')]
    for name,ck in static_checkpoints:
        target='runs/final_static/'+name
        jobs.append(dict(name='final_static_'+name,command=[python,'scripts/evaluate_frozen_static.py','--checkpoint',ck,'--seen','data/revised_static/graphnav_val_seen.jsonl','--unseen','data/revised_static/graphnav_val_unseen.jsonl','--output',target,'--batch-size','64','--workers','4'],log=target+'.log',complete_artifact=target+'/metrics.json'))
    plan=dict(environment=old['environment'],requires=[],jobs=jobs,evaluation_workers=2,provenance=provenance(),primary_candidate=primary,primary_selection='Among preserved legacy joint-current and three independently trained full-system seeds, maximize full Seen SR, then SPL, then minimize NE. No Unseen selection. Other module rows and negative P1 controls are reporting ablations, not post-Unseen candidate alternatives.',seen_selection_scores=seen_scores,candidates=candidates,protocol='Complete unchanged Revised CityNav ValUnseen2697; same SBF public observations/action bounds/horizon20/success20m and all metric denominators. Current-only joint inference; full-history is a predeclared legacy ablation. Actual observed pixels only; no future RGB or hidden target inputs. All final candidates fixed before first new GraphNav Unseen rollout. Unseen results must not change checkpoints/configuration or be repeatedly tuned.',seed_uncertainty='Three independently trained A3/residual/trajectory seeds, conditional on shared frozen BERT and P2 Large encoder/head. SBF has one available trained checkpoint; do not claim three SBF training seeds.',execution_note='User authorized two independent evaluations in parallel. Concurrent latency is labeled and not compared as isolated GPU speed against the historical isolated SBF runs. No training parallelism or evaluation protocol/model change.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(plan,indent=2));print(json.dumps(dict(plan=str(out),primary_candidate=primary,jobs=len(jobs))))


if __name__=='__main__':main()
