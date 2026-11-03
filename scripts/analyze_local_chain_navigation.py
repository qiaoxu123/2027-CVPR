"""TRAIN-only post-hoc action/Goal/Stop attribution from frozen real rollouts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from graphnav.checkpointing import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','data-root','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    data=CityNavDataset(a.data_root,'train_seen',version='processed_citynav')
    result={}
    for path in sorted(a.run.glob('*/predictions.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()];details=[]
        for row in rows:
            target=data[row['index']].target_position[:2]
            poses=np.array(row['poses']);ds=row['step_diagnostics'];goals=np.array([d['selected_goal_xy'] for d in ds])
            distance=np.linalg.norm(poses[:,:2]-target,axis=-1);goal_ne=np.linalg.norm(goals-target,axis=-1)
            inside=distance<=20;entered=bool(inside.any());first=int(np.flatnonzero(inside)[0]) if entered else None
            jitter=np.linalg.norm(np.diff(goals,axis=0),axis=-1)
            known_first=first is not None and first<len(ds)
            first_goal_correct=bool(goal_ne[first]<=20) if known_first else None
            # Overlapping explanations: never force one unique failure cause.
            blocks=[]
            for step,d in enumerate(ds):
                if distance[step]>20:continue
                reason=[]
                if d['stop_probability']<.3:reason.append('learned_probability_below_threshold')
                if d.get('predicted_goal_distance',0)>1e-3:reason.append('predicted_goal_arrival_guard')
                if not d.get('stop_altitude_guard',True):reason.append('altitude_guard')
                blocks.append(dict(step=step,goal_correct=bool(goal_ne[step]<=20),reasons=reason))
            details.append(dict(index=row['index'],episode_id=row['episode_id'],success=bool(row['metrics']['SR']),
                termination=row['termination'],ever_entered20m=entered,oracle_stop_recovers=bool(entered and not row['metrics']['SR']),
                first_entry_step=first,first_entry_goal_correct=first_goal_correct,
                first_entry_goal_unknown=bool(entered and not known_first),last_decision_goal_correct=bool(goal_ne[-1]<=20),
                never_goal_correct=bool(not (goal_ne<=20).any()),goal_lost=int(((goal_ne[:-1]<=20)&(goal_ne[1:]>20)).sum()),
                large_goal_jumps20m=int((jitter>20).sum()),median_goal_jump_m=float(np.median(jitter)) if len(jitter) else 0.,
                stop_blocks_inside_target=blocks,final_ne_m=float(distance[-1])))
        fail=[r for r in details if not r['success']]
        result[path.parent.name]=dict(episodes=len(details),failures=len(fail),
            failed_never_entered20m=sum(not r['ever_entered20m'] for r in fail),
            failed_entered_with_wrong_goal=sum(r['first_entry_goal_correct'] is False for r in fail),
            failed_entered_with_correct_goal=sum(r['first_entry_goal_correct'] is True for r in fail),
            failed_entry_goal_unknown=sum(r['first_entry_goal_unknown'] for r in fail),
            oracle_stop_recoverable=sum(r['oracle_stop_recovers'] for r in details),
            failed_last_decision_goal_correct=sum(r['last_decision_goal_correct'] for r in fail),
            never_correct_goal_failures=sum(r['never_goal_correct'] for r in fail),
            goal_lost=sum(r['goal_lost'] for r in details),large_goal_jumps20m=sum(r['large_goal_jumps20m'] for r in details),
            episodes_stop_arrival_blocked_while_goal_correct=sum(any(b['goal_correct'] and 'predicted_goal_arrival_guard' in b['reasons'] for b in r['stop_blocks_inside_target']) for r in details),
            details=details)
    original={r['index']:r for r in result['original']['details']}
    for method,summary in result.items():
        summary['sr_corrections_vs_original']=sum(not original[r['index']]['success'] and r['success'] for r in summary['details'])
        summary['sr_harms_vs_original']=sum(original[r['index']]['success'] and not r['success'] for r in summary['details'])
    atomic_json(dict(scope='post-hoc real TRAIN closed-loop attribution; overlapping causes, not proof of unique causation',
                     methods=result,warning='Last-decision Goal may precede the final action. A final pose without a decision has unknown Goal/Stop state.'),a.output)
    print(json.dumps({m:{k:v for k,v in r.items() if k!='details'} for m,r in result.items()},indent=2))


if __name__=='__main__':main()
