"""Strictly post-rollout stable controller/Belief consistency and paired checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
    tasks={t['index']:t for t in json.loads((a.selection/'tasks_public.json').read_text())}
    previous=[json.loads(x) for x in (a.previous_run/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==192 and len({(r['method'],r['index']) for r in rows})==192
    now={(r['method'],r['index']):r for r in rows};old={(r['method'],r['index']):r for r in previous}
    reproduction={}
    for method in ('Geometry','TrackedPGVB'):
        differences=[]
        for i in range(64):
            x,y=now[method,i],old[method,i]
            if x['termination']!=y['termination'] or not np.array_equal(x['poses'],y['poses']) or x['metrics']!=y['metrics']:
                differences.append(i)
        reproduction[method]=dict(episodes=64,exact_pose_metric_termination_mismatch_ids=differences)
    initial_equal=[]
    for i in range(64):
        group=[now[m,i] for m in ('Geometry','TrackedPGVB','StableTrackedPGVB')]
        initial_equal.append(all(r['poses'][0]==group[0]['poses'][0] and r['step_diagnostics'][0]['rgb_sha256']==group[0]['step_diagnostics'][0]['rgb_sha256'] and r['step_diagnostics'][0]['camera_footprint_world']==group[0]['step_diagnostics'][0]['camera_footprint_world'] for r in group))
    assert all(initial_equal)
    old_horizon=[i for i in range(64) if old['TrackedPGVB',i]['termination']=='horizon']
    recovery=[]
    details=[];context_pairs=0;xy_errors=[];events=Counter()
    for i in range(64):
        r=now['StableTrackedPGVB',i]
        region=unary_union([Polygon(c) for c in tasks[i]['public_polygons']])
        if i in old_horizon:
            recovery.append(dict(index=i,name=r['landmark_name'],old_termination=old['TrackedPGVB',i]['termination'],new_termination=r['termination'],old_SR=old['TrackedPGVB',i]['metrics']['SR'],new_SR=r['metrics']['SR'],new_NE=r['metrics']['NE'],new_steps=r['metrics']['steps'],correct_stop=bool(r['termination']=='policy_stop' and r['metrics']['SR'])))
        previous_held=None
        for t,d in enumerate(r['step_diagnostics']):
            s=d['stable_navigation_goal'];events[s['event']]+=1
            data=np.load(a.run/'decisions'/f'{i:02d}_StableTrackedPGVB'/f'step_{t:02d}.npz')
            ids=data['final_candidate_ids'].astype(int)
            expected=np.stack([(ids%28+.5)/28,(ids//28+.5)/28],axis=-1)
            err=float(abs(expected-data['final_candidate_xy']).max());xy_errors.append(err)
            assert err<1e-6 and int(ids[0])==s['held_goal_id']
            original_ids=data['original_candidate_ids'].astype(int)
            for j,idx in enumerate(ids):
                if idx in original_ids:
                    k=list(original_ids).index(idx)
                    np.testing.assert_array_equal(data['final_candidate_context'][j],data['original_candidate_context'][k]);context_pairs+=1
            changed=not np.array_equal(data['original_logits'],data['final_logits'])
            raw=s['visual_top1_id'];held=s['held_goal_id']
            assert held==previous_held or previous_held is None or s['event']=='replanned'
            previous_held=held
            if s['event']=='replanned':
                assert d['pgvb_active'] and d['landmark_tracking_state']=='tracking'
            bounds=data['bounds'];grid=lambda idx: bounds[:2]+np.array([(idx%28+.5)/28,(idx//28+.5)/28])*(bounds[2:]-bounds[:2])
            # Original candidate zero is geometry; raw RGB Top1 is explicitly audited before stabilization.
            # Public contour supplied only to this offline scoring tool.
            goal_ne=d['public_landmark_goal_distance_m']
            details.append(dict(index=i,step=t,event=s['event'],raw_visual_top1=raw,held_goal=held,
                held_distance_to_raw_m=float(np.linalg.norm(grid(raw)-grid(held))),
                held_differs=int(raw!=held),visual_belief_updated=changed,visual_active=d['pgvb_active'],
                raw_visual_goal_polygon_NE=float(region.distance(Point(grid(raw)))),held_goal_polygon_NE=goal_ne,stop_probability=d['stop_probability'],distance_to_held_goal=d['predicted_goal_distance']))
    out=dict(all_64_initial_pose_rgb_footprint_identical=all(initial_equal),baseline_reproduction=reproduction,previous_horizon_ids=old_horizon,horizon_recovery=recovery,
        recovered_correct_stops=sum(x['correct_stop'] for x in recovery),stable_events=dict(events),
        held_differs_frames=sum(d['held_differs'] for d in details),
        held_despite_updated_visual_frames=sum(d['held_differs'] and d['visual_belief_updated'] for d in details),
        active_visual_frames=sum(d['visual_active'] for d in details),
        unchanged_visual_logits_on_active_frames=sum(d['visual_active'] and not d['visual_belief_updated'] for d in details),
        candidate_context_comparisons=context_pairs,max_normalized_xy_error=max(xy_errors),
        raw_correct_but_held_wrong_frames=sum(d['raw_visual_goal_polygon_NE']<=20 and d['held_goal_polygon_NE']>20 for d in details),
        raw_wrong_but_held_correct_frames=sum(d['raw_visual_goal_polygon_NE']>20 and d['held_goal_polygon_NE']<=20 for d in details),
        candidate_alignment_passed=True,hold_change_only_on_replan=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'stable_interface_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    (a.output/'stable_step_audit.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in details))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
