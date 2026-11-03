"""Posthoc interface and TRAIN-distribution audit, never an inference input."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import shapely
from graphnav.landmark_navigation_diagnostic import public_region


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--train-classifications',type=Path,required=True)
    a=p.parse_args();root=a.report
    rows=[json.loads(x) for x in (root/'results/predictions.jsonl').read_text().splitlines()]
    tasks=json.loads((root/'tasks_public.json').read_text())
    train=json.loads(a.train_classifications.read_text())
    own=[r for r in train if set(r.get('analysis_target_ids',[])) & {i for m in r.get('mentions',[]) for i in m['public_ids']}]
    e2=[r for r in rows if r['method']=='E2'];steps=[d for r in e2 for d in r['step_diagnostics']]
    audit=dict(train_tasks=len(train),train_own_target_named_mentions=len(own),
        train_own_target_named_mention_classes=dict(Counter(r['category'] for r in own)),
        train_parsed_direct_identity=sum(r['category']=='direct_identity' for r in train),
        other_types=dict(Counter('+'.join(t['public_types']) for t in tasks if t['category']=='Other')),
        e2_tasks_goal_changed=int(sum(np.max(np.linalg.norm(np.array([d['selected_goal_xy'] for d in r['step_diagnostics']])-np.array(r['step_diagnostics'][0]['selected_goal_xy']),axis=-1))>1e-3 for r in e2)),
        e2_steps=len(steps),e2_steps_selected_goal_matches_static_top1=int(sum(np.linalg.norm(np.array(d['selected_goal_xy'])-np.array(d['static_top16_world_xy'][0]))<1e-3 for d in steps)),
        e2_max_observed_residual_abs=max(d['residual_max_abs'] for d in steps),
        decision_median_ms={m:float(np.median([s for r in rows if r['method']==m for s in r['decision_seconds']])*1000) for m in ('E1','E2')},
        strict_polygon_inside_threshold_m=.001,source_target_geometry_used=False,
        caveat='TRAIN identity metadata used only in this distribution audit; automatic own-name counts are not human semantic gold.')
    failures=[]
    static_initial=[]
    for r in e2:
        t=tasks[r['index']];region=public_region(t['public_polygons'],t['world_bounds']);bounds=np.array(t['world_bounds'])
        static_distance=float(shapely.distance(shapely.Point(r['step_diagnostics'][0]['static_top16_world_xy'][0]),region))
        static_initial.append((static_distance<=20,r['goal_analysis']['initial_ne_m']<=20))
        if r['metrics']['SR']:continue
        axis=(np.arange(28)+.5)/28;u,v=np.meshgrid(axis,axis)
        xy=np.stack([u.ravel(),v.ravel()],-1)*(bounds[2:]-bounds[:2])+bounds[:2]
        distance=shapely.distance(shapely.points(xy),region)
        failures.append(dict(index=r['index'],name=r['landmark_name'],min_available_grid_polygon_distance_m=float(distance.min()),
            grid_cells_within20=int((distance<=20).sum()),initial_predicted_goal_polygon_distance_m=r['goal_analysis']['initial_ne_m'],
            final_NE=r['metrics']['NE'],OSR=r['metrics']['OSR']))
    audit['failure_grid_representability']=failures
    audit['e2_initial_static_correct']=sum(s for s,d in static_initial)
    audit['e2_initial_dynamic_correct']=sum(d for s,d in static_initial)
    audit['e2_initial_visual_corrections']=sum(d and not s for s,d in static_initial)
    audit['e2_initial_visual_harms']=sum(not d and s for s,d in static_initial)
    (root/'interface_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
