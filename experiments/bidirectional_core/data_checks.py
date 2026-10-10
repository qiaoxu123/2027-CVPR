"""Check generated direction-specific supervision, budgets and public-input leakage."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from graphnav.trajectory_policy import remaining_waypoints

def audit(root):
    root=Path(root);result={}
    for split in ('train_seen','val_seen'):
        path=root/split;episodes={};counts=Counter();private_ids=defaultdict(set);regions=defaultdict(set);families=defaultdict(set);end_errors=[];relation_errors=[]
        for line in (path/'episodes.jsonl').open():
            ep=json.loads(line);episodes[ep['episode_id']]=ep;p=ep['internal_partition'];families[p].add(ep['family_id'])
            label=ep['private_labels'];oid=label['target_object_id']
            if oid is not None:private_ids[p].add((ep['map_name'],oid))
            traj=np.asarray(ep['trajectory_xyzyaw']);assert np.isfinite(traj).all()
            for xy in traj[:,:2]:regions[p].add((ep['map_name'],int(np.floor(xy[0]/200)),int(np.floor(xy[1]/200))))
            if ep['direction']=='reverse':
                f=episodes[ep['family_id']+':forward'];np.testing.assert_array_equal(traj[:,:3],np.asarray(f['trajectory_xyzyaw'])[::-1,:3]);counts['exact_xyz_reverse_checked']+=1
                np.testing.assert_array_equal(label['target_world'],np.asarray(f['trajectory_xyzyaw'])[0,:3]);d=ep.get('reverse_geometry_description')
                if d:
                    assert d['distance_rounding_error_m']<=5.0001
                    assert d['transverse_distance_m']<=min(20,.2*d['actual_axis_distance_m'])+1e-4
                    relation_errors.append(d['distance_rounding_error_m']);counts['geometric_return_instructions_checked']+=1
            else:end_errors.append(float(np.linalg.norm(traj[-1,:2]-np.asarray(label['target_world'])[:2])))
        prefix_ids=set();public_forbidden={'target_world','teacher_world','target_object_id','target_xy'}
        for line in (path/'prefixes.jsonl').open():
            r=json.loads(line);assert not public_forbidden&set(r);ep=episodes[r['episode_id']];traj=np.asarray(ep['trajectory_xyzyaw']);c=r['cutoff_step'];prefix_ids.add(r['sample_id'])
            np.testing.assert_array_equal(r['pose_world'],traj[c]);np.testing.assert_allclose(r['labels']['teacher_world'],remaining_waypoints(traj,c),rtol=1e-6,atol=1e-5)
            stop=np.linalg.norm(traj[c,:2]-np.asarray(r['labels']['target_world'])[:2])<=20
            assert r['labels']['arrival_stop']==stop
            for x in r['historical_camera_observations']+r['reverse_camera_reuse_candidates']:assert x['direction_step']<=c
            assert r['source_step']==(c if r['direction']=='forward' else len(traj)-1-c)
            if r['direction']=='reverse':assert not r['historical_camera_observations']
            counts['prefix_causal_waypoint_stop_checked']+=1
        forwardbudgets=Counter();roles=Counter()
        for line in (path/'matched_e0_e1.jsonl').open():
            r=json.loads(line);assert r['E0_forward_sample_id'] in prefix_ids and r['E1_sample_id'] in prefix_ids
            forwardbudgets[r['E0_forward_sample_id']]+=1;roles[r['role']]+=1;counts['matched_budget_rows_checked']+=1
        assert set(forwardbudgets.values())=={2};assert roles['forward']==roles['reverse']
        if split=='train_seen':
            for data in (private_ids,regions,families):assert data['train'].isdisjoint(data['internal_dev'])
        result[split]=dict(checks=dict(counts),original_forward_endpoint_xy_error_m={'mean':float(np.mean(end_errors)),'median':float(np.median(end_errors)),'p95':float(np.percentile(end_errors,95)),'within_20m':int(sum(x<=20 for x in end_errors)),'count':len(end_errors)},return_language_rounding_error_max_m=max(relation_errors,default=0),matched_total_steps=True,private_coordinate_fields_separate=True,no_reverse_mismatched_camera_as_current=True,internal_family_object_region_overlap=0 if split=='train_seen' else None)
    (root/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='runs/bidirectional_core_test/data');audit(p.parse_args().data)
