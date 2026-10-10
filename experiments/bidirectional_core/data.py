"""Auditable CityFlight path reversal metadata; labels remain separate from public inputs.

Reversed paths are offline traversal supervision of measured positions, not recorded
return flights. A changed heading never reuses a forward camera crop as a current
reverse image. Original RGB features can be reused only with explicit camera/source
orientation and direction-specific observation times.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from shapely.geometry import Point, Polygon
from graphnav.bidirectional import cardinal_relation
from graphnav.trajectory_policy import remaining_waypoints

DEFAULT_DATA = '/home/rental/20260922_1/Workspace/DATA/sbfnav-data'
DEFAULT_SBF = '/home/rental/20260922_1/Workspace/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def pose4(pose):
    p = np.asarray(pose, dtype=np.float64)
    if len(p) == 5: yaw = p[3]
    elif len(p) == 6: yaw = math.atan2(p[4], p[3])
    elif len(p) == 4: yaw = p[3]
    else: raise ValueError('CityFlight pose must have 4/5/6 fields')
    if not np.isfinite(p).all(): raise ValueError('nonfinite source pose')
    return np.array([*p[:3], yaw])

def reverse_trajectory(trajectory):
    """Reverse xyz exactly, recomputing travel heading with stationary-run handling."""
    source = np.asarray([pose4(p) for p in trajectory])
    if len(source) < 2: raise ValueError('at least two measured trajectory positions required')
    result = source[::-1].copy()
    # Preserve observed altitude. Vertical-only intervals use the next movement;
    # the endpoint uses the latest incoming direction, never a future source label.
    previous = float((source[-1, 3] + np.pi + np.pi) % (2*np.pi) - np.pi)
    for i in range(len(result)):
        delta = None
        for j in range(i + 1, len(result)):
            d = result[j, :2] - result[i, :2]
            if np.linalg.norm(d) > 1e-6: delta = d; break
        if delta is not None: previous = math.atan2(delta[1], delta[0])
        result[i, 3] = previous
    return result

def fixed_prefixes(length):
    return sorted({0, round(.25*(length-1)), round(.5*(length-1)), round(.75*(length-1)), length-1})

def catalog_geometry(objects):
    out = {}
    for map_name, entries in objects.items():
        valid = []
        for oid, item in entries.items():
            contour = np.asarray(item.get('contour', []), np.float64)
            if contour.ndim != 2 or contour.shape[1:] != (2,) or len(contour) < 3 or not np.isfinite(contour).all(): continue
            poly = Polygon(contour)
            if not poly.is_valid or poly.area <= 0: continue
            center = poly.centroid
            if not poly.covers(center): center = poly.representative_point()
            valid.append(dict(id=str(oid), object_type=item.get('object_type','Unknown'), name=item.get('name',''), polygon=poly, center=np.asarray(center.coords[0]), item=item))
        out[map_name] = valid
    return out

def ground_start(start_xy, entries, margin=2.):
    """Only unique, interior reliable surface annotations; proximity is not identity."""
    p = Point(start_xy)
    matches = [e for e in entries if e['object_type'] in ('Building','Parking','Ground') and e['polygon'].contains(p) and e['polygon'].boundary.distance(p) >= margin]
    return (matches[0] if len(matches) == 1 else None), [e['id'] for e in matches]

def geometry_description(target, entries, start_xy=None):
    """Describe the actual launch point; never substitute a large polygon centroid.

    Candidate polygons intersecting the explicitly approximate relation region
    are possible positives, rather than arbitrarily chosen semantic negatives.
    Names are unchanged human map metadata, not verified appearance captions.
    """
    if start_xy is None:
        if target is None: return None
        start_xy=target['center']
    start_xy=np.asarray(start_xy,dtype=float)
    options=[]
    for anchor in entries:
        if not str(anchor['name']).strip() or (target is not None and anchor['id']==target['id']): continue
        relation=cardinal_relation(anchor['center'],start_xy)
        if relation is not None:
            direction,distance=relation
            options.append((distance,anchor['id'],direction,anchor))
    if not options:return None
    distance,_,direction,anchor=min(options,key=lambda x:x[:2])
    rounded=max(10,10*round(distance/10))
    assert abs(rounded-distance)<=5.0001
    delta=start_xy-anchor['center'];axis=0 if direction in ('east','west') else 1
    transverse=abs(float(delta[1-axis]))
    assert transverse<=min(20.,.2*distance)+1e-4
    location='launch point'
    positives=[];negatives=[];unknown=[]
    if target is not None:
        typ={'Parking':'parking area','Ground':'ground region','Building':'building'}.get(target['object_type'],target['object_type'].lower())
        location=f'launch point on the {typ}'
        from shapely.geometry import box
        halfside=min(20.,.2*rounded);sign=1 if direction in ('east','north') else -1
        low=anchor['center'].copy();high=anchor['center'].copy()
        low[axis]+=min(sign*(rounded-5),sign*(rounded+5));high[axis]+=max(sign*(rounded-5),sign*(rounded+5))
        low[1-axis]-=halfside;high[1-axis]+=halfside
        relation_region=box(*low,*high)
        for obj in entries:
            if obj['object_type']!=target['object_type'] or obj['id']==anchor['id']:continue
            coords=np.asarray(obj['polygon'].exterior.coords)
            if obj['polygon'].intersects(relation_region):positives.append(obj['id'])
            elif np.max(sign*(coords[:,axis]-anchor['center'][axis]))<=-5:negatives.append(obj['id'])
            else:unknown.append(obj['id'])
        # Floating point CityFlight/SBF coordinate conversion is bounded; actual
        # containing target must intersect this approximate language region.
        assert target['id'] in positives
    return dict(instruction=f'Return to the {location} approximately {rounded} meters to the {direction} of {anchor["name"]}.',
        public_anchor={k:anchor['item'].get(k) for k in ('name','object_type','contour')},
        label_positive_object_ids=positives,label_negative_object_ids=negatives,label_unknown_object_ids=unknown,
        relation_direction=direction,rounded_distance_m=rounded,actual_axis_distance_m=float(distance),transverse_distance_m=float(transverse),distance_rounding_error_m=float(abs(rounded-distance)),
        annotation_confidence='verified_launch_point_anchor_geometry_not_human_return_language',
        anchor_name_confidence='original_human_map_name_not_verified_visual_attributes',
        visual_pair_eligible=target is not None)

class UnionFind:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x,x)
        while x != self.parent[x]: self.parent[x] = self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)

def internal_partitions(records):
    """Connected-component isolation of family, hidden object and traversed region.

    All source trajectory cells, not only endpoints, are grouped. Large connected
    maps can leave few independent dev components; report rather than relax.
    """
    uf=UnionFind(); nodes={}
    for r in records:
        base='family:'+r['family_id']; ns=[base, f'object:{r["map_name"]}:{r["target_id"]}']
        if r['start_object_id'] is not None: ns.append(f'object:{r["map_name"]}:{r["start_object_id"]}')
        for x,y in r['trajectory'][:,:2]: ns.append(f'region:{r["map_name"]}:{math.floor(x/200)}:{math.floor(y/200)}')
        for n in ns: uf.union(base,n)
        nodes[r['family_id']]=set(ns)
    components=defaultdict(list)
    for r in records: components[uf.find('family:'+r['family_id'])].append(r['family_id'])
    # Deterministic component hash; no model/Seen/Unseen scores influence split.
    parts={f:('internal_dev' if int(hashlib.sha256(c.encode()).hexdigest(),16)%5==0 else 'train') for c,fs in components.items() for f in fs}
    sets=defaultdict(set)
    for f,ns in nodes.items(): sets[parts[f]].update(ns)
    overlap=sets['train'] & sets['internal_dev']
    assert not overlap
    return parts,dict(components=len(components),largest_component_families=max(map(len,components.values()),default=0),families=dict(Counter(parts.values())),node_overlap=len(overlap),family_object_region_disjoint=True,region_m=200)

def load_observations(path):
    index=defaultdict(list)
    if Path(path).exists():
        with Path(path).open() as handle:
            for line in handle:
                r=json.loads(line)
                for image in r['rgb_observations']: index[int(r['source_episode_index'])].append(image)
    return index

def build(args):
    started=time.monotonic();out=Path(args.output)
    if out.exists(): raise ValueError('refusing new data directory overwrite')
    out.mkdir(parents=True)
    sys.path.insert(0,args.sbf_source)
    from sbfnav.mapdata import GROUND_LEVEL
    source_root=Path(args.data_root); objects_path=source_root/'cityrefer/objects.json'
    geometry=catalog_geometry(json.loads(objects_path.read_text()))
    stats={}; hashes={'objects':sha256(objects_path)}
    for split in args.splits:
        src=source_root/f'processed_citynav/citynav_{split}.json';raw=json.loads(src.read_text());hashes[split]=sha256(src)
        observations=load_observations(Path(args.observations)/f'observations_{split}.jsonl')
        records=[]; count=Counter()
        for i,r in enumerate(raw):
            map_name=f'{r["area"]}_block_{r["block"]}';trajectory=np.asarray([pose4(p) for p in r['trajectory']])
            if len(trajectory)<2: count['too_short']+=1;continue
            target_id=str(r['object_ids'][0]);start,matches=ground_start(trajectory[0,:2],geometry[map_name]);desc=geometry_description(start,geometry[map_name],trajectory[0,:2])
            family=f'{split}:{map_name}:{target_id}:{int(r["ann_ids"][0])}:{i}'
            records.append(dict(raw=r,index=i,map_name=map_name,target_id=target_id,trajectory=trajectory,start_object_id=start['id'] if start else None,start_matches=matches,reverse_description=desc,family_id=family))
            count['source_episodes']+=1;count['start_unique_geometry']+=start is not None;count['start_ambiguous']+=len(matches)>1;count['reverse_static_language_eligible']+=desc is not None;count['reverse_object_language_eligible']+=desc is not None and start is not None
        parts,partition_stats=internal_partitions(records) if split=='train_seen' else ({r['family_id']:split for r in records},{'original_split_retained':True})
        splitout=out/split;splitout.mkdir(); episodes_path=splitout/'episodes.jsonl';prefix_path=splitout/'prefixes.jsonl';fair_path=splitout/'matched_e0_e1.jsonl'
        with episodes_path.open('w') as eh,prefix_path.open('w') as ph,fair_path.open('w') as fh:
            for rec in records:
                rawrec=rec['raw'];n=len(rec['trajectory']);family=rec['family_id'];part=parts[family];end=rec['trajectory'][-1,:3];launch=rec['trajectory'][0,:3]
                for role in ('forward','reverse'):
                    traj=rec['trajectory'] if role=='forward' else reverse_trajectory(rawrec['trajectory'])
                    goal=np.asarray(rawrec['target_positions'][-1],float) if role=='forward' else launch
                    # Actual return endpoint is launch XYZ; forward target altitude is
                    # surface annotation, not interpolated human flight altitude.
                    instruction=rawrec['descriptions'][0] if role=='forward' else (rec['reverse_description']['instruction'] if rec['reverse_description'] else 'Return to the launch location.')
                    eid=family+':'+role
                    ep=dict(episode_id=eid,family_id=family,split=split,internal_partition=part,map_name=rec['map_name'],source_episode_index=rec['index'],direction=role,instruction=instruction,trajectory_xyzyaw=traj.tolist(),private_labels={'target_world':goal.tolist(),'target_object_id':rec['target_id'] if role=='forward' else rec['start_object_id'],'original_goal_world':np.asarray(rawrec['target_positions'][-1]).tolist(),'start_candidate_object_ids':rec['start_matches']},static_language_eligible=role=='forward' or rec['reverse_description'] is not None,reverse_geometry_description=rec['reverse_description'] if role=='reverse' else None,trajectory_source='measured_CityFlight_forward_xyz' if role=='forward' else 'offline_reversal_of_measured_CityFlight_xyz_not_recorded_return_flight')
                    eh.write(json.dumps(ep)+'\n');count['episodes_'+role]+=1
                    for slot,cutoff in enumerate(fixed_prefixes(n)):
                        teacher=remaining_waypoints(traj,cutoff);ground=GROUND_LEVEL[rec['map_name']]
                        history=[]
                        for image in observations[rec['index']]:
                            source_step=int(image['step']);direction_step=source_step if role=='forward' else n-1-source_step
                            if direction_step>cutoff: continue
                            history.append(dict(image,source_step=source_step,direction_step=direction_step,camera_pose_world=rec['trajectory'][source_step].tolist(),camera_orientation='original_recorded_forward_camera_yaw',reuse_kind='current_or_past_forward_observation' if role=='forward' else 'same_visited_pose_camera_requires_yaw_turn_not_reverse_current_image'))
                        history.sort(key=lambda x:x['direction_step']);assert all(x['direction_step']<=cutoff for x in history)
                        row=dict(sample_id=f'{eid}:{cutoff}',episode_id=eid,family_id=family,split=split,internal_partition=part,map_name=rec['map_name'],source_episode_index=rec['index'],direction=role,slot=slot,cutoff_step=cutoff,source_step=cutoff if role=='forward' else n-1-cutoff,instruction=instruction,pose_world=traj[cutoff].tolist(),ground_level=ground,historical_camera_observations=history if role=='forward' else [],reverse_camera_reuse_candidates=history if role=='reverse' else [],current_observation_requires_render=role=='reverse' or not any(x['direction_step']==cutoff for x in history),static_language_eligible=ep['static_language_eligible'],labels={'target_world':goal.tolist(),'teacher_world':teacher.tolist(),'valid_teacher_z':((teacher[:,2]>=ground+10)&(teacher[:,2]<=ground+100)).tolist(),'arrival_stop':bool(np.linalg.norm(traj[cutoff,:2]-goal[:2])<=20),'stop_radius_m':20},attribute_label_status='none_synthetic_colors_forbidden')
                        ph.write(json.dumps(row)+'\n');count['prefixes_'+role]+=1;count['stop_positive_'+role]+=row['labels']['arrival_stop'];count['historical_observations_'+role]+=len(history)
                        # Same family/prefix train budget: E0 repeats a forward row for
                        # each E1 forward/reverse row. Trainer chooses equal steps.
                        forward_id=f'{family}:forward:{cutoff}'
                        fh.write(json.dumps(dict(budget_id=row['sample_id'],family_id=family,internal_partition=part,E0_forward_sample_id=forward_id,E1_sample_id=row['sample_id'],role=role))+'\n')
        count['independent_target_objects']=len({(r['map_name'],r['target_id']) for r in records});count['maps']=len({r['map_name'] for r in records})
        stats[split]=dict(count,partition=partition_stats)
        print(json.dumps(dict(split=split,stats=stats[split],seconds=time.monotonic()-started)),flush=True)
        del raw,records,observations
    manifest=dict(complete=True,source_sha256=hashes,stats=stats,seconds=time.monotonic()-started,protocol={'original_map_splits':True,'reversed_xyz_exact':True,'reverse_heading':'next_nonstationary_travel_tangent_then_last_incoming_at_endpoint','stop':'distance_xy_to_direction_goal<=20m_recomputed','prefixes':'fixed 0,.25,.5,.75,1 without target-driven selection','return_language':'named-anchor cardinal geometry for reliable start surface or launch point; absent named-anchor relationship is trajectory-only','reverse_rgb':'original source camera frames explicitly indexed by reverse time; not asserted identical to recomputed-yaw current images','new_siglip_encoding':False,'attribute_supervision':'no invented attributes; prior labels language-weak only','sampling':'matched E0 forward repetitions vs E1 forward/reverse, equal per-family fixedprefix budgets','unseen_selection':False})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'complete':True,'seconds':manifest['seconds']}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default=DEFAULT_DATA);p.add_argument('--sbf-source',default=DEFAULT_SBF);p.add_argument('--observations',default='runs/visual_grounding_benchmark/data_audit/full_observations');p.add_argument('--output',default='runs/bidirectional_core_test/data');p.add_argument('--splits',nargs='+',choices=['train_seen','val_seen','val_unseen'],default=['train_seen','val_seen']);build(p.parse_args())
if __name__=='__main__': main()
