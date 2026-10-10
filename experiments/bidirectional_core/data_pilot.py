"""Fixed map-balanced pilot and causal endpoint ROI indices; no model inference."""
import argparse,json,hashlib,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np

def balanced(records,limit):
    groups=defaultdict(list)
    for r in records:groups[r['map_name']].append(r)
    for m in groups:groups[m].sort(key=lambda r:hashlib.sha256(r['family_id'].encode()).hexdigest())
    selected=[];offset=0
    while len(selected)<limit:
        added=False
        for m in sorted(groups):
            if offset<len(groups[m]):selected.append(groups[m][offset]);added=True
            if len(selected)==limit:break
        if not added:break
        offset+=1
    return selected

def build(args):
    out=Path(args.data);result={};started=time.monotonic()
    for split in args.splits:
        limit={'train_seen':args.train_families,'val_seen':args.seen_families,'val_unseen':args.unseen_families}[split]
        root=out/split;records=[];lookup={}
        for line in (root/'episodes.jsonl').open():
            r=json.loads(line);lookup[r['episode_id']]=r
            if r['direction']=='forward' and (split!='train_seen' or r['internal_partition']=='train'):records.append(r)
        chosen=balanced(records,limit if limit>0 else len(records));families={r['family_id'] for r in chosen};index={r['source_episode_index']:r for r in chosen}
        with (root/'pilot_episodes.jsonl').open('w') as eh,(root/'pilot_prefixes.jsonl').open('w') as ph,(root/'pilot_matched_e0_e1.jsonl').open('w') as fh,(root/'pilot_frame_plan.jsonl').open('w') as ih:
            for f in sorted(families):
                for role in ('forward','reverse'):eh.write(json.dumps(lookup[f+':'+role])+'\n')
            n=0
            for line in (root/'prefixes.jsonl').open():
                r=json.loads(line)
                if r['family_id'] in families:
                    ph.write(line);n+=1
                    exact=[x for x in r['historical_camera_observations'] if x['direction_step']==r['cutoff_step']]
                    plan=dict(frame_id=r['sample_id'],family_id=r['family_id'],map_name=r['map_name'],direction=r['direction'],split=split,source_episode_index=r['source_episode_index'],source_step=r['source_step'],direction_step=r['cutoff_step'],observation_cutoff_step=r['cutoff_step'],pose_world=r['pose_world'],ground_level=r['ground_level'],existing_exact_current_source=exact[0] if exact else None,requires_render=not bool(exact),image_size=384,provenance='visited trajectory position with direction-specific yaw; source orthophoto remains renderer-private')
                    ih.write(json.dumps(plan)+'\n')
            for line in (root/'matched_e0_e1.jsonl').open():
                r=json.loads(line)
                if r['family_id'] in families:fh.write(line)
        counts=Counter();pairpath=root/'visual_pairs.jsonl'
        # Original RGB crops are genuine observed nadir source frames. Direction
        # timestamps are explicit. They are not current reverse-yaw crop aliases.
        with pairpath.open('w') as handle:
            region_path=Path(args.visual_data)/split/'regions.jsonl'
            for line in region_path.open():
                r=json.loads(line);i=int(r['source_episode_index'])
                if i not in index:continue
                forward=index[i];reverse=lookup[forward['family_id']+':reverse'];nsteps=len(forward['trajectory_xyzyaw']);step=int(r['observed_step']);native=r['roi_pixel_size']
                if min(native)<24:counts['native_lt24_excluded']+=1;continue
                for role,ep in [('forward',forward),('reverse',reverse)]:
                    oid=ep['private_labels']['target_object_id']
                    if oid is None or str(r['object_id'])!=str(oid):continue
                    if role=='reverse' and not ep['static_language_eligible']:continue
                    pose=np.asarray(forward['trajectory_xyzyaw'][step]);goal=np.asarray(ep['private_labels']['target_world']);distance=float(np.linalg.norm(pose[:2]-goal[:2]))
                    if distance>50:continue
                    t=step if role=='forward' else nsteps-1-step
                    desc=ep.get('reverse_geometry_description') or {}
                    item=dict(pair_id=f'{ep["episode_id"]}:{r["sample_id"]}',family_id=ep['family_id'],episode_id=ep['episode_id'],split=split,internal_partition=ep['internal_partition'],map_name=r['map_name'],direction=role,source_episode_index=i,source_step=step,observed_step=t,observation_cutoff_step=t,instruction=ep['instruction'],attribute_phrase=r.get('attribute_phrase'),roi_sample_id=r['sample_id'],roi_path=r.get('roi_path'),roi_source=r['roi_source'],roi_bbox=r['roi_bbox'],roi_pixel_size=native,roi_visibility=r['roi_visibility'],footprint_world=r['footprint_world'],object_id=str(oid),object_type=r['object_type'],endpoint_distance_xy_m=distance,positive_object_ids=desc.get('label_positive_object_ids',[str(oid)]),negative_object_ids=desc.get('label_negative_object_ids',[]),unknown_object_ids=desc.get('label_unknown_object_ids',[]),attributes=r.get('attributes',{}),attribute_label_status='language_weak_not_visual_verified',observation_semantics='actual_forward_observed_source_camera_view' if role=='forward' else 'reverse_visited_pose_source_camera_orientation_requires_camera_turn',requires_reverse_current_render=role=='reverse',target_id_is_private_supervision=True)
                    handle.write(json.dumps(item)+'\n');counts['pairs_'+role]+=1;counts['roof_color_weak']+=('roof_color' in r.get('attributes',{}));counts['vehicle_color_weak']+=('vehicle_color' in r.get('attributes',{}))
        result[split]=dict(pilot_families=len(families),pilot_prefixes=n,maps=dict(Counter(r['map_name'] for r in chosen)),visual=dict(counts),pilot_seed_protocol='sha256 family_id then round robin map sorted; Train partition only; no validation score used')
        print(json.dumps({split:result[split],'seconds':time.monotonic()-started}),flush=True)
    (out/'pilot_manifest.json').write_text(json.dumps(dict(stats=result,complete=True,seconds=time.monotonic()-started,endpoint_near_radius_m=50,minimum_native_roi_px=24,human_verified_attributes=0),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='runs/bidirectional_core_test/data');p.add_argument('--visual-data',default='runs/visual_grounding_benchmark/data_audit/full');p.add_argument('--train-families',type=int,default=2048);p.add_argument('--seen-families',type=int,default=128);p.add_argument('--unseen-families',type=int,default=0,help='0 selects every predetermined Unseen family; only used when explicitly requested');p.add_argument('--splits',nargs='+',choices=['train_seen','val_seen','val_unseen'],default=['train_seen','val_seen']);build(p.parse_args())
