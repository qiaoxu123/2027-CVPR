"""Aggregate complete TRAIN paired diagnostics and exact footprint geometry posthoc."""
import argparse,json,sys,csv
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,Point,box
from graphnav.memory import ObservedRGBMemory
from scripts.unified_pgvb_v3_pilot import blind_public_landmarks,fixed_yaw_public_poses

def mean(values):return float(np.mean(values)) if len(values) else None
def metrics(rows,variant,method='scale1'):
    rr=[r['variants'][variant] for r in rows]
    return dict(n=len(rr),R20=mean([r['counterfactual'][method]['hit'] for r in rr]),NE=mean([r['counterfactual'][method]['NE'] for r in rr]),
                fixes=sum(not r['geometry']['hit'] and r['counterfactual'][method]['hit'] for r in rr),
                breaks=sum(r['geometry']['hit'] and not r['counterfactual'][method]['hit'] for r in rr))

def main():
    p=argparse.ArgumentParser()
    for k in ('experiment','train-annotations','sbf-source','source-v2','objects','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
    manifest=json.loads((a.experiment/'full/manifest.json').read_text());rows=[json.loads(p.read_text()) for p in sorted((a.experiment/'full/rows').glob('*.json'))]
    assert manifest['n']==len(rows)==4405 and len({r['episode_id'] for r in rows})==4405
    ann={r['episode_id']:r for r in map(json.loads,a.train_annotations.read_text().splitlines())}
    public=[dict(map_name=r['map_name'],instruction=r['instruction'],world_bounds=ann[r['episode_id']]['world_bounds'],landmarks=[]) for r in rows]
    # Same public-name cap recorded by the frozen base checkpoint.
    import torch
    config=torch.load(a.source_v2,map_location='cpu',weights_only=True)['config']['geometry_args']
    blind_public_landmarks(public,json.loads(a.objects.read_text()),int(config['max_landmarks']))
    hierarchy={};ungated=[]
    for row,pub in zip(rows,public):
        record=ann[row['episode_id']];bounds=np.asarray(record['world_bounds']);target=bounds[:2]+np.asarray(record['target_xy'])*(bounds[2:]-bounds[:2]);point=Point(target)
        assert record['instruction']==row['instruction'],'TRAIN cached/official text mismatch'
        base=np.asarray(row['start_pose']);zero=base.copy();zero[3]=0.;high=zero.copy();high[2]=GROUND_LEVEL[row['map_name']]+80
        plan=np.asarray(row['planned_pose']);rotated=plan.copy();rotated[3]=base[3]
        planned_poses=fixed_yaw_public_poses(pub,ground=GROUND_LEVEL[row['map_name']],altitude_m=80,yaw=0,offset_m=35,views=5)
        if planned_poses:assert np.allclose(planned_poses[0],plan,atol=.001), 'public planner reconstruction mismatch'
        views={'start_native':[base],'start_yaw0':[zero],'start_h80_yaw0':[high],'plan1_yaw0':[plan],'plan1_nativeyaw':[rotated],'plan5_yaw0':planned_poses}
        # Clip public planned XY exactly as original planner. No target-selected locations.
        for name,poses in views.items():
            polygons=[]
            if not (name.startswith('plan') and row['anchor_count']==0):
                for pose in poses:
                    pose=pose.copy()
                    if name.startswith('plan'):pose[:2]=np.clip(pose[:2],bounds[:2]+.1,bounds[2:]-.1)
                    polygons.append(Polygon(ObservedRGBMemory.view_corners(pose[:2],pose[3],pose[2]-GROUND_LEVEL[row['map_name']])))
            s=row['variants'][name]['support'];s['exact_target_in_footprint']=any(p.covers(point) for p in polygons)
            s['exact_20m_region_intersects_footprint']=any(p.intersection(box(bounds[0],bounds[1],bounds[2],bounds[3])).distance(point)<=20 for p in polygons)
            if not s['effective_region_support']:
                reason='outside_20m_footprint' if not s['exact_20m_region_intersects_footprint'] else 'depth_patch_invalid' if not s['raw_depth_region_support'] else 'landmark_gate' if row['anchor_count']==0 else 'unknown'
                hierarchy.setdefault(name,{});hierarchy[name][reason]=hierarchy[name].get(reason,0)+1
        if 'ungated' in row['variants']['start_native']:ungated.append(row['variants']['start_native'])
    summary=json.loads((a.experiment/'full/summary.json').read_text())
    result=dict(manifest=manifest,variants=summary['variants'],support_failure_hierarchy=hierarchy,paired={},geometry_reference={},ungated={})
    for name in summary['variants']:
        rr=[r['variants'][name] for r in rows]
        result['geometry_reference'][name]=dict(R20=mean([r['geometry']['hit'] for r in rr]),NE=mean([r['geometry']['NE'] for r in rr]))
        result['variants'][name]['exact_footprint_support']={k:sum(r['support'][k] for r in rr) for k in ('exact_target_in_footprint','exact_20m_region_intersects_footprint')}
        for field in ('tanh_saturated_fraction','visual_positive_fraction','wrong_minus_correct_peak'):
            vals=[r['support'][field] for r in rr if r['support'][field] is not None]
            result['variants'][name][field+'_quantiles']=np.quantile(vals,[0,.1,.5,.9,1]).tolist() if vals else None
    for left,right in (('start_native','start_yaw0'),('start_yaw0','start_h80_yaw0'),('start_h80_yaw0','plan1_yaw0'),('plan1_yaw0','plan1_nativeyaw'),('plan1_yaw0','plan5_yaw0')):
        common=[r for r in rows if r['variants'][left]['support']['effective_region_support'] and r['variants'][right]['support']['effective_region_support']]
        result['paired'][left+'__'+right]=dict(all={left:metrics(rows,left),right:metrics(rows,right)},common_effective_support={left:metrics(common,left),right:metrics(common,right)})
    for kind in ('score','support'):
        if kind=='score':
            result['ungated']=dict(n=len(ungated),gated_R20=mean([r['fused']['hit'] for r in ungated]),ungated_R20=mean([r['ungated']['score']['hit'] for r in ungated]),
                gated_NE=mean([r['fused']['NE'] for r in ungated]),ungated_NE=mean([r['ungated']['score']['NE'] for r in ungated]),
                fixes=sum(not r['fused']['hit'] and r['ungated']['score']['hit'] for r in ungated),breaks=sum(r['fused']['hit'] and not r['ungated']['score']['hit'] for r in ungated),
                raw_depth_region_support=sum(r['support']['raw_depth_region_support'] for r in ungated),restored_effective_region_support=sum(r['ungated']['support']['effective_region_support'] for r in ungated))
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'paired_summary.json').write_text(json.dumps(result,indent=2))
    with (a.output/'paired_rows.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    table=['|观测条件|N|Geometry R20|融合R20|NE|纠正|破坏|有效区域支持|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in result['variants'].items():
        m=v['counterfactual']['scale1'];table.append(f"|{name}|{v['n']}|{v['geometry_R20']:.4%}|{m['R20']:.4%}|{m['NE']:.4f}|{m['fixes']}|{m['breaks']}|{v['support']['effective_region_support']}|")
    (a.output/'tables.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({k:v['counterfactual']['scale1'] for k,v in result['variants'].items()},indent=2))

if __name__=='__main__':main()
