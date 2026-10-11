"""Small Train-only RGB engineering checks and offline candidate ceiling."""
import gzip,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from .run import OUT,load,aggregate,dump_rows
from .objects import ObjectGraph
from .rgb import CausalAttributes
from experiments.semantic_polygon_grounding.audit import DATA,save,goal_metrics
from graphnav.renderer import CityNavObservedViewRenderer

def main():
    start=time.perf_counter();states,labels,_=load('train');graphs={}
    for name in {s['metadata']['map_name'] for s in states}:graphs[name]=ObjectGraph.restore(torch.load(OUT/'graphs'/f'{name}.pt',weights_only=True))
    # Selection solely sorted public family IDs; not target visibility or truth.
    by_map={}
    for s in states:
        if s['regions']:by_map.setdefault(s['metadata']['map_name'],s)
    selected=[by_map[k] for k in sorted(by_map)[:8]];renderer=CityNavObservedViewRenderer(DATA/'rgbd',OUT/'rgb_qa',image_size=384)
    observations=[];(OUT/'rgb_qa').mkdir(exist_ok=True)
    for j,s in enumerate(selected):
        regions=sorted(s['regions'],key=lambda r:(r['source_step'],r['view_key']));region=regions[-1]
        assert region['source_step']<=s['metadata']['cutoff_step'];g=graphs[s['metadata']['map_name']]
        obs=renderer.current_rgbd(s['metadata']['map_name'],region['actual_camera_pose'],region['ground_level'])
        cv2.imwrite(str(OUT/'rgb_qa'/f'{j:02d}_rgb.png'),obs['bgr']);cv2.imwrite(str(OUT/'rgb_qa'/f'{j:02d}_valid.png'),obs['depth_valid'].astype(np.uint8)*255)
        attrs={}
        for control in ('real','black','shuffled'):
            memory=CausalAttributes();memory.observe(obs,g,region['source_step'],s['metadata']['cutoff_step'],control=control)
            attrs[control]={str(i):v for i,v in memory.nodes.items()}
        assert set(attrs['real'])==set(attrs['black'])==set(attrs['shuffled'])
        image=obs['bgr'].copy()
        from graphnav.landmark_projection import project_landmark
        for i in attrs['real']:
            roi=project_landmark(g.objects[int(i)],obs['footprint_world'],image.shape,min_visible=.9,min_pixels=256)
            cv2.polylines(image,[np.rint(roi.contour_px).astype(np.int32)],True,(0,255,0),1)
            cv2.putText(image,g.ids[int(i)],roi.bbox_xyxy[:2],cv2.FONT_HERSHEY_SIMPLEX,.3,(0,255,255),1)
        cv2.imwrite(str(OUT/'rgb_qa'/f'{j:02d}_polygons.png'),image)
        observations.append(dict(metadata=s['metadata'],source_step=region['source_step'],camera_pose=region['actual_camera_pose'],footprint_world=obs['footprint_world'].tolist(),
            map=g.objects[0].get('map_name'),valid_fraction=float(obs['depth_valid'].mean()),observed_objects=len(attrs['real']),attributes=attrs,
            status='Train-only pipeline QA, not A4 performance or human attribute labels'))
    save(OUT/'rgb_qa'/'observations.json',dict(observations=observations,seconds=time.perf_counter()-start,
       source='Actual causal cached camera poses rendered by existing CityNav nadir simulator; vertical depth proxy, not perspective depth. Raw simulator map never policy input.'))
    # Identity-selected public polygon alone cannot recover a continuous target
    # outside that polygon; exact GT nearest grid is a separate diagnostic.
    ceilings={}
    for split in ('train','dev','seen'):
        states,labels,_=load(split);result=[]
        with gzip.open(OUT/f'{split}_IdentityOracle_w0.5.jsonl.gz','rt') as f:oracle=[json.loads(line) for line in f]
        for s,o in zip(states,oracle):
            gname=s['metadata']['map_name']
            if gname not in graphs:graphs[gname]=ObjectGraph.restore(torch.load(OUT/'graphs'/f'{gname}.pt',weights_only=True))
            g=graphs[gname];p=s['public'];world=np.asarray(p['bounds'][:2])+np.asarray(p['grid_xy'])*(np.asarray(p['bounds'][2:])-p['bounds'][:2]);label=labels[s['metadata']['sample_id']]
            target=g.byid.get(str(label['target_object_id']));grid_distance=np.linalg.norm(world-np.asarray(label['target_world']),axis=1)
            from .reasoning import object_kernels
            kernel=object_kernels(g,world)[target] if target is not None else np.ones(784)/784
            m,order,rank=goal_metrics(np.log(np.maximum(kernel,1e-30)),p['grid_xy'],p['bounds'],label['target_world'])
            base,_,_=goal_metrics(p['static_logits'],p['grid_xy'],p['bounds'],label['target_world']);m.update(recovery=float((1-base['goal_r1'])*m['goal_r1']),damage=float(base['goal_r1']*(1-m['goal_r1'])),net_recovery=float(m['goal_r1']-base['goal_r1']))
            result.append(dict(sample_id=s['metadata']['sample_id'],family_id=s['metadata']['family_id'],map_name=gname,metrics=m,goal_order=order,goal_target_rank=rank,
                target_public_node_available=target is not None,gt_nearest_grid_hit=bool(grid_distance.min()<=20),identity_public_polygon_valid=bool(target is not None and g.valid[target])))
        dump_rows(OUT/f'{split}_pure_identity_oracle.jsonl.gz',result)
        ceilings[split]=dict(metrics=aggregate(result),target_nodes=sum(x['target_public_node_available'] for x in result),valid_target_polygons=sum(x['identity_public_polygon_valid'] for x in result),gt_nearest_grid_hits=sum(x['gt_nearest_grid_hit'] for x in result),n=len(result))
    save(OUT/'oracle_ceilings.json',ceilings)

if __name__=='__main__':main()
