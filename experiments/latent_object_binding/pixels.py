"""Bounded public-policy causal HSV/Lab extraction from actual CityNav crops."""
import collections,json,time
import cv2,numpy as np,torch
from pathlib import Path
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import project_landmark
from experiments.semantic_polygon_grounding.audit import DATA,save
from .resources import OUT,primary_features,regions
from .language import parse
from experiments.language_object_graph.objects import ObjectGraph
COLORS=('white','gray','black','red','blue','green','yellow','brown','orange','beige','silver')
BLACK={c:float(c=='black') for c in COLORS}


def color_scores(pixels):
    hsv=cv2.cvtColor(np.asarray(pixels,np.uint8).reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3);h,s,v=hsv.T.astype(float)
    masks={'white':(s<55)&(v>=180),'gray':(s<65)&(v>=65)&(v<180),'black':v<65,'red':((h<12)|(h>168))&(s>=65)&(v>=65),'blue':(h>=95)&(h<=130)&(s>=65)&(v>=65),'green':(h>=35)&(h<=90)&(s>=65)&(v>=65),'yellow':(h>=20)&(h<=35)&(s>=65)&(v>=90),'brown':(h>=5)&(h<=25)&(s>=65)&(v>=50)&(v<180),'orange':(h>=5)&(h<=20)&(s>=65)&(v>=180),'beige':(h>=10)&(h<=35)&(s<100)&(v>=140),'silver':(s<65)&(v>=100)}
    return {c:float(a.mean()) for c,a in masks.items()}


def main():
    start=time.time();torch.set_num_threads(1);cv2.setNumThreads(1);dest=OUT/'pixels';dest.mkdir(exist_ok=True);graphs={};bank={};audit=[];stats={};renderer=CityNavObservedViewRenderer(DATA/'rgbd',dest/'environment_private',image_size=384)
    for split in ('train','dev'):
        features,states,manifest=primary_features(split);groups={};meta={}
        for state in states:
            m=state['metadata'];name=m['map_name']
            if name not in graphs:graphs[name]=ObjectGraph.restore(torch.load(f'runs/language_object_graph/graphs/{name}.pt',weights_only=True))
            language=parse(m['instruction'],graphs[name]);types={e['object_type'] for e in language['variables'] if e['object_type']};framegroups=collections.defaultdict(list)
            for r in regions(state).values():
                if r['object_type'] not in types:continue
                key=(name,r['episode_id'],r['source_step']);framegroups[key].append(r)
            # At most two visited frames per family, selected solely for public
            # relevance/quality; no GT visibility or private target identity.
            order=sorted(framegroups,key=lambda k:(-sum(min(128,r['native_min_side']) for r in framegroups[k]),k))[:2]
            for key in order:
                if key not in groups:groups[key]={};meta[key]=m
                for r in framegroups[key]:groups[key][r['view_key']]=r
        selected=sorted(groups,key=lambda k:(-len(groups[k]),k))[:128]
        failures=collections.Counter();before=len(bank)
        for j,key in enumerate(sorted(selected)):
            regs=list(groups[key].values());ref=regs[0];obs=renderer.current_rgbd(key[0],ref['actual_camera_pose'],ref['ground_level']);g=graphs[key[0]]
            for r in regs:
                obj=g.objects[g.byid[r['object_id']]];roi=project_landmark(dict(obj,id=r['object_id']),obs['footprint_world'],obs['bgr'].shape,min_visible=.9,min_pixels=256)
                if roi is None:failures['projection_failed']+=1;continue
                if tuple(roi.bbox_xyxy)!=tuple(r['bbox']):failures['cached_bbox_mismatch']+=1;continue
                mask=roi.mask.astype(bool)&obs['depth_valid'];pixel=obs['bgr'][mask]
                if len(pixel)<256:failures['insufficient_depth_pixels']+=1;continue
                x,y,u,v=roi.bbox_xyxy;lab=cv2.cvtColor(pixel.reshape(-1,1,3),cv2.COLOR_BGR2LAB).reshape(-1,3).mean(0)
                record=dict(color_scores=color_scores(pixel),mean_lab=lab.tolist(),pixel_count=len(pixel),native_min_side=r['native_min_side'],depth_support=r['depth_support'],foreground_fraction=float(mask[y:v,x:u].mean()),source_step=r['source_step'],episode_id=r['episode_id'],map_name=key[0],camera_pose=r['actual_camera_pose'],view_key=r['view_key'],object_id=r['object_id'],status='geometric weak foreground, no human attribute label')
                bank[r['view_key']]=record;audit.append(dict(record,split=split))
            if j<4:
                cv2.imwrite(str(dest/f'{split}_view_{j:02d}.png'),obs['bgr']);np.savez_compressed(dest/f'{split}_view_{j:02d}_geometry.npz',depth=obs['depth_m'],valid=obs['depth_valid'],footprint=obs['footprint_world'],pose=obs['pose'])
            if j%32==0:print('pixels',split,j,len(selected),round(time.time()-start,1),flush=True)
        stats[split]=dict(frames=len(selected),selected_public_frames=len(groups),valid_rois=len(bank)-before,failures=dict(failures),families=len(states))
        del features
    torch.save(bank,dest/'bank.pt');save(dest/'manifest.json',dict(stats=stats,seconds=time.time()-start,black_reference=BLACK,source='actual visited pose orthographic CityNav crop, vertical depth proxy; no perspective occlusion claim',attribute_oracle_n=0,selection='no GT; public variable type + native pixel quality, <=2frames/family then fixed128frames/split',audit=audit));print('PIXELS DONE',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
