"""Optional frozen PGVB Anchor evidence from the original current TRAIN start only."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
from graphnav.checkpointing import file_sha256
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
from graphnav.pgvb_navigation_runtime import PGVBNamedLandmarkPolicy
from graphnav.landmark_region_supervision import public_type
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_visual_only import visual_only_world_belief


def main():
 p=argparse.ArgumentParser()
 for k in ['public','pgvb','catalog','data-root','sbf-source','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
 torch.set_num_threads(4);rows=[r for r in json.loads(a.public.read_text()) if r['subset']=='dev'];objects=json.loads(a.catalog.read_text());checkpoint_sha=file_sha256(a.pgvb);assert checkpoint_sha=='ba3073723703ce8188c7f74ae810a622342631bb150bd2597656ff0039f2cf56'
 # Reuse the existing native pooling/PGVB forward without loading or invoking
 # the navigator. Its scorer accepts only the four public attributes below.
 class FrozenAnchor:
  pass
 wrapper=FrozenAnchor();wrapper.device=torch.device('cuda');wrapper.native=ObservedSigLIPMatchingEncoder('google/siglip2-large-patch16-384',device='cuda');wrapper.pgvb=PolygonGuidedVisualBelief().to('cuda').eval().requires_grad_(False);wrapper.pgvb.load_state_dict(torch.load(a.pgvb,map_location='cpu',weights_only=True)['model'],strict=True)
 renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'observations',image_size=384);result=[];probs={};started=time.perf_counter()
 for r in rows:
  mp=r['map_name'];byid={str(k):v for k,v in objects[mp].items()};typ=public_type([byid[x]['object_type'] for x in r['public_object_ids']]);height=r['start_pose'][2]-GROUND_LEVEL[mp];reason='unsupported_or_invalid_height';available=False
  if typ in ['Building','Road'] and height>0:
   wrapper._landmark=dict(name=r['landmark_name'],typ=typ,contours=r['public_polygons']);wrapper.record=dict(world_bounds=r['bounds']);obs=renderer.current_rgbd(mp,r['start_pose'],GROUND_LEVEL[mp]);evidence,reason=PGVBNamedLandmarkPolicy._observed_scores(wrapper,obs)
   if evidence is not None:
    uv,scores,valid=evidence;v=visual_only_world_belief(local_uv=uv,scores=scores,valid=valid);available=bool(v['has_view'][0])
    if available:probs[str(r['index'])]=v['probabilities'][0].detach().cpu().numpy().reshape(28,28)
  result.append(dict(index=r['index'],available=available,reason=reason))
 np.savez_compressed(a.output/'anchors.npz',**probs);(a.output/'availability.json').write_text(json.dumps(result,indent=2));assert file_sha256(a.pgvb)==checkpoint_sha
 (a.output/'manifest.json').write_text(json.dumps(dict(public_sha256=file_sha256(a.public),pgvb_sha256=checkpoint_sha,available=len(probs),attempted=len(rows),seconds=time.perf_counter()-started,normal_inputs='Original current pose RGB only; requested public name/type/Polygon; no target GT, future image or target appearance',producer_sha256=file_sha256(__file__)),indent=2));print('PGVB current Anchor available',len(probs),'/',len(rows))
if __name__=='__main__':main()
