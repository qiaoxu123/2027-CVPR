"""Validate cached-vs-online observed query production and belief inputs on one real frame."""
import argparse,json
from pathlib import Path
import torch,cv2
from graphnav.native_object_runtime import NativeObjectEncoder,prepare_native_frame
from graphnav.checkpointing import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='aerialvg_objects_train256_20261101');p.add_argument('--prepared',default='aerialvg_native_training_train256_20261101');p.add_argument('--output',default='native_runtime_validation_20261101.json');a=p.parse_args()
 root=Path(a.cache);manifest=json.loads((root/'manifest.json').read_text());entry=manifest['frames'][0];cached=torch.load(root/(entry['key']+'.pt'),map_location='cpu',weights_only=True)
 rows=torch.load(Path(a.prepared)/'rows_0000.pt',map_location='cpu',weights_only=True);row=next(r for r in rows if r['episode']==cached['episode_index'] and r['step']==cached['step'])
 observation=dict(bgr=cv2.imread(str(root/(entry['key']+'.png'))),depth_valid=cached['depth_valid'].numpy(),footprint_world=cached['footprint_world'].numpy())
 encoder=NativeObjectEncoder();produced=encoder.encode(observation,cached['instruction'])
 expected=cached['all_query_features'].float();expected[cached['query_ids']]=cached['query_features'].float()
 errors=dict(features_max=float((produced['features'].cpu()-expected).abs().max()),boxes_max=float((produced['boxes'].cpu()-cached['all_boxes_cxcywh']).abs().max()),quality_max=float((produced['quality'].cpu()-cached['all_valid_pixel_fraction']).abs().max()))
 assert errors['features_max']<2e-4 and errors['boxes_max']<1e-5 and errors['quality_max']<1e-5,errors
 prepared=prepare_native_frame(produced,row['grid_world'].cuda(),row['base_logits'].cuda(),observation,sigma=8.)
 assert torch.equal(prepared['valid'].cpu()[0],row['valid'])
 assert torch.equal(prepared['features'].cpu()[0],row['features'].float())
 assert torch.equal(prepared['observed'].cpu()[0],row['observed'])
 errors['geometry_max']=float((prepared['geometry'].cpu()[0]-row['geometry']).abs().max());assert errors['geometry_max']<1e-4
 atomic_json(dict(pass_=True,key=entry['key'],errors=errors,actual_observation=True,labels_not_used=True),Path(a.output));print(json.dumps(errors),flush=True)


if __name__=='__main__':main()
