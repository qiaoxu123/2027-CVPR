"""TRAIN-only Anchor/direction records; real destination labels stay separate."""
import argparse,collections,gzip,json,sys
from pathlib import Path
import numpy as np,shapely,torch
from graphnav.checkpointing import file_sha256
from graphnav.geometry import grid_centers
from graphnav.local_landmark_anchor import named_public_polygons
from graphnav.landmark_navigation_diagnostic import public_region
from scripts.analyze_instruction_landmark_association import iter_train
WORDS=['north','south','east','west'];INVERSE={'north':'south','south':'north','east':'west','west':'east'}
EXACT={w+' of':w for w in WORDS}|{w+' side of':w for w in WORDS}
BAD={'relation_scope_unresolved','target_role_unresolved','target_category_ambiguous_or_unknown','earlier_target_category_noun_scope_requires_review','target_lexical_fallback_scope_uncertain'}


def main():
 p=argparse.ArgumentParser()
 for k in ['structured','train','catalog','data-root','sbf-source','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);sys.path.insert(0,str(a.sbf_source))
 from sbfnav.coordinates import MapTransform
 from sbfnav.dataset import _pose4
 with gzip.open(a.structured,'rt') as f:parsed=json.load(f)
 catalog=json.loads(a.catalog.read_text());public=[];labels=[];geometry_cache={};transforms={};rejections=collections.Counter();raw=collections.Counter();unknown_relative=collections.Counter();raw_examples=[]
 labelraw={i:dict(target_world=row['target_positions'][-1],object_ids=row['object_ids'],first_pose=_pose4(row['trajectory'][0]).tolist(),text=row['descriptions'][0],map_name=f"{row['area']}_block_{row['block']}") for i,row in iter_train(a.train)}
 for r in parsed:
  assert r['target_GT_used'] is False
  edges={e['id']:e for e in r['relations']};seen=set()
  for path in r['direct_relations']:
   if len(path['relations'])!=1:continue
   link=path['relations'][0];edge=edges[link['relation_id']];pred=edge['predicate'];frame=edge['reference_frame']['frame'];raw[pred]+=1
   if frame=='reference_frame_unknown':
    for word in ['front','behind','left','right']:
     if word in pred.split():unknown_relative[word]+=1
   if pred not in EXACT:continue
   if set(r['ambiguity'])&BAD:rejections['target_or_scope_uncertain']+=1;continue
   if edge['ambiguity'] or frame!='world_cardinal':rejections['frame_uncertain']+=1;continue
   if path['entities'][0]!=r['target']['entity'] or len(path['landmark_names'])!=1:rejections['role_or_name_ambiguity']+=1;continue
   word=EXACT[pred]
   if link['traversal']=='reverse':word=INVERSE[word]
   name=path['landmark_names'][0];key=r['map_name'],name,word
   if key in seen:continue
   seen.add(key);mp=r['map_name'];observed=labelraw[r['episode']];assert observed['text']==r['original_text'] and observed['map_name']==mp
   if mp not in transforms:
    t=MapTransform.from_raster(a.data_root/'rgbd'/f'{mp}.tif');transforms[mp]=[t.x_min,t.y_min,t.x_max,t.y_max]
   bounds=np.asarray(transforms[mp]);span=bounds[2:]-bounds[:2];gkey=mp,name
   if gkey not in geometry_cache:
    match=named_public_polygons(catalog[mp],name)
    try:
     region=public_region(match['contours'],bounds);uv=grid_centers(28).numpy();world=bounds[:2]+uv*span;dx,dy=span/28
     cells=shapely.box(world[:,0]-dx/2,world[:,1]-dy/2,world[:,0]+dx/2,world[:,1]+dy/2);area=shapely.area(shapely.intersection(cells,region));prob=(area/area.sum()).astype(np.float32).reshape(28,28)
     geometry_cache[gkey]=(match,prob)
    except ValueError:geometry_cache[gkey]=None
   if geometry_cache[gkey] is None:rejections['invalid_or_outside_public_anchor']+=1;continue
   match,prob=geometry_cache[gkey];index=len(public);public.append(dict(index=index,episode=r['episode'],map_name=mp,subset='fit' if mp.startswith('cambridge_') else 'dev',word=word,direction_id=WORDS.index(word),landmark_name=name,public_object_ids=match['ids'],public_polygons=match['contours'],bounds=bounds.tolist(),span=span.tolist(),anchor=prob.tolist(),start_pose=observed['first_pose'],instruction=r['original_text'],source_relation=edge,parser_ambiguity=r['ambiguity'],supervision='parser_proposal_not_human_verified',actual_visibility='unknown'))
   objects=catalog[mp];ids=[str(x) for x in observed['object_ids']];typ=objects.get(ids[0],{}).get('object_type','Unknown');labels.append(dict(index=index,episode=r['episode'],target_world=observed['target_world'],target_type=typ,target_ids=ids))
 assert public
 fit={r['map_name'] for r in public if r['subset']=='fit'};dev={r['map_name'] for r in public if r['subset']=='dev'};assert not fit&dev
 # city-qualified names are conservative identities across blocks.
 groups={s:{(r['map_name'].split('_block_')[0],r['landmark_name'].casefold()) for r in public if r['subset']==s} for s in ['fit','dev']};assert not groups['fit']&groups['dev']
 subsets={s:{r['index'] for r in public if r['subset']==s} for s in ['fit','dev']}
 for name,value in [('public.json',public),('private_labels.json',labels),('private_fit_labels.json',[r for r in labels if r['index'] in subsets['fit']]),('private_dev_labels.json',[r for r in labels if r['index'] in subsets['dev']])]: (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
 proof=dict(train_instructions=len(parsed),pairs=len(public),unique_episodes=len({r['episode'] for r in public}),counts={s:dict(collections.Counter(r['word'] for r in public if r['subset']==s)) for s in ['fit','dev']},raw_direct_predicates=dict(raw),relative_frame_unresolved_edges=dict(unknown_relative),rejections=dict(rejections),split='Cambridge fit / Birmingham dev; no city/map/anchor identity overlap',gt_in_public=False,inference_inputs='Anchor Belief, direction ID, public map span; original text retained provenance only',source_sha256={str(p):file_sha256(p) for p in [a.structured,a.train,a.catalog,Path(__file__)]},public_sha256=file_sha256(a.output/'public.json'),private_labels_sha256=file_sha256(a.output/'private_labels.json'))
 (a.output/'manifest.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
