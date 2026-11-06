"""Public-only features, with private supervision written to separate split files."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from graphnav.token_language import FrozenBERTTokens
from graphnav.relation_bearing_scene import RelationSceneBuilder
from graphnav.polygon_neighborhood_bearing import PublicPolygonNeighborhood
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256


def main():
 p=argparse.ArgumentParser()
 for k in ['run','data-manifest','public-objects','bert-cache']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();started=time.perf_counter();torch.set_num_threads(4);r=a.run
 data=json.loads(a.data_manifest.read_text());records=data['records'];split=json.loads((r/'data/split.json').read_text());parsed=[json.loads(x) for x in (r/'data/parsed.jsonl').read_text().splitlines()];assert len(parsed)==len(records)==21878
 catalog=json.loads(a.public_objects.read_text());builder=RelationSceneBuilder(catalog);old=PublicPolygonNeighborhood(catalog)
 bert=FrozenBERTTokens('bert-base-uncased',device='cuda',max_length=512,cache_limit=200000);bert.load_cache(a.bert_cache)
 scene_indices={};scenes=[];queries=[];phrases=[];phrase_lookup={}
 def phrase(text):
  if text not in phrase_lookup:phrase_lookup[text]=len(phrases);phrases.append(text)
  return phrase_lookup[text]
 for i,(row,language) in enumerate(zip(records,parsed)):
  refs=row['landmarks'];scene=builder.scene(row['map_name'],refs);key=(row['map_name'],tuple(sorted(x['id'] for x in refs)))
  if key not in scene_indices:
   scene_indices[key]=len(scenes)
   scenes.append(dict(shape=torch.from_numpy(scene['shape']).half(),edges=torch.from_numpy(scene['edges']).half(),types=torch.from_numpy(scene['types']),valid=torch.from_numpy(scene['valid']),name_ids=torch.tensor([phrase(x) for x in scene['node_names']]),node_ids=scene['node_ids'],map_name=row['map_name']))
  refids=[str(x['id']) for x in refs];anchor_idx=np.zeros(16,np.int64);anchor_valid=np.zeros(16,bool)
  steps_entity=np.zeros((16,4),np.int64);steps_rel=np.zeros((16,4),np.int64);step_valid=np.zeros((16,4),bool);kernel=np.full((16,64,8),1/8,np.float32)
  paths=language['paths'];fallback=language['target_span']['text'] if language['target_span'] else row['instruction']
  for j,ref in enumerate(refs):
   if str(ref['id']) not in scene['node_ids']:continue
   anchor_idx[j]=scene['node_ids'].index(str(ref['id']));anchor_valid[j]=True
   options=[p for p in paths if str(ref['id']) in p['public_ids'] and p.get('semantic_confident',False)]
   options.sort(key=lambda p:(p['hops'],str(p['nodes'])))
   if options:
    chosen=options[0]
    for t,k in enumerate(reversed(range(chosen['hops']))):
     st=chosen['steps'][k];entity=chosen['nodes'][k]['text'];relation=f"{st['subject']['text']} {st['relation']} {st['object']['text']}. Starting at {'reference' if st['orientation']==1 else 'subject'} toward {'subject' if st['orientation']==1 else 'reference'}."
     steps_entity[j,t]=phrase(entity);steps_rel[j,t]=phrase(relation);step_valid[j,t]=True
   else:
    steps_entity[j,0]=phrase(fallback);steps_rel[j,0]=phrase('Unspecified spatial relation.');step_valid[j,0]=True
  if refids:kernel[:len(refids)]=builder.direction_kernel(scene,refids)
  prior=old.for_references(row['map_name'],refs)
  queries.append(dict(scene=scene_indices[key],text=row['instruction'],anchor_indices=torch.from_numpy(anchor_idx),anchor_valid=torch.from_numpy(anchor_valid),entity_ids=torch.from_numpy(steps_entity),relation_ids=torch.from_numpy(steps_rel),step_valid=torch.from_numpy(step_valid),angle_kernel=torch.from_numpy(kernel).half(),old_names=torch.tensor([phrase((x.get('object_type','')+' '+x['name']).strip()) for x in refs]+[phrase('')]*(16-len(refs))),old_anchor=torch.from_numpy(prior['anchor_geometry']),old_neighbors=torch.from_numpy(prior['neighbor_geometry']),old_types=torch.from_numpy(prior['neighbor_types']),old_neighbor_valid=torch.from_numpy(prior['neighbor_valid']),category=language['category'],has_direct=language['has_direct'],has_indirect=language['has_indirect'],index=i))
  if (i+1)%1000==0:print(json.dumps(dict(features=i+1,scenes=len(scenes),phrases=len(phrases),seconds=time.perf_counter()-started)),flush=True)
 print('Encoding phrases',len(phrases),flush=True);vectors=[]
 with torch.no_grad():
  for j in range(0,len(phrases),64):
   text=phrases[j:j+64];vec=bert.encode(text).cpu();vectors.append(vec)
   if j%4096==0:print('encoded',j,flush=True)
 vectors=torch.cat(vectors)
 # Keep only complete original instruction tokens for all later training arms.
 texts=set(row['instruction'] for row in records)
 bert._load(sorted(texts));bert.cache={k:v for k,v in bert.cache.items() if k in texts};bert.save_cache(r/'instruction_tokens.pt')
 atomic_save(dict(scenes=scenes,queries=queries,phrase_vectors=vectors,phrases=phrases),r/'public_inputs.pt')
 # PRIVATE labels below; no target fields were read by feature selection.
 for group,ids in split['indices'].items():
  labels={}
  for i in ids:
   row=records[i];q=queries[i];scene=scenes[q['scene']];maprows=builder.map_index(row['map_name']);bounds=np.asarray(row['world_bounds']);gt=bounds[:2]+np.asarray(row['target_xy'])*(bounds[2:]-bounds[:2]);y=np.full(16,-100,np.int64);theta=np.zeros(16,np.float32)
   for j,ref in enumerate(row['landmarks']):
    if not q['anchor_valid'][j]:continue
    d=gt-maprows[str(ref['id'])]['center']
    if np.linalg.norm(d)<=1e-9:continue
    t=math.atan2(d[1],d[0])%(2*math.pi);theta[j]=t;y[j]=int(np.rint(t/(math.pi/4)))%8
   target_id=str(row['episode_id']).split(':')[1]
   target=scene['node_ids'].index(target_id) if target_id in scene['node_ids'] else -1
   labels[i]=dict(bins=torch.from_numpy(y),theta=torch.from_numpy(theta),target_index=target,physical_id=':'.join(str(row['episode_id']).split(':')[:2]))
  atomic_save(labels,r/('PRIVATE_'+group+'_labels.pt'))
 atomic_json(dict(N=len(queries),scenes=len(scenes),phrases=len(phrases),seconds=time.perf_counter()-started,public_inputs_SHA=file_sha256(r/'public_inputs.pt'),frozen_BERT=True,GT_only_in_private_label_files=True,parser_SHA=file_sha256(r/'data/parsed.jsonl'),shape_dim=39,radius_m=200,max_nodes=64),r/'features.json')
 print('Features complete',flush=True)
if __name__=='__main__':main()
