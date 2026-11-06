"""Reuse v2 exactly, extend text-type binding, separate public inputs and GT evaluation."""
import argparse,collections,copy
import spacy,pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from common import *
from geometry_frames import compute_pair

def run(root,out):
 out.mkdir(exist_ok=True,parents=True);E=readj(root/'original_citynav/citynav_train_seen.json');O=readj(root/'cityrefer_original/objects.json');old=readlines(V2/'results/text_relations.jsonl.gz');oldpairs=readlines(V2/'results/geometry_pairs.jsonl.gz');oldidx={r['occurrence_id']:r for r in old};oldbound={r['occurrence_id']:r for r in oldpairs};maps=sorted({r['map'] for r in old});catalogs={m:TypedRoadCatalog([Entity(str(i),o['name'],o['object_type']) for i,o in O[m].items() if o['name']]) for m in maps}
 nlp=spacy.load('en_core_web_sm',disable=['ner']);texts=sorted({r['instruction'] for r in old});docs={t:d for t,d in zip(texts,nlp.pipe(texts,batch_size=128))};rows=[];expansion=[]
 for i,e in enumerate(E):
  text=e['descriptions'][0]
  if text not in docs:continue
  m=f"{e['area']}_block_{e['block']}";tid=str(e['object_ids'][0]);ann=e['ann_ids'][0];obj=O[m][tid]
  rr,_=parse_landmark_relations(docs[text],obj['object_type'],catalogs[m])
  for new in rr:
   rid=f"{m}:{tid}:{ann}:{new['span_start']}";prev=oldidx.get(rid)
   if prev is None:continue
   if rid in oldbound:
    row=copy.deepcopy(prev);row['origin']='v2_preserved'
   else:
    row={**copy.deepcopy(new),'occurrence_id':rid,'episode':i,'map':m,'target_id':tid,'target_category':obj['object_type'],'ann_id':ann,'instruction':text,'cluster':f'{m}:{tid}','origin':'typed_road_extension','annotation_alignment':prev['annotation_alignment']}
    if row.get('subject_role')=='annotated_target':row['subject_ids']=[tid]
    reason=None
    if row['status']=='bound' and row.get('entity_resolution') not in {'semantic_road_type_unique','semantic_road_type_all_segments'}:reason='extension_requires_independent_typed_road_identity'
    if row['status']=='bound' and row['annotation_alignment']=='unmatched':reason='annotation_text_mismatch'
    if row['status']=='bound' and len(row.get('subject_ids',[]))!=1:reason='subject_not_independently_localized'
    if row['status']=='bound' and set(row.get('subject_ids',[]))&set(row.get('reference_ids',[])):reason='self_reference'
    if reason:row.update(status='excluded',reason=reason,confidence_tier='excluded',parse_confidence=0.)
   row['v2_status']=prev['status'];row['v2_reason']=prev['reason'];rows.append(row)
   if row['status']=='bound' and rid not in oldbound:expansion.append({'occurrence_id':rid,'instruction':text,'word':row['word'],'original_candidates':new.get('reference_candidate_ids',[]),'selected_reference_ids':row['reference_ids'],'resolution':row['entity_resolution'],'parse_confidence':row['parse_confidence'],'rule':'explicit road semantics + TrafficRoad ontology; retain ALL matching road segments; no target geometry'})
 assert len(rows)==len(old) and {r['occurrence_id'] for r in rows}==set(oldidx)
 public=[];evaluation=[];pairs=[]
 for r in rows:
  if r['status']!='bound':continue
  ep=E[r['episode']];obs=O[r['map']];p,tp,rp=compute_pair(r,ep,obs);pairs.append(p)
  forbidden=set(r['subject_ids']);scene={}
  # Scene contains only public NAMED objects or Water, excluding relation subject ID.
  for oid,o in obs.items():
   if oid not in forbidden and (o['name'] or o['object_type']=='Water'):
    scene[oid]={'id':oid,'name':o['name'],'category':o['object_type'],'position':o['position'][:2],'contour':o['contour']}
  hits=catalogs[r['map']].spans(r['instruction']);mentioned=[]
  for h in hits:
   ids,res,confidence=catalogs[r['map']].resolve(h)
   mentioned.append({**h,'resolved_ids':ids or [],'resolution':res,'confidence':confidence})
  inp={'occurrence_id':r['occurrence_id'],'cluster':p['cluster'],'episode':r['episode'],'map':r['map'],'origin':r['origin'],'instruction':r['instruction'],'word':r['semantic_word'],'direction_span':[r['span_start'],r['span_end']],'reference_ids':r['reference_ids'],'reference_text':r['reference_text'],'reference_span':r.get('reference_span'),'subject_role':r['subject_role'],'subject_ids':r['subject_ids'],'parse_tier':r['confidence_tier'],'parse_confidence':r['parse_confidence'],'binding_rule':r.get('binding_rule'),'start_xy':ep['trajectory'][0][:2],'mentioned':mentioned,'scene':scene}
  public.append(inp)
  ev={'occurrence_id':r['occurrence_id'],'cluster':p['cluster'],'map':r['map'],'word':r['semantic_word'],'origin':r['origin'],'parse_tier':r['confidence_tier'],'geometry_status':p['geometry_status'],'subject_xy':p['target_xy'],'reference_xy':p['reference_xy'],'angle_world':p['angle_world'],'target_polygon':obs[r['subject_ids'][0]]['contour'],'reference_ids':r['reference_ids'],'reference_category':p['reference_category'],'subject_category':p['relation_subject_category'],'polygon_distance_m':p.get('polygon_distance_m'),'intersection_area':p.get('intersection_area'),'episode':r['episode']}
  # Secondary roadside predicate: being on the LEFT HALF of a road need not be
  # outside the road polygon. This is explicitly separate from v2 disjoint-sector metrics.
  ev['roadside_eligible']=False
  if tp is not None and rp is not None and ev['reference_category']=='TrafficRoad' and r['semantic_word'] in {'left','right'}:
   ax=shape_axis(rp)
   if ax and ax['eigenvalue_ratio']>=8:
    a=np.deg2rad(ax['axis_deg']);normal=np.array([-np.sin(a),np.cos(a)]);c=np.array(rp.centroid.coords[0]);z=(vertices(tp)-c)@normal
    ev.update(roadside_eligible=bool(min(z)>1 or max(z)<-1),roadside_unsigned_axis_deg=ax['axis_deg'],roadside_axis_ratio=ax['eigenvalue_ratio'],roadside_center_xy=c.tolist(),roadside_min_margin=float(min(abs(min(z)),abs(max(z)))) if min(z)>0 or max(z)<0 else 0.)
  evaluation.append(ev)
 basevalid=[p for p in oldpairs if p['geometry_status']=='disjoint'];assert len(basevalid)==423 and sum(p['confidence_tier']=='high' for p in basevalid)==250
 for p in oldpairs:
  q=next(x for x in pairs if x['occurrence_id']==p['occurrence_id']);assert q['reference_ids']==p['reference_ids'] and q['confidence_tier']==p['confidence_tier'] and q['geometry_status']==p['geometry_status'] and np.allclose(q['target_xy'],p['target_xy'])
 writelines(out/'relations.jsonl.gz',rows);writelines(out/'public_inputs.jsonl.gz',public);writelines(out/'evaluation_gt.jsonl.gz',evaluation)
 pd.DataFrame(expansion).to_csv(out/'binding_expansion.csv',index=False);pd.DataFrame([{k:v for k,v in e.items() if k!='target_polygon'} for e in evaluation]).to_csv(out/'evaluation_cohorts.csv',index=False)
 writej(out/'dataset.json',{'base_commit':'a942369e3e5306701a38ee8666526c4e01a021c0','train_episodes':len(E),'landmark_candidates':len(rows),'bound':len(public),'v2_preserved_bound':len(oldpairs),'new_bound':len(expansion),'v2_disjoint':423,'v2_high_disjoint':250,'expanded_disjoint':sum(e['geometry_status']=='disjoint' for e in evaluation),'expanded_high_disjoint':sum(e['geometry_status']=='disjoint' and e['parse_tier']=='high' for e in evaluation),'roadside_secondary_eligible':sum(e['roadside_eligible'] for e in evaluation),'public_input_sha256':sha(out/'public_inputs.jsonl.gz'),'source_files':[{'path':str(f),'sha256':sha(f)} for f in [root/'original_citynav/citynav_train_seen.json',root/'cityrefer_original/objects.json']],'target_GT_not_in_public_inference_input':True,'future_trajectory_not_in_public_inference_input':True})
 print(json.dumps(readj(out/'dataset.json'),indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);a=ap.parse_args();run(a.data_root,BASE/'results')
