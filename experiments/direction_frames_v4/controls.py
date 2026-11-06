"""Matched-support nulls, semantic interventions and repeated-landmark audits."""
from shared import *
from infer import anchor,region
from frames import derive,angular_support
from roads import directed_field
from evaluate import field_geometry,metrics
import pandas as pd,copy

def role_view(x,source,dest):
 # Constructed LANGUAGE intervention on REAL fixed geometry. Never used as a
 # natural instruction or in real-data direction-consistency denominators.
 a=x['scene'][source[0]]['name'];b=x['scene'][dest[0]]['name'];ref=x['scene'][x['reference_ids'][0]]['name'];t=f'If you stand at {a} facing {b}, the target is to the right of {ref}.';y=copy.deepcopy(x);y['instruction']=t;y['word']='right';lo=t.index('right');y['direction_span']=[lo,lo+5];y['reference_span']=[t.rindex(ref),t.rindex(ref)+len(ref)];y['mentioned']=[]
 catalog=TypedRoadCatalog([Entity(i,o['name'],o['category']) for i,o in x['scene'].items() if o['name']])
 for h in catalog.spans(t):
  ids,res,confidence=catalog.resolve(h);y['mentioned'].append({**h,'resolved_ids':ids or [],'resolution':res,'confidence':confidence})
 return y

def fixed_origin_interventions(X):
 fixed_origin=[];map_inputs={}
 for x in X.values():
  if region(x,x['reference_ids']) is not None:map_inputs.setdefault(x['map'],x)
 for m,x in map_inputs.items():
  namecounts=collections.Counter(name_key(o['name']) for o in x['scene'].values() if o['name'])
  buildings=sorted([i for i,o in x['scene'].items() if o['category']=='Building' and o['name'] and namecounts[name_key(o['name'])]==1 and region(x,[i]) is not None],key=lambda i:(name_key(x['scene'][i]['name']),i))
  if len(buildings)<3:continue
  aa,bb=buildings[:2];origin=anchor(x,[aa]);base=angle(anchor(x,[bb])-origin);cc=max(buildings[2:],key=lambda i:float(adiff(angle(anchor(x,[i])-origin),base)))
  left=derive(role_view(x,[aa],[bb]),{});right=derive(role_view(x,[aa],[cc]),{});u=left.get('heading_support');v=right.get('heading_support')
  fixed_origin.append({'map':m,'constructed_language_control':True,'real_geometry_unchanged':True,'origin_ids_fixed':[aa],'destination_before':[bb],'destination_after':[cc],'landmark_selection':'first two lexically ordered public named buildings; third maximizes independent anchor separation, no subject/GT coordinates','GT_used':False,'origin_output_unchanged':left['origin_region_ids']==right['origin_region_ids']==[aa],'before_support':u,'after_support':v,'support_center_change_deg':float(adiff(u['heading_deg'],v['heading_deg'])) if u and v and u['heading_deg'] is not None and v['heading_deg'] is not None else None,'states':[left['state'],right['state']]})
 return fixed_origin


def run():
 X={r['occurrence_id']:r for r in readlines(OUT/'public_inputs.jsonl.gz')};P={r['occurrence_id']:r for r in readlines(OUT/'frozen_frames.jsonl.gz')};G={r['occurrence_id']:r for r in readlines(OUT/'evaluation_gt.jsonl.gz')};fields={r['key']:r for r in readlines(OUT/'road_fields.jsonl.gz')};Q=readlines(OUT/'geometry_checks.jsonl.gz');unique=[r for r in Q if r['state']=='unique_field' and r['word'] in {'left','right'}];rng=np.random.default_rng(SEED);B=300;null=[];details=[]
 for mode in ['independent_random_rotation','road_positive_direction_180_flip']:
  vals=[]
  for b in range(B if mode=='independent_random_rotation' else 1):
   results=[]
   for r in unique:
    p=P[r['occurrence_id']];g=G[r['occurrence_id']];rot=float(rng.uniform(0,360)) if mode=='independent_random_rotation' else 180.;results.append(field_geometry(fields[p['road_field_key']],g['target_polygon'],r['word'],rotation=rot)['result']=='consistent')
   vals.append(float(np.mean(results)))
  observed=float(np.mean([r['result']=='consistent' for r in unique]));null.append({'control':mode,'support_n':len(unique),'coverage_retained':True,'B':len(vals),'observed_confirmed_fraction':observed,'null_mean':float(np.mean(vals)),'null95':np.quantile(vals,[.025,.975]).tolist(),'one_sided_MC_p':float((1+sum(z>=observed for z in vals))/(1+len(vals))) if len(vals)>1 else None,'unknown_ambiguous_as_unconfirmed':True})
 for r in unique:
  rid=r['occurrence_id'];p=P[rid];x=X[rid];g=G[rid];f=fields[p['road_field_key']];roadids=f['road_ids'];source=f['source_ids']
  # Reversal identity tests geometry/role algebra, not author-frame accuracy.
  swappedword='right' if r['word']=='left' else 'left';reverse=field_geometry(f,g['target_polygon'],swappedword,rotation=180.)
  details.append({'occurrence_id':rid,'control':'opposite_word_and_reverse_positive_direction_identity','result':reverse['result'],'original_result':r['result'],'identical':reverse['result']==r['result']})
  for mode in ['wrong_landmark_binding','semantic_road_A_B_swap']:
   if mode=='wrong_landmark_binding':
    names=collections.defaultdict(list)
    for i,o in x['scene'].items():
     if o['category']=='TrafficRoad' and o['name'] and i not in roadids+source:names[name_key(o['name'])].append(i)
    choices=sorted(names);wrong=names[choices[fold_id(rid)%len(choices)]] if choices else []
    alt=directed_field(x,wrong,source) if wrong else {'state':'unknown','reason':'no_wrong_reference_available'}
   else:wrong=source;alt=directed_field(x,source,roadids)
   result=field_geometry(alt,g['target_polygon'],r['word'])['result'] if alt['state']=='unique_field' else 'not_evaluable'
   details.append({'occurrence_id':rid,'control':mode,'state':alt['state'],'wrong_reference_ids':wrong,'result':result,'original_result':r['result'],'GT_used_to_choose_wrong_binding':False,'reason':alt.get('reason')})
 for mode in ['wrong_landmark_binding','semantic_road_A_B_swap']:
  rr=[r for r in details if r['control']==mode];n=len(rr);k=sum(r['state']=='unique_field' for r in rr);ok=sum(r['result']=='consistent' for r in rr);null.append({'control':mode,'support_n':n,'unique_fields_emitted':k,'unique_fraction':k/n,'confirmed_n':ok,'confirmed_fraction_all_support':ok/n,'conditional_consistency':ok/k if k else None,'note':'reference roles changed; abstentions retained in original support denominator'})
 # Fixed-map intervention: A->B versus B->A rotates admissible observation support
 # by 180deg without altering ANY scene polygon. These are constructed controls.
 interventions=[]
 for key,f in fields.items():
  if not f.get('road_ids') or not f.get('source_ids'):continue
  x=next((x for x in X.values() if x['map']==key.split(':')[0] and all(i in x['scene'] for i in f['road_ids']+f['source_ids'])),None)
  if not x:continue
  left=derive(role_view(x,f['road_ids'],f['source_ids']),{});right=derive(role_view(x,f['source_ids'],f['road_ids']),{});u=left.get('heading_support');v=right.get('heading_support')
  if u and v and (u['heading_deg'] is None or v['heading_deg'] is None):interventions.append({'road_key':key,'constructed_language_control':True,'real_geometry_unchanged':True,'informative_role_vector':False,'reason':'overlapping_anchor_regions_allow_full_360deg; role reversal cannot identify a heading','states':[left['state'],right['state']]})
  if u and v and u['heading_deg'] is not None and v['heading_deg'] is not None:interventions.append({'road_key':key,'constructed_language_control':True,'real_geometry_unchanged':True,'GT_read_for_prediction':False,'forward_support_width_deg':u['width_deg'],'reverse_support_width_deg':v['width_deg'],'support_center_rotation_deg':float(adiff(u['heading_deg'],v['heading_deg'])),'expected_180_error_deg':abs(180-float(adiff(u['heading_deg'],v['heading_deg']))),'states':[left['state'],right['state']]})
 writej(OUT/'fixed_origin_view_interventions.json',fixed_origin_interventions(X))
 # Shared observed view: named-to-named constraints use PUBLIC coordinates only.
 shared=[]
 for x in X.values():
  p=P[x['occurrence_id']];cc=[c for c in p.get('candidates',[]) if c['type']=='named_pair_shared_frame_constraint']
  if not cc:continue
  hs=np.arange(360.);valid=np.ones(360,dtype=bool)
  for c in cc:valid&=adiff(hs,c['heading_deg'])<90.
  shared.append({'occurrence_id':x['occurrence_id'],'n_independent_named_pair_constraints':len(cc),'allowed_heading_grid_deg':hs[valid].tolist(),'allowed_fraction':float(valid.mean()),'exact_yaw_identified':False,'shared_scope_assumption_verified':False,'subject_geometry_not_in_scene':not(set(x['subject_ids'])&set(x['scene'])),'note':'binary side constraints define intervals, not an exact observed yaw'})
 # Same landmark/source invariance and same-object opposite wording.
 repeat=[];pairs=[];byref=collections.defaultdict(list)
 for r in Q:
  if r['occurrence_id'] in X:byref[r['reference_group']].append(r)
 for ref,rr in byref.items():
  uu=[r for r in rr if r['state']=='unique_field'];keys={P[r['occurrence_id']]['road_field_key'] for r in uu};repeat.append({'reference_group':ref,'relation_count':len(rr),'instruction_count':len({r['source_instruction'] for r in rr}),'unique_field_relations':len(uu),'source_dependent_fields':len(keys),'same_source_same_frozen_field_by_construction':True})
  bysubject=collections.defaultdict(list)
  for r in rr:bysubject[r['cluster']].append(r)
  for subject,zz in bysubject.items():
   for i,u in enumerate(zz):
    for v in zz[i+1:]:
     if {u['word'],v['word']} not in [{'left','right'},{'front','behind'}]:continue
     pu=P[u['occurrence_id']];pv=P[v['occurrence_id']];pairs.append({'reference_group':ref,'subject_cluster':subject,'relation_a':u['occurrence_id'],'relation_b':v['occurrence_id'],'words':[u['word'],v['word']],'states':[u['state'],v['state']],'same_independent_road_frame':bool(pu.get('road_field_key') and pu.get('road_field_key')==pv.get('road_field_key') and u['state']==v['state']=='unique_field'),'direction_results':[u['result'],v['result']],'proves_annotation_contradiction':False,'reason':'author view unverified; most opposite pairs still lack an identified shared frame'})
 writej(OUT/'negative_controls.json',null);pd.DataFrame(null).to_csv(OUT/'negative_controls.csv',index=False);writelines(OUT/'negative_control_details.jsonl.gz',details);writej(OUT/'fixed_geometry_language_interventions.json',interventions);writej(OUT/'shared_view_constraints.json',shared);writej(OUT/'same_landmark_instructions.json',repeat);pd.DataFrame(repeat).to_csv(OUT/'same_landmark_instructions.csv',index=False);writej(OUT/'opposite_relation_pairs.json',pairs);pd.DataFrame(pairs).to_csv(OUT/'opposite_relation_pairs.csv',index=False)
 print(null);print('interventions',len(interventions),'opposite_pairs',len(pairs),'shared',len(shared))
if __name__=='__main__':
 if '--fixed-origin-only' in sys.argv:
  xx={r['occurrence_id']:r for r in readlines(OUT/'public_inputs.jsonl.gz')};writej(OUT/'fixed_origin_view_interventions.json',fixed_origin_interventions(xx));print('fixed-origin interventions complete')
 else:run()
