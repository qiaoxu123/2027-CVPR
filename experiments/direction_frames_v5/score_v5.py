"""Phase 2: GT is opened only after v5 roles and all geometry variants are frozen."""
from v5core import *
from geometry_v5 import *
from shapely.geometry import Point,Polygon
from shapely.ops import unary_union
import copy

def run():
 M=readj(RESULT/'phase1_manifest.json');assert M['phase1_fully_frozen'];assert sha(RESULT/'new_frames.jsonl.gz')==M['new_frames_sha256'] and sha(RESULT/'road_fields.jsonl.gz')==M['fields_sha256'] and sha(RESULT/'exploratory_fields.jsonl.gz')==M['exploratory_fields_sha256'] and sha(RESULT/'negative_fields.jsonl.gz')==M['negative_fields_sha256']
 X={x['occurrence_id']:x for x in readlines(RESULT/'public_inputs.jsonl.gz')};oldR=readlines(V4/'results/relations.jsonl.gz');new=readlines(RESULT/'new_candidates.jsonl.gz');NP={r['occurrence_id']:r for r in readlines(RESULT/'new_frames.jsonl.gz')};P={r['occurrence_id']:r for r in readlines(V4/'results/frozen_frames.jsonl.gz')};E={r['occurrence_id']:r for r in readlines(V4/'results/evaluation_gt.jsonl.gz')};F={f['key']:f for f in readlines(RESULT/'road_fields.jsonl.gz')};FF={f['key']:f for f in readlines(RESULT/'exploratory_fields.jsonl.gz')};NF={(f['key'],f['control']):f for f in readlines(RESULT/'negative_fields.jsonl.gz')};T=readj(DATA/'original_citynav/citynav_train_seen.json');O=readj(DATA/'cityrefer_original/objects.json');bynew={r['occurrence_id']:r for r in new};R={r['occurrence_id']:r for r in oldR}
 for r in new:
  if r['status']=='bound' or r['occurrence_id'] not in R:R[r['occurrence_id']]=r
 for rid,p in NP.items():
  r=bynew[rid];e=T[r['episode']];o=O[r['map']][r['target_id']];x=X[rid];rp=region(x,r['reference_ids']);gt=e['target_positions'][0][:2];pose=np.array(e['trajectory'][0]);E[rid]={'occurrence_id':rid,'cluster':r['cluster'],'map':r['map'],'word':r['word'],'parse_tier':r['confidence_tier'],'origin':r['origin'],'subject_xy':gt,'target_polygon':o['contour'],'reference_xy':(anchor(x,r['reference_ids']).tolist() if anchor(x,r['reference_ids']) is not None else np.mean([x['scene'][j]['position'] for j in r['reference_ids']],axis=0).tolist()),'reference_category':'TrafficRoad','subject_category':o['object_type'],'initial_yaw_deg':angle(pose[3:5]) if len(pose)==6 else float(np.degrees(pose[3])%360),'start_xy':pose[:2].tolist()}
 writelines(RESULT/'evaluation_gt.jsonl.gz',list(E.values()));records=[];projections={};methods=[];checks=[]
 for rid,r in R.items():
  p=NP.get(rid,P.get(rid));state=p['state'] if p and rid in NP else p['state'] if p else 'unknown';state='unique_field' if state=='unique_field' else 'multiple' if state in {'unique','multiple'} else 'unknown';bound=rid in X;fkey=p.get('road_field_key') if p else None
  q={'occurrence_id':rid,'cluster':r['cluster'],'map':r['map'],'word':r.get('semantic_word',r['word']),'bound':bound,'state':state,'result':'not_evaluable','reason':p.get('reason') if p else r.get('reason','unresolved'),'origin':'v5_upgrade' if rid in NP else 'v5_candidate_unknown' if rid in bynew and rid not in P else 'v4_frozen','road_group':r['map']+':'+','.join(r.get('reference_ids',[])),'road_field_key':fkey,'object_category':E[rid]['subject_category'] if rid in E else r.get('target_category','unbound'),'template':bynew[rid]['template'] if rid in bynew else 'v4_historical','novelty':bynew[rid]['novelty'] if rid in bynew else 'historical_development','parse_tier':r.get('confidence_tier','excluded'),'instruction':r['instruction']}
  for k in ['new_map_vs_v4_unique','new_road_vs_v4_unique','new_road_pair_vs_v4_unique','new_subject_vs_v4_unique']:q[k]=bynew[rid].get(k,False) if rid in bynew else False
  if state=='unique_field' and rid in E and q['word'] in {'left','right'}:
   f=F[fkey];e=E[rid];pr=projection(f,e['target_polygon']);projections[rid]=pr;answer=orientation_score(pr,q['word']);original=field_geometry(f,e['target_polygon'],q['word']);assert answer['result']==original['result'],rid;q.update(answer)
   methods.append({**q,'method':'A_frozen_v4_field','score':answer['score']})
   # Frozen end-to-end syntax versus explicitly exposed A/B roles is separate.
   if rid in NP:checks.append({'occurrence_id':rid,'frozen_v4_end_to_end_state':p['frozen_v4_end_to_end_state_on_new_input'],'new_language_adapter_geometry_state':state})
  records.append(q)
 # Common support and exact same projection/mask/predicate for A--E.
 rr=[r for r in records if r['occurrence_id'] in projections];gridcache={}
 for r in rr:
  rid=r['occurrence_id'];e=E[rid];x=X[rid];pr=projections[rid];f=F[r['road_field_key']];los=angle(np.array(e['reference_xy'])-np.array(e['start_xy']));axis=shape_axis(region(x,x['reference_ids']))
  for name,h in [('C_start_to_landmark',los),('D_initial_UAV_yaw',e['initial_yaw_deg'])]+[(f'B_world_forward_{h}',h) for h in [0,90,180,270]]:
   z=orientation_score(pr,r['word'],heading=h);methods.append({**r,**z,'method':name,'score':z['score']});gridcache[rid,name]=z['score']
  if axis:
   parts=[orientation_score(pr,r['word'],heading=axis['axis_deg']+delta) for delta in [0,180]];score=sum(z['score'] for z in parts)/2;methods.append({**r,'method':'E_unsigned_PCA_equal_signs','score':score,'result':'candidate_mixture','reason':'equal_mass_two_signs_NOT_best_sign','sign_results':[z['result'] for z in parts]})
  ff=FF[r['road_field_key']]
  if ff['state']=='unique_field':fp=projection(ff,e['target_polygon']);z=orientation_score(fp,r['word'])
  else:z={'score':0.,'result':'not_evaluable','reason':ff['reason']}
  methods.append({**r,**z,'method':'F_exploratory_smoothed_geometry','score':z['score']})
  z=orientation_score(pr,r['word'],rotation=180.);methods.append({**r,**z,'method':'negative_180_flip','score':z['score']})
  for control in ['swap_source_road','wrong_landmark_binding']:
   alt=NF[r['road_field_key'],control]
   if alt['state']=='unique_field':z=orientation_score(projection(alt,e['target_polygon']),r['word'])
   else:z={'score':0.,'result':'not_evaluable','reason':alt['reason']}
   methods.append({**r,**z,'method':'negative_'+control,'score':z['score'],'alternative_field_state':alt['state']})
 # World shared dictionary chosen only in other maps OR other named roads.
 for scheme in ['leave_map_out','leave_road_out']:
  for r in rr:
   group='map' if scheme=='leave_map_out' else 'road_group';train=[z for z in rr if z[group]!=r[group]];h=max([0,90,180,270],key=lambda h:sum(gridcache[z['occurrence_id'],f'B_world_forward_{h}'] for z in train));z=orientation_score(projections[r['occurrence_id']],r['word'],heading=h);methods.append({**r,**z,'method':'B_world_crossfit_'+scheme,'score':z['score'],'chosen_world_forward_deg':h,'fit_group_excluded':r[group]})
 cohorts={'all_named_candidates':records,'core_all_named':[r for r in records if r['word'] in MAIN],'bound_core':[r for r in records if r['bound'] and r['word'] in MAIN],'v4_original_unique44':[r for r in records if P.get(r['occurrence_id'],{}).get('state')=='unique_field'],'v5_all_new_candidates':[r for r in records if r['occurrence_id'] in bynew],'v5_new_bound':[r for r in records if r['occurrence_id'] in NP]}
 for flag in ['new_map_vs_v4_unique','new_road_vs_v4_unique','new_road_pair_vs_v4_unique','new_subject_vs_v4_unique']:cohorts['v5_'+flag]=[r for r in records if r['occurrence_id'] in NP and r[flag]]
 for novelty in ['previously_uncovered_expression','old_rule_semantics_missed_binding','duplicate_or_highly_similar']:cohorts['v5_'+novelty]=[r for r in records if r['occurrence_id'] in bynew and r['novelty']==novelty]
 summaries=[];breakdown=[];comparisons=[]
 for cohort,rows in cohorts.items():
  for tier in ['all','high']:summaries.append({'cohort':cohort,'tier':tier,**stats(rows if tier=='all' else [r for r in rows if r['parse_tier']=='high'])})
  for key in ['map','road_group','template','object_category','word']:
   for val in sorted({r[key] for r in rows}):breakdown.append({'cohort':cohort,'group_type':key,'group':val,**stats([r for r in rows if r[key]==val])})
  ids={r['occurrence_id'] for r in rows};base=[r for r in methods if r['method']=='A_frozen_v4_field' and r['occurrence_id'] in ids];baseby={r['occurrence_id']:r for r in base}
  for method in sorted({r['method'] for r in methods}):
   z=[r for r in methods if r['method']==method and r['occurrence_id'] in ids]
   if not z:continue
   assert {r['occurrence_id'] for r in z}==set(baseby)
   values=[r['score'] for r in z];delta=[r['score']-baseby[r['occurrence_id']]['score'] for r in z];comparisons.append({'cohort':cohort,'method':method,'same_support_n':len(z),'common_base_geometry_scorable':sum(baseby[r['occurrence_id']]['result'] in {'consistent','conflict'} for r in z),'confirmed_or_compatibility_all_support':float(np.mean(values)),'conditional_on_common_base_geometry':float(np.mean([r['score'] for r in z if baseby[r['occurrence_id']]['result'] in {'consistent','conflict'}])) if any(baseby[r['occurrence_id']]['result'] in {'consistent','conflict'} for r in z) else None,'object_cluster95':cluster_ci(z,values),'difference_from_A':float(np.mean(delta)),'paired_object95':cluster_ci(z,delta),'paired_road95':cluster_ci([{**r,'cluster':r['road_group']} for r in z],delta),'paired_map95':cluster_ci([{**r,'cluster':r['map']} for r in z],delta),'score_kind':'fixed equal-sign mixture compatibility' if method.startswith('E_') else 'full-contour confirmation fraction; ambiguous or unavailable=0','geometry_changed':method.startswith('F_') or method.startswith('negative_swap') or method.startswith('negative_wrong')})
 # Independent relation units: exact same object/road/direction/field cannot count twice.
 unique={}
 for r in rr:unique.setdefault((r['cluster'],r['road_group'],r['word'],r['road_field_key']),r)
 dedup=[{'cohort':'all_unique_fields',**stats(list(unique.values()))},{'cohort':'new_unique_fields',**stats([r for r in unique.values() if r['occurrence_id'] in NP])}]
 writelines(RESULT/'geometry_checks.jsonl.gz',records);savecsv('geometry_checks',records);writelines(RESULT/'method_predictions.jsonl.gz',methods);savecsv('method_comparisons',comparisons);savecsv('summary',summaries);savecsv('grouped_results',breakdown);savecsv('deduplicated',dedup);savecsv('frozen_language_comparison',checks)
 writej(RESULT/'evaluation_manifest.json',{'frozen_phase1_manifest_sha256':sha(RESULT/'phase1_manifest.json'),'GT_sha256':sha(RESULT/'evaluation_gt.jsonl.gz'),'score_code_sha256':sha(HERE/'score_v5.py'),'geometry_predicate_sha256':sha(HERE/'geometry_v5.py'),'target_GT_used_for_frame_selection':False,'future_trajectory_used':False,'raw_v4_A_result_reproduction':True,'same_projection_A_to_E':True,'same_primary_support_for_all_methods':True})
 print([(s['cohort'],s['n'],s['unique'],s['scorable'],s['consistent']) for s in summaries if s['tier']=='all'])
if __name__=='__main__':run()
