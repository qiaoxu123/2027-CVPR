"""Classify all frozen-v4 ambiguous cases; no diagnosis is used to flip direction."""
from v5core import *
from geometry_v5 import *

def run():
 P={p['occurrence_id']:p for p in readlines(V4/'results/frozen_frames.jsonl.gz')};X={x['occurrence_id']:x for x in readlines(V4/'results/public_inputs.jsonl.gz')};G={e['occurrence_id']:e for e in readlines(V4/'results/evaluation_gt.jsonl.gz')};Q=readlines(V4/'results/geometry_checks.jsonl.gz');F={f['key']:f for f in readlines(RESULT/'road_fields.jsonl.gz')};FF={f['key']:f for f in readlines(RESULT/'exploratory_fields.jsonl.gz')};profiles=readlines(RESULT/'resolution_fields.jsonl.gz');rows=[];variants=[]
 for q in Q:
  if q['state']!='unique_field':continue
  rid=q['occurrence_id'];g=G[rid];x=X[rid];f=F[P[rid]['road_field_key']];pr=projection(f,g['target_polygon']);result=orientation_score(pr,q['word']);assert result['result']==q['result'];tp=Polygon(g['target_polygon']);gt=Point(*g['subject_xy']);point=point_side(f,g['subject_xy'],q['word']);centroid=point_side(f,list(tp.centroid.coords)[0],q['word']);tols=[]
  for tol in [0.,.5,1.,2.]:
   z=orientation_score(projection(f,g['target_polygon'],tol=tol),q['word'],tol=tol);tols.append({'tol_m':tol,'result':z['result']})
  vr=[]
  for alt in profiles:
   if alt['base_key']!=f['key']:continue
   z=orientation_score(projection(alt,g['target_polygon']),q['word']) if alt['state']=='unique_field' else {'result':'not_evaluable','reason':alt['reason']};vr.append({'resolution_m':alt['profile_resolution_m'],'state':alt['state'],'result':z['result']});variants.append({'occurrence_id':rid,**vr[-1]})
  ff=FF[f['key']];fz=orientation_score(projection(ff,g['target_polygon']),q['word']) if ff['state']=='unique_field' else {'result':'not_evaluable'}
  trigger='none'
  if q['result']=='ambiguous':
   if pr['mask_reason']=='multiple_nearest_tangents':trigger='nearest_tangent_nonunique'
   elif tp.intersects(pr['line']):trigger='polygon_intersects_centerline'
   elif tp.distance(pr['line'])<=1:trigger='boundary_tolerance_only'
   else:trigger='curvature_or_endcap_full_contour_side_nonunique'
  stable=all(z['state']=='unique_field' and z['result']==q['result'] for z in vr)
  attribution='not_ambiguous'
  if q['result']=='ambiguous':
   if not tp.buffer(1).covers(gt):attribution='GT_point_vs_object_polygon_disagreement'
   elif trigger=='boundary_tolerance_only':attribution='strict_1m_margin_conservative_effect_verified'
   elif not stable:attribution='geometry_resolution_sensitive_ERROR_NOT_INDEPENDENTLY_PROVEN'
   else:attribution='full_contour_side_indeterminate_under_all_tested_centerlines'
  rows.append({'occurrence_id':rid,'map':q['map'],'object_category':g['subject_category'],'word':q['word'],'frozen_result':q['result'],'verified_geometric_trigger':trigger,'attribution_status':attribution,'centerline_error_confirmed_by_independent_lane_annotation':False,'GT_point_side':point['result'],'GT_point_margin_m':point['signed_margin_m'],'centroid_side':centroid['result'],'GT_inside_target_polygon_1m':tp.buffer(1).covers(gt),'GT_to_polygon_m':float(tp.distance(gt)),'polygon_area_m2':float(tp.area),'distance_polygon_centerline_m':pr['distance_to_centerline_m'],'tolerance_diagnostics':tols,'resolution_diagnostics':vr,'all_resolution_results_stable':stable,'F_exploration_result':fz['result'],'strict_polygon_predicate_retained':True,'instruction':x['instruction']})
 savecsv('v4_all44_reaudit',rows);savecsv('v4_ambiguous22_classification',[r for r in rows if r['frozen_result']=='ambiguous']);savecsv('resolution_sensitivity',variants)
 abbey=next(r for r in rows if r['occurrence_id']=='cambridge_block_28:17:3:26');rid=abbey['occurrence_id'];p=P[rid];x=X[rid];f=F[p['road_field_key']];topo=next(r for r in readj(RESULT/'source_topology_audit.json') if r['key']==f['key']);related=[r for r in rows if P[r['occurrence_id']]['road_field_key']==f['key']];O=readj(DATA/'cityrefer_original/objects.json')
 writej(RESULT/'Abbey_failure_analysis.json',{'case':abbey,'reference_ids':x['reference_ids'],'source_ids':f['source_ids'],'source_components':[{'id':i,'name':x['scene'][i]['name'],'category':x['scene'][i]['category'],'distance_to_A_m':region(x,[i]).distance(region(x,f['road_ids']))} for i in f['source_ids']],'topology':topo,'original_scope':p['scope_span'],'language_evidence':p['evidence'],'all_object_descriptions':O[x['map']]['17']['descriptions'],'same_road_entry_cases':related,'target_binding_original_annotation_exact':True,'direction_reversed_to_fit_GT':False,'conclusion':'独立名称/作用域、单端接触和多分辨率方向一致；字面进入道路的侧别规则仍冲突。不能归咎于已证实拓扑错误，也缺少作者视角/意图金标准来确认真实标注错误。保留语义约定/标注不一致未决，不翻转方向。'})
 # Cheap fixed-support random-rotation control with identical cached projection.
 rng=np.random.default_rng(SEED);active=[r for r in readlines(RESULT/'geometry_checks.jsonl.gz') if r['state']=='unique_field'];GP={r['occurrence_id']:r for r in readlines(RESULT/'evaluation_gt.jsonl.gz')};FP={**P,**{r['occurrence_id']:r for r in readlines(RESULT/'new_frames.jsonl.gz')}};cache={r['occurrence_id']:projection(F[FP[r['occurrence_id']]['road_field_key']],GP[r['occurrence_id']]['target_polygon']) for r in active};observed=float(np.mean([r['result']=='consistent' for r in active]));null=[]
 for _ in range(300):null.append(float(np.mean([orientation_score(cache[r['occurrence_id']],r['word'],rotation=float(rng.uniform(0,360)))['score'] for r in active])))
 writej(RESULT/'random_rotation_control.json',{'support_n':len(active),'B':300,'observed_confirmed_all':observed,'null_mean':float(np.mean(null)),'null95':np.quantile(null,[.025,.975]).tolist(),'one_sided_MC_p':float((1+sum(v>=observed for v in null))/301),'fixed_geometric_mask':True,'exploratory_multiple_testing_unadjusted':True})
 print('v4 reproduction',len(rows),collections.Counter(r['frozen_result'] for r in rows));print('ambiguous classification',collections.Counter(r['verified_geometric_trigger'] for r in rows if r['frozen_result']=='ambiguous'))
if __name__=='__main__':run()
