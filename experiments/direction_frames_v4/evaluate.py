"""Phase 2 offline geometry evaluation of byte-frozen semantic predictions."""
from shared import *
from shapely.geometry import Polygon,LineString
from shapely.ops import unary_union
from geometry_frames import boundary_class,vertices
import pandas as pd

def wilson(k,n):
 if not n:return [None,None]
 z=1.95996398454;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [float(m-r),float(m+r)]

def boundary_samples(contour,step=1.):
 p=Polygon(contour);line=p.exterior;return np.array([line.interpolate(s).coords[0] for s in np.linspace(0,line.length,max(4,int(np.ceil(line.length/step))),endpoint=False)])

def field_geometry(f,contour,word,rotation=0.):
 if not f.get('edges'):return {'result':'not_evaluable','reason':'no_centerline'}
 xy=np.array(f['points_xy']);edges=np.array(f['edges']);a=xy[edges[:,0]];v=xy[edges[:,1]]-a;ll=(v*v).sum(1);d=np.array(f['distance_from_source_m']);orient=np.where(d[edges[:,1]]>d[edges[:,0]],1.,-1.);vdir=v*orient[:,None];heads=np.deg2rad(np.array([0. if h is None else h for h in f['heading_deg']]));unit=np.stack([np.cos(heads),np.sin(heads)],axis=1);smooth=unit[edges[:,0]]+unit[edges[:,1]];smooth/=np.linalg.norm(smooth,axis=1)[:,None];pts=boundary_samples(contour);rel=pts[:,None,:]-a[None,:,:];s=np.clip(np.sum(rel*v[None,:,:],axis=2)/ll,0,1);proj=a[None,:,:]+s[:,:,None]*v[None,:,:];dist=np.sum((pts[:,None,:]-proj)**2,axis=2);ix=np.argmin(dist,axis=1);q=proj[np.arange(len(pts)),ix];u=smooth[ix];rad=np.deg2rad(rotation);u=u@np.array([[np.cos(rad),np.sin(rad)],[-np.sin(rad),np.cos(rad)]]);normal=np.stack([-u[:,1],u[:,0]],axis=1);side=((pts-q)*normal).sum(1);expected=side if word=='left' else -side
 # Ambiguous nearest branches/bends are not chosen by agreement with the word.
 ties=dist<=dist.min(1)[:,None]+.25
 for j in range(len(pts)):
  zz=np.flatnonzero(ties[j]);hh=np.degrees(np.arctan2(smooth[zz,1],smooth[zz,0]));base=np.degrees(np.arctan2(smooth[ix[j],1],smooth[ix[j],0]))
  if len(zz)>1 and np.max(adiff(hh,base))>30:return {'result':'ambiguous','reason':'multiple_nearest_tangents','margin_m':float(expected.min())}
 line=unary_union([LineString([xy[i],xy[j]]) for i,j in edges]);target=Polygon(contour)
 if target.intersects(line.buffer(1.)):return {'result':'ambiguous','reason':'target_contour_touches_centerline_tolerance','margin_m':float(expected.min())}
 result='consistent' if expected.min()>1 else 'conflict' if expected.max()<-1 else 'ambiguous'
 return {'result':result,'reason':'full_contour_directed_side','margin_m':float(expected.min()),'heading_at_boundary_mean_deg':angle(np.mean(u,axis=0)),'n_boundary_samples':len(pts)}

def constant_side(e,h):
 ref=unary_union([Polygon(c) for c in e['reference_polygon']]);origin=np.array(ref.centroid.coords[0]);a=np.deg2rad(h);normal=np.array([-np.sin(a),np.cos(a)]);z=(np.array(e['target_polygon'])-origin)@normal;z=z if e['word']=='left' else -z
 return 'consistent' if min(z)>1 else 'conflict' if max(z)<-1 else 'ambiguous'

def metrics(rows):
 n=len(rows);unique=[r for r in rows if r['state'] in {'unique','unique_field'}];sc=[r for r in unique if r['result'] in {'consistent','conflict'}];k=sum(r['result']=='consistent' for r in sc);values=[float(r['result']=='consistent') for r in sc]
 return {'n':n,'unique':len(unique),'unique_fraction':len(unique)/n if n else None,'multiple':sum(r['state']=='multiple' for r in rows),'unknown':sum(r['state']=='unknown' for r in rows),'geometry_scorable':len(sc),'consistent':k,'conflict':len(sc)-k,'unique_geometry_ambiguous_or_unavailable':len(unique)-len(sc),'conditional_consistency':k/len(sc) if sc else None,'wilson95':wilson(k,len(sc)),'object_cluster95':cluster_ci(sc,values),'reference_cluster95':cluster_ci([{**r,'cluster':r['reference_group']} for r in sc],values),'map_cluster95':cluster_ci([{**r,'cluster':r['map']} for r in sc],values),'end_to_end_confirmed_fraction':k/n if n else None,'unique_success_lower_bound':k/len(unique) if unique else None}

def run():
 F=readj(OUT/'freeze_manifest.json');assert sha(OUT/'frozen_frames.jsonl.gz')==F['frame_predictions_sha256'];X={r['occurrence_id']:r for r in readlines(OUT/'public_inputs.jsonl.gz')};P={r['occurrence_id']:r for r in readlines(OUT/'frozen_frames.jsonl.gz')};E={r['occurrence_id']:r for r in readlines(OUT/'evaluation_gt.jsonl.gz')};fields={r['key']:r for r in readlines(OUT/'road_fields.jsonl.gz')};R=readlines(OUT/'relations.jsonl.gz');records=[]
 assert set(E)==set(X),'GT materialization incomplete or stale'
 gm=readj(OUT/'GT_phase_manifest.json');assert gm['GT_sha256']==sha(OUT/'evaluation_gt.jsonl.gz') and gm['frozen_predictions_sha256_before_GT']==F['frame_predictions_sha256']
 for r in R:
  rid=r['occurrence_id'];p=P[rid];e=E.get(rid);q={'occurrence_id':rid,'cluster':r['cluster'],'map':r['map'],'word':r['semantic_word'],'origin':r['origin'],'parse_tier':r.get('confidence_tier','excluded'),'reference_group':r['map']+':'+','.join(r.get('reference_ids',[])),'reference_category':e['reference_category'] if e else 'unbound','frame_type':p['frame_type'],'state':p['state'],'result':'not_evaluable','reason':p['reason'],'metric':'none','geometry_status':e['geometry_status'] if e else 'unbound','source_instruction':r['instruction']}
  if p['state']=='unique_field' and e and e['word'] in {'left','right'}:
   q.update(metric='directed_curved_road_full_contour_side',**field_geometry(fields[p['road_field_key']],e['target_polygon'],e['word']))
  elif p['state']=='unique' and e and e['geometry_status']=='disjoint':
   h=p['forward_heading_deg'];err=float(adiff(e['angle_world'],h+OFFSETS[e['word']]));u=np.array([np.cos(np.deg2rad(h+OFFSETS[e['word']])),np.sin(np.deg2rad(h+OFFSETS[e['word']]))]);rp=unary_union([Polygon(c) for c in e['reference_polygon']]);bc,margin,opp=boundary_class(Polygon(e['target_polygon']),rp,u)
   q.update(metric='object_angular_sector_and_polygon',result='consistent' if err<=45 else 'conflict',angle_error_deg=err,polygon_result=bc,polygon_margin_m=margin)
  elif p.get('heading_support') and e and e['geometry_status']=='disjoint':
   hs=np.array(p['heading_support']['grid_deg']);errs=adiff(e['angle_world'],hs+OFFSETS[e['word']]);q.update(candidate_sector_compatibility=float(np.mean(errs<=45)),all_candidates_consistent=bool(np.all(errs<=45)),any_candidate_consistent=bool(np.any(errs<=45)),metric='candidate_range_diagnostic_NOT_identified_accuracy')
  records.append(q)
 writelines(OUT/'geometry_checks.jsonl.gz',records);pd.DataFrame(records).to_csv(OUT/'geometry_checks.csv',index=False)
 cohorts={'all_named_candidates':records,'core_all_named':[r for r in records if r['word'] in MAIN],'core_bound':[r for r in records if r['word'] in MAIN and r['occurrence_id'] in X],'v3_core425':[r for r in records if r['word'] in MAIN and r['geometry_status']=='disjoint' and not r['origin'].startswith('v4')],'expanded_core_disjoint':[r for r in records if r['word'] in MAIN and r['geometry_status']=='disjoint'],'road_semantic':[r for r in records if r['frame_type']=='road' and r['word'] in {'left','right'}],'new_semantic_road_anchors':[r for r in records if r['origin']=='v4_explicit_road_anchor']}
 summary=[];breakdown=[]
 for cohort,rr in cohorts.items():
  for tier in ['all','high']:
   qq=rr if tier=='all' else [r for r in rr if r['parse_tier']=='high'];summary.append({'cohort':cohort,'tier':tier,**metrics(qq)})
  for key in ['word','map','reference_category','state','origin']:
   for val in sorted({r[key] for r in rr}):breakdown.append({'cohort':cohort,'group_type':key,'group':val,**metrics([r for r in rr if r[key]==val])})
 writej(OUT/'summary.json',summary);pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);writej(OUT/'breakdown.json',breakdown);pd.DataFrame(breakdown).to_csv(OUT/'breakdown.csv',index=False)
 # Rule system has no learned parameters: leave-map-out is a transportability
 # breakdown, not a claim that seeing development maps was blinded.
 maps=[]
 for m in sorted({r['map'] for r in records}):
  q=[r for r in records if r['map']==m and r['frame_type']=='road' and r['word'] in {'left','right'}];maps.append({'heldout_map':m,'fitted_parameters':0,'same_frozen_rules':True,**metrics(q)})
 writej(OUT/'leave_map_out.json',maps);pd.DataFrame(maps).to_csv(OUT/'leave_map_out.csv',index=False)
 # Matched unique-road population, global world heading calibrated on OTHER maps.
 road=[r for r in records if r['state']=='unique_field' and r['word'] in {'left','right'}];baseline=[]
 for r in road:
  e=E[r['occurrence_id']];train=[E[z['occurrence_id']] for z in records if z['map']!=r['map'] and z['frame_type']=='road' and z['word'] in {'left','right'} and z['occurrence_id'] in E and E[z['occurrence_id']]['reference_polygon']]
  h=max([0,90,180,270],key=lambda h:sum(constant_side(z,h)=='consistent' for z in train));start=np.array(e['start_xy']);ref=np.array(e['reference_xy']);los=angle(ref-start)
  for name,hh in [('world_crossfit_other_maps',h),('initial_yaw',e['initial_yaw_deg']),('start_to_landmark',los),('landmark_to_start',(los+180)%360)]:baseline.append({'occurrence_id':r['occurrence_id'],'cluster':r['cluster'],'map':r['map'],'baseline':name,'heading_deg':hh,'result':constant_side(e,hh),'semantic_result':r['result'],'same_phase1_unique_support':True})
 writej(OUT/'matched_baselines.json',baseline);pd.DataFrame(baseline).to_csv(OUT/'matched_baselines.csv',index=False)
 bs=[]
 for name in sorted({r['baseline'] for r in baseline}):
  qq=[r for r in baseline if r['baseline']==name];delta=[float(r['semantic_result']=='consistent')-float(r['result']=='consistent') for r in qq];bs.append({'baseline':name,'support_n':len(qq),'semantic_confirmed_fraction':float(np.mean([r['semantic_result']=='consistent' for r in qq])),'baseline_confirmed_fraction':float(np.mean([r['result']=='consistent' for r in qq])),'paired_difference':float(np.mean(delta)),'object_cluster95':cluster_ci(qq,delta),'map_cluster95':cluster_ci([{**r,'cluster':r['map']} for r in qq],delta),'ambiguous_counted_as_unconfirmed':True})
 writej(OUT/'matched_baseline_summary.json',bs)
 writej(OUT/'evaluation_manifest.json',{'phase':2,'frozen_prediction_sha256':sha(OUT/'frozen_frames.jsonl.gz'),'frozen_road_fields_sha256':sha(OUT/'road_fields.jsonl.gz'),'GT_sha256':sha(OUT/'evaluation_gt.jsonl.gz'),'checks_sha256':sha(OUT/'geometry_checks.jsonl.gz'),'code_sha256':sha(BASE4/'evaluate.py'),'orientation_selection_uses_GT':False,'boundary_tolerance_m':1.,'boundary_sampling_max_step_m':1.})
 print('SUMMARY',[(s['cohort'],s['tier'],s['n'],s['unique'],s['geometry_scorable'],s['consistent']) for s in summary]);print('BASELINES',bs)
if __name__=='__main__':run()
