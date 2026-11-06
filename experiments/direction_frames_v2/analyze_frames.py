"""Evaluate prespecified candidate frames on immutable text bindings, TRAIN only."""
import argparse,collections,gzip,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from geometry_frames import compute_pair,evaluate,transforms,OFFSETS,adiff
from extract import dump
BASE=Path(__file__).parent;SEED=20261106;MAIN={'left','right','front','behind'}

def cluster_ci(df,vals,B=1000):
 if not len(df):return [None,None]
 g=pd.DataFrame({'cluster':df['cluster'].values,'v':np.asarray(vals,float)}).groupby('cluster').v.agg(['sum','count']);rng=np.random.default_rng(SEED);idx=rng.integers(0,len(g),(B,len(g)));a=g['sum'].values[idx].sum(1)/g['count'].values[idx].sum(1);return np.quantile(a,[.025,.975]).tolist()

def summarize(d):
 if not len(d):return {'n':0,'consistency':None,'cluster95':[None,None]}
 correct=d['sector_correct'].astype(float);p=d[d.polygon_class!='ambiguous'];p=p[p.polygon_class!='unavailable']
 return {'n':len(d),'clusters':int(d.cluster.nunique()),'consistency':float(correct.mean()),'cluster95':cluster_ci(d,correct),'mean_abs_error_deg':float(d.angle_error_deg.mean()),'median_abs_error_deg':float(d.angle_error_deg.median()),'halfplane_consistency':float(d.halfplane_correct.mean()),'polygon_evaluable':len(p),'polygon_ambiguous':int((d.polygon_class=='ambiguous').sum()),'polygon_consistency':float((p.polygon_class=='consistent').mean()) if len(p) else None,'polygon_cluster95':cluster_ci(p,(p.polygon_class=='consistent').astype(float))}

def run(root,out):
 rows=[json.loads(x) for x in gzip.open(out/'text_relations.jsonl.gz','rt')];E=json.loads((root/'original_citynav/citynav_train_seen.json').read_text());O=json.loads((root/'cityrefer_original/objects.json').read_text());pairs=[];ev=[];Ts=transforms();equiv=collections.defaultdict(list)
 for t in Ts:equiv[tuple(np.array(t['matrix']).ravel())].append(t['name'])
 uniqueA=[names[0] for names in equiv.values()]
 dump(out/'map_transformations.json',{'transformations':Ts,'equivalence_classes':list(equiv.values()),'unique_candidate_names':uniqueA,'global_dictionary':'base image left=-X, right=+X, front/above=+Y, behind/back/below=-Y IS A CANDIDATE ONLY. Transform entire dictionary jointly; do not choose per-word axes.'})
 for r in rows:
  if r['status']!='bound':continue
  pair,tp,rp=compute_pair(r,E[r['episode']],O[r['map']]);pairs.append(pair)
  for t in Ts:
   q=evaluate(pair,tp,rp,t['name'],t)
   if q:ev.append(q)
  for name,f in pair['frames'].items():
   q=evaluate(pair,tp,rp,name,f)
   if q:ev.append(q)
 with gzip.open(out/'geometry_pairs.jsonl.gz','wt',encoding='utf8') as fp:
  for p in pairs:fp.write(json.dumps(p,ensure_ascii=False)+'\n')
 pd.DataFrame([{k:v for k,v in p.items() if k!='frames'} for p in pairs]).to_csv(out/'geometry_pairs.csv',index=False)
 D=pd.DataFrame(ev);D.to_csv(out/'frame_evaluations.csv',index=False);P=pd.DataFrame(pairs)
 word_counts=collections.Counter(r['word'] for r in rows);flow=[]
 for w in word_counts:
  rr=[r for r in rows if r['word']==w];pp=P[P.word==w]
  flow.append({'word':w,'occurrences':len(rr),'text_bound':len(pp),'high_text_bound':int((pp.confidence_tier=='high').sum()),'medium_text_bound':int((pp.confidence_tier=='medium').sum()),'disjoint_all':int((pp.geometry_status=='disjoint').sum()),'disjoint_high':int(((pp.geometry_status=='disjoint')&(pp.confidence_tier=='high')).sum()),'polygon_ambiguous':int((pp.geometry_status=='ambiguous_overlap_or_boundary').sum()),'invalid_polygon':int((pp.geometry_status=='invalid_polygon').sum())})
 dump(out/'coverage.json',flow);pd.DataFrame(flow).to_csv(out/'coverage.csv',index=False)
 pd.DataFrame([{'word':r['word'],'reason':r['reason'],'scope':r['scope'],'v1_reason':r['v1_reason']} for r in rows if r['status']!='bound']).groupby(['word','reason','scope']).size().reset_index(name='n').to_csv(out/'exclusions.csv',index=False)
 transitions=collections.Counter((r['v1_status'],r['v1_reason'],r['status'],r['reason']) for r in rows)
 pd.DataFrame([{'v1_status':k[0],'v1_reason':k[1],'v2_status':k[2],'v2_reason':k[3],'n':n} for k,n in transitions.items()]).to_csv(out/'v1_v2_transitions.csv',index=False)
 summaries=[];groups=[];matched=[];cross=[];cross_predictions=[];axis=[];negative=[]
 for subset in ['high','all_effective']:
  valid=D[D.geometry_status=='disjoint'].copy()
  if subset=='high':valid=valid[valid.parse_tier=='high']
  for (w,h),sub in valid.groupby(['word','hypothesis']):
   s=summarize(sub);s.update(subset=subset,word=w,hypothesis=h,mention_coverage=len(sub)/word_counts[w],future=bool(sub.future.any()),target_conditioned=bool(sub.target_conditioned.any()),heading_verified=bool(sub.heading_verified.all()));summaries.append(s)
  for (h,key),sub in valid[valid.word.isin(MAIN)].groupby(['hypothesis','reference_category']):
   s=summarize(sub);s.update(subset=subset,group_type='reference_category',group=key,hypothesis=h);groups.append(s)
  for key in ['map','target_category','explicit_view','local_layout']:
   for (h,k),sub in valid[valid.word.isin(MAIN)].groupby(['hypothesis',key]):
    s=summarize(sub);s.update(subset=subset,group_type=key,group=str(k),hypothesis=h);groups.append(s)
  # Joint D4 transformation selected on held-out TARGET groups or MAPs.
  table=valid[valid.hypothesis.isin(uniqueA)&valid.word.isin(MAIN)].copy()
  idcols=['occurrence_id','cluster','map','word']
  for scheme in ['target_5fold','leave_map_out']:
   table['fold']=table.cluster.map(lambda x:int(hashlib.sha256(x.encode()).hexdigest()[:8],16)%5) if scheme=='target_5fold' else table['map']
   preds=[]
   for fold in sorted(table.fold.unique(),key=str):
    train=table[table.fold!=fold];test=table[table.fold==fold]
    if not len(train) or not len(test):continue
    ranked=train.groupby('hypothesis').agg(acc=('sector_correct','mean'),err=('angle_error_deg','mean')).sort_values(['acc','err'],ascending=[False,True]);best=ranked.index[0];picked=test[test.hypothesis==best].copy();picked['fitted_transform']=best;picked['fold']=str(fold);preds.append(picked)
    cross.append({'subset':subset,'scheme':scheme,'fold':str(fold),'fit_occurrences':train.occurrence_id.nunique(),'test_occurrences':test.occurrence_id.nunique(),'selected_transform':best})
   if not preds:continue
   pred=pd.concat(preds);pred['scheme']=scheme;pred['subset']=subset;cross_predictions.extend(pred.to_dict('records'))
   if scheme!='target_5fold':continue
   for w in sorted(MAIN|{'ALL_MAIN'}):
    baseline=pred if w=='ALL_MAIN' else pred[pred.word==w]
    s=summarize(baseline);s.update(subset=subset,word=w,hypothesis='A_joint_D4_crossfit',mention_coverage=len(baseline)/(sum(word_counts[x] for x in MAIN) if w=='ALL_MAIN' else word_counts[w]),future=False,target_conditioned=False,heading_verified=False);summaries.append(s)
   # Every proxy compared to the same crossfit A prediction on its own common support.
   base=pred.set_index('occurrence_id')
   for h,sub in valid[valid.word.isin(MAIN)&~valid.hypothesis.str.startswith('A_')].groupby('hypothesis'):
    common=sub[sub.occurrence_id.isin(base.index)].copy()
    if not len(common):continue
    a=base.loc[common.occurrence_id,'sector_correct'].values.astype(float);delta=common.sector_correct.values.astype(float)-a
    s=summarize(common);s.update(subset=subset,hypothesis=h,comparison='paired_against_joint_D4_crossfit',A_consistency_on_same_support=float(a.mean()),delta_consistency=float(delta.mean()),delta_cluster95=cluster_ci(common,delta),future=bool(common.future.any()),target_conditioned=bool(common.target_conditioned.any()));matched.append(s)
  # Axes that were independently fixed before seeing target GT: descriptive four-axis sweep.
  pp=P[P.geometry_status=='disjoint']
  if subset=='high':pp=pp[pp.confidence_tier=='high']
  for w,sub in pp.groupby('semantic_word'):
   for a in [0,90,180,270]:
    c=adiff(sub.angle_world.values,a)<=45;axis.append({'subset':subset,'word':w,'axis_deg':a,'n':len(sub),'consistency':float(c.mean()),'cluster95':cluster_ci(sub,c),'mean_abs_error_deg':float(adiff(sub.angle_world.values,a).mean())})
  # Future-frame negative control: shuffle direction labels within map on fixed support.
  # Not a causal test; indicates how much label-independent trajectory geometry can score.
  rng=np.random.default_rng(SEED)
  for h,sub in valid[valid.word.isin(MAIN)&valid.hypothesis.str.startswith('C_')].groupby('hypothesis'):
   offsets={'left':90,'right':270,'front':0,'behind':180};means=[]
   owner=P.set_index('occurrence_id')
   for _ in range(300):
    corr=[]
    for m,g in sub.groupby('map'):
     ws=rng.permutation(g.word.to_numpy());ang=owner.loc[g.occurrence_id,'angle_world'].to_numpy();head=np.array([owner.loc[rid,'frames'][h]['heading_deg'] for rid in g.occurrence_id]);corr.extend(adiff(ang,head+np.array([offsets[w] for w in ws]))<=45)
    means.append(np.mean(corr))
   negative.append({'subset':subset,'hypothesis':h,'n':len(sub),'observed':float(sub.sector_correct.mean()),'map_shuffled_word_mean':float(np.mean(means)),'map_shuffled_word_interval95':np.quantile(means,[.025,.975]).tolist(),'interpretation':'diagnostic randomization only; future motion or GT-conditioned window is never author-frame proof'})
 dump(out/'hypothesis_summary.json',summaries);pd.DataFrame(summaries).to_csv(out/'hypothesis_summary.csv',index=False)
 dump(out/'group_summary.json',groups);pd.DataFrame(groups).to_csv(out/'group_summary.csv',index=False)
 dump(out/'matched_comparisons.json',matched);pd.DataFrame(matched).to_csv(out/'matched_comparisons.csv',index=False)
 dump(out/'crossfit_folds.json',cross);pd.DataFrame(cross_predictions).to_csv(out/'crossfit_predictions.csv',index=False)
 dump(out/'word_axis_sweep.json',axis);pd.DataFrame(axis).to_csv(out/'word_axis_sweep.csv',index=False)
 dump(out/'future_motion_controls.json',negative)
 # Signed directions and unsigned-axis compatibility must never be conflated.
 axial=[]
 for p in pairs:
  a=p.get('reference_unsigned_axis');w=p['semantic_word']
  if p['geometry_status']!='disjoint' or not a or not a['reliable_unsigned_axis'] or w not in OFFSETS:continue
  err=min(float(adiff(p['angle_world'],a['axis_deg']+OFFSETS[w])),float(adiff(p['angle_world'],a['axis_deg']+180+OFFSETS[w])))
  axial.append({'occurrence_id':p['occurrence_id'],'cluster':p['cluster'],'word':w,'category':p['reference_category'],'parse_tier':p['confidence_tier'],'axis_error_mod180_deg':err,'axis_compatible':err<=45,'independent_direction_verified':False,'chance_if_uniform':.5})
 pd.DataFrame(axial).to_csv(out/'unsigned_axis_compatibility.csv',index=False)
 availability=[]
 expected=list(dict.fromkeys([t['name'] for t in Ts]+['B_start_to_reference','B_reference_to_start','C_initial_yaw','C_initial_actual_motion','C_motion_approaching_reference','C_motion_approaching_target','D_road_axis_sign0','D_road_axis_sign180','D_road_axis_aligned_initial_motion','D_road_local_tangent_start_sign0','D_road_local_tangent_start_sign180','D_building_axis_offset0','D_building_axis_offset90','D_building_axis_offset180','D_building_axis_offset270','D_verified_building_facade','D_verified_car_heading','D_parking_lane','A_true_geographic_north','D_explicit_stand_facing_water']))
 for h in expected:
  for tier in ['high','all_effective']:
   pp=[p for p in pairs if p['semantic_word'] in MAIN and (tier!='high' or p['confidence_tier']=='high')];eligible=[p for p in pp if p['geometry_status']=='disjoint'];supported=[p for p in eligible if h in p['frames'] and p['frames'][h]['heading_deg'] is not None] if not h.startswith('A_image') else eligible
   availability.append({'subset':tier,'hypothesis':h,'bound_main_relations':len(pp),'disjoint_main_relations':len(eligible),'supported_disjoint':len(supported),'coverage_among_disjoint':len(supported)/len(eligible) if eligible else None,'unavailable_reason':'no independently bound parking lane' if h=='D_parking_lane' else 'no directed heading annotation/verified image cue' if h in {'D_verified_car_heading','D_verified_building_facade'} else 'no independent geodetic calibration' if h=='A_true_geographic_north' else ''})
 dump(out/'availability.json',availability);pd.DataFrame(availability).to_csv(out/'availability.csv',index=False)
 dump(out/'corpus.json',{'episodes':len(E),'occurrences':len(rows),'text_bound':len(pairs),'high_text_bound':sum(p['confidence_tier']=='high' for p in pairs),'all_disjoint':sum(p['geometry_status']=='disjoint' for p in pairs),'high_disjoint':sum(p['geometry_status']=='disjoint' and p['confidence_tier']=='high' for p in pairs),'same_name_roads_retained_as_sets':sum(p['entity_resolution']=='all_named_road_segments' for p in pairs),'future_motion_deployment_allowed':False,'parse_scores_are_probabilities':False})
 print(json.dumps(json.loads((out/'corpus.json').read_text()),indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);ap.add_argument('--out',type=Path,default=BASE/'results');a=ap.parse_args();run(a.data_root,a.out)
