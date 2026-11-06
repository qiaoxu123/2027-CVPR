"""Crossfit weak semantic router and evaluate already-frozen frames using GT offline."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from common import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
from infer import routing_ids,FAMILIES
from shapely.geometry import Polygon
from shapely.ops import unary_union

def candidate_weights(pred,ids):
 pool=[c for c in pred['candidates'] if c['candidate_id'] in ids];types=collections.Counter(c['type'] for c in pool)
 return {c['candidate_id']:1/len(types)/types[c['type']] for c in pool} if types else {}

def wilson(p,n):
 if n==0:return [None,None]
 z=1.95996398454;d=1+z*z/n;m=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [float(m-r),float(m+r)]

def local_score(pred,weights,ev,metric='sector',rotation=0.):
 if not weights:return None,None,None
 by={c['candidate_id']:c for c in pred['candidates']};v=np.array(ev['subject_xy'])-np.array(ev['reference_xy']);scores=[];errs=[]
 for cid,w in weights.items():
  c=by[cid];h=(c['heading_deg']+rotation)%360;expect=(h+OFFSETS[ev['word']])%360;err=float(adiff(ev['angle_world'],expect));errs.append(w*err)
  if metric=='sector':s=err<=45
  elif metric=='halfplane':
   a=np.deg2rad(expect);s=float(v@np.array([np.cos(a),np.sin(a)]))>1
  else:
   # Roadside predicate: evaluate target contour relative to an independently fixed
   # road centerline. Using GT to TEST the side does not choose frame or sign.
   a=np.deg2rad(h);left=np.array([-np.sin(a),np.cos(a)]);x=(np.array(ev['target_polygon'])-np.array(ev['roadside_center_xy']))@left
   s=min(x)>1 if ev['word']=='left' else max(x)<-1
  scores.append(w*float(s))
 return float(sum(scores)),float(sum(errs)),bool(len(weights)==1)

def uniform_orientation_score(ev,metric):
 if metric=='sector':return .25
 if metric=='halfplane':
  d=np.linalg.norm(np.array(ev['subject_xy'])-np.array(ev['reference_xy']))
  return float(np.arccos(1/d)/np.pi) if d>1 else 0.
 a=np.linspace(0,2*np.pi,720,endpoint=False);u=np.stack([np.cos(a),np.sin(a)],axis=1);v=np.array(ev['target_polygon'])-np.array(ev['roadside_center_xy']);return float((np.min(v@u.T,axis=0)>1).mean())

def summarize(records,local=False):
 n=len(records);covered=[r for r in records if r['local_score'] is not None];k=len(covered)
 if not n:return {'n':0}
 vals=[r['local_score'] if local else r['hybrid_score'] for r in (covered if local else records)];rr=covered if local else records
 out={'n':n,'covered':k,'coverage':k/n,'unknown_fraction':1-k/n,'consistency':float(np.mean(vals)) if vals else None,'cluster95':cluster_ci(rr,vals) if vals else [None,None],'mean_angle_error_deg':float(np.mean([r['local_error'] if local else r['hybrid_error'] for r in rr])) if rr else None,'coverage_adjusted_success':float(sum(r['local_score'] or 0 for r in records)/n),'baseline_consistency_on_covered':float(np.mean([r['baseline_score'] for r in covered])) if covered else None,'baseline_consistency_all':float(np.mean([r['baseline_score'] for r in records])),'net_improvement_all':float(np.mean([r['hybrid_score']-r['baseline_score'] for r in records])),'net_cluster95':cluster_ci(records,[r['hybrid_score']-r['baseline_score'] for r in records])}
 explicit=sum(r.get('explicit_licensed_direction',False) for r in records)
 out.update(explicit_heading_coverage=explicit/n,unknown_fraction_explicit_heading=1-explicit/n,unknown_fraction_no_candidate_support=1-k/n,conditional_score_interpretation='candidate-weighted compatibility; a hard directional success rate ONLY when exactly one licensed heading is emitted',conditional_halfplane_compatibility=float(np.mean([r['local_halfplane_score'] for r in covered])) if covered else None,baseline_halfplane_all=float(np.mean([r['baseline_halfplane_score'] for r in records])),hybrid_halfplane_all=float(np.mean([r['hybrid_halfplane_score'] for r in records])),net_halfplane_improvement=float(np.mean([r['hybrid_halfplane_score']-r['baseline_halfplane_score'] for r in records])),net_halfplane_cluster95=cluster_ci(records,[r['hybrid_halfplane_score']-r['baseline_halfplane_score'] for r in records]))
 # Reference/map cluster sensitivity addresses shared headings among nearby objects.
 if covered:
  refs=[{**r,'cluster':r['reference_group']} for r in covered];maps=[{**r,'cluster':r['map']} for r in covered]
  out['conditional_reference_cluster95']=cluster_ci(refs,[r['local_score'] for r in covered]);out['conditional_map_cluster95']=cluster_ci(maps,[r['local_score'] for r in covered])
 if vals and all(v in [0.,1.] for v in vals):out['wilson95_if_binary']=wilson(np.mean(vals),len(vals))
 return out

def run(out):
 REL=readlines(out/'relations.jsonl.gz');global_core=sum(r['semantic_word'] in MAIN for r in REL);X=readlines(out/'public_inputs.jsonl.gz');P=readlines(out/'inference_predictions.jsonl.gz');E=readlines(out/'evaluation_gt.jsonl.gz');ids=[x['occurrence_id'] for x in X];idx={r:i for i,r in enumerate(ids)};byX={x['occurrence_id']:x for x in X};byP={p['occurrence_id']:p for p in P};byE={e['occurrence_id']:e for e in E};B=np.load(out/'bert_features.npz');assert list(B['ids'])==ids
 F=B['features'];core=np.array([i for i,x in enumerate(X) if x['word'] in MAIN]);y=np.array([p['family_silver'] for p in P]);meta=np.array([[float(P[i]['reference_category']==c) for c in ['Building','TrafficRoad','Footpath','Ground','Water']]+[min(5,len(X[i]['mentioned']))/5] for i in range(len(X))],dtype=float);F=np.concatenate([F,meta],axis=1)
 # All supervision labels are text-derived silver categories, NEVER angle-best labels.
 router=[];folds=[];routing_quality=[];baseline={};roadside_baseline={};cohort_records=[]
 v2q=pd.read_csv(V2/'results/crossfit_predictions.csv')
 allA=transforms();seen=set();A=[]
 for a in allA:
  key=tuple(np.array(a['matrix']).ravel())
  if key not in seen:A.append(a);seen.add(key)
 def base_expect(e,a):return angle(np.array(a['matrix'])@IMAGE[e['word']])
 for scheme in ['object_5fold','leave_map_out']:
  splits=np.array([str(fold_id(x['cluster'])) if scheme=='object_5fold' else x['map'] for x in X]);groups=sorted(set(splits[core]));predictions={model:{} for model in ['bert_router','shuffled_label_router']}
  for fold in groups:
   te=np.array([i for i in core if splits[i]==fold]);tr=np.array([i for i in core if splits[i]!=fold]);test_texts={text_key(X[i]['instruction']) for i in te};purged=np.array([i for i in tr if text_key(X[i]['instruction']) not in test_texts]);assert not {X[i]['cluster'] for i in purged}&{X[i]['cluster'] for i in te}
   folds.append({'scheme':scheme,'fold':fold,'train_n':len(purged),'test_n':len(te),'purged_duplicate_English':len(tr)-len(purged),'train_clusters':sorted({X[i]['cluster'] for i in purged}),'test_clusters':sorted({X[i]['cluster'] for i in te})})
   for model in predictions:
    train_y=y[purged].copy()
    if model=='shuffled_label_router':np.random.default_rng(SEED+fold_id(fold)).shuffle(train_y)
    classes=np.unique(train_y)
    if len(classes)<2:pr=np.repeat(classes[0],len(te));probs=np.ones(len(te))
    else:
     clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=600,class_weight='balanced',random_state=SEED))
     with threadpool_limits(limits=1):clf.fit(F[purged],train_y);prob=clf.predict_proba(F[te])
     pr=clf.classes_[np.argmax(prob,axis=1)];probs=prob.max(1)
    for i,label,confidence in zip(te,pr,probs):predictions[model][ids[i]]={'label':str(label),'confidence':float(confidence)}
   # Geometric baseline uses GT only in globally shared dictionary calibration,
   # trained on OTHER groups; never assigns per-sample best local frames.
   fit=[E[i] for i in core if splits[i]!=fold and E[i]['geometry_status']=='disjoint']
   ranked=[]
   for a in A:
    errors=[float(adiff(e['angle_world'],base_expect(e,a))) for e in fit];ranked.append((np.mean(np.array(errors)<=45) if errors else -1,-np.mean(errors) if errors else -180,a))
   chosen=max(enumerate(ranked),key=lambda t:(t[1][0],t[1][1],-t[0]))[1][2]
   for i in te:baseline[scheme,ids[i]]=base_expect(E[i],chosen)
   sidefit=[E[i] for i in core if splits[i]!=fold and E[i]['roadside_eligible']]
   siderank=[]
   for aa in A:
    vals=[]
    for e in sidefit:
     expected=np.deg2rad(base_expect(e,aa));u=np.array([np.cos(expected),np.sin(expected)]);z=(np.array(e['target_polygon'])-np.array(e['roadside_center_xy']))@u;vals.append(min(z)>1)
    siderank.append((float(np.mean(vals)) if vals else -1,aa))
   sidechosen=max(enumerate(siderank),key=lambda z:(z[1][0],-z[0]))[1][1]
   for i in te:roadside_baseline[scheme,ids[i]]=base_expect(E[i],sidechosen)
  for model,preds in predictions.items():
   truth=[y[i] for i in core];guess=[preds[ids[i]]['label'] for i in core];routing_quality.append({'scheme':scheme,'model':model,'n':len(core),'silver_label_accuracy':float(accuracy_score(truth,guess)),'silver_label_macroF1':float(f1_score(truth,guess,labels=FAMILIES,average='macro',zero_division=0)),'warning':'agreement with text-rule silver labels, NOT verified author-frame accuracy'})
   for rid,z in preds.items():router.append({'scheme':scheme,'model':model,'occurrence_id':rid,'silver_label':byP[rid]['family_silver'],**z})
  # Evaluate identical policies across disjoint primary and separately eligible road-side cohorts.
  for cohort in ['v2_all','v2_high','expanded_all','expanded_high','expanded_excluding_known_boat','roadside_secondary']:
   eligible=[e for e in E if e['word'] in MAIN and (e['roadside_eligible'] if cohort=='roadside_secondary' else e['geometry_status']=='disjoint') and (not cohort.startswith('v2') or e['origin']=='v2_preserved') and (not cohort.endswith('high') or e['parse_tier']=='high') and (cohort!='expanded_excluding_known_boat' or byX[e['occurrence_id']].get('binding_rule')!='explicit_stand_at_facing_frame')]
   for e in eligible:
    rid=e['occurrence_id'];p=byP[rid];metric='roadside' if cohort=='roadside_secondary' else 'sector';a=roadside_baseline[scheme,rid] if cohort=='roadside_secondary' else baseline[scheme,rid]
    # Frozen v2 baseline comparison uses exactly v2 predictions on its original support.
    if cohort.startswith('v2'):
     vv=v2q[(v2q.subset==('high' if cohort.endswith('high') else 'all_effective'))&(v2q.scheme==('target_5fold' if scheme=='object_5fold' else 'leave_map_out'))&(v2q.occurrence_id==rid)];assert len(vv)==1;a=float(vv.iloc[0].expected_world_deg)
    baseline_err=float(adiff(e['angle_world'],a))
    if metric=='sector':bc=float(baseline_err<=45)
    else:
     aa=np.deg2rad(a);u=np.array([np.cos(aa),np.sin(aa)]);vals=(np.array(e['target_polygon'])-np.array(e['roadside_center_xy']))@u;bc=float(min(vals)>1)
    for model in ['strict_rules','semantic_candidates','semantic_no_road_facing_prior','all_geometry_candidates','bert_router','bert_support_uniform_orientation','shuffled_label_router']:
     if model=='strict_rules':chosen=p['strict_ids']
     elif model in {'semantic_candidates','semantic_no_road_facing_prior'}:
      chosen=routing_ids(p,p['family_silver'])
      if model=='semantic_no_road_facing_prior':chosen=[cid for cid in chosen if next(c for c in p['candidates'] if c['candidate_id']==cid)['type']!='landmark_to_mentioned_road']
     elif model=='all_geometry_candidates':chosen=[c['candidate_id'] for c in p['candidates']]
     else:
      pred=predictions['bert_router' if model=='bert_support_uniform_orientation' else model][rid];chosen=routing_ids(p,pred['label']) if pred['confidence']>=.60 else []
     weights=candidate_weights(p,chosen)
     if metric=='roadside':
      # Only frames of road-travel/road-axis type are meaningful for this secondary predicate.
      allowed={c['candidate_id'] for c in p['candidates'] if c['family']=='road_travel' or c['type']=='road_unsigned_axis'};weights=candidate_weights(p,[k for k in chosen if k in allowed])
     score,err,hard=local_score(p,weights,e,metric)
     half,_,_=local_score(p,weights,e,'halfplane') if metric!='roadside' else (score,None,hard)
     expected=np.deg2rad(a);v=np.array(e['subject_xy'])-np.array(e['reference_xy']);baseline_half=float(v@np.array([np.cos(expected),np.sin(expected)]))>1 if metric!='roadside' else bc
     explicit_licensed=bool(len(weights)==1 and next(c for c in p['candidates'] if c['candidate_id'] in weights)['level']=='explicit')
     if model=='bert_support_uniform_orientation' and weights:
      score=uniform_orientation_score(e,metric);err=90.;hard=False;half=uniform_orientation_score(e,'halfplane') if metric!='roadside' else score;explicit_licensed=False
     rec={'scheme':scheme,'cohort':cohort,'model':model,'occurrence_id':rid,'cluster':e['cluster'],'map':e['map'],'word':e['word'],'parse_tier':e['parse_tier'],'origin':e['origin'],'metric':'signed_roadside_halfplane' if metric=='roadside' else '90deg_sector','baseline_expected_deg':a,'baseline_score':bc,'baseline_error':baseline_err,'weights':weights,'local_score':score,'local_error':err,'single_candidate_direction':hard,'explicit_licensed_direction':explicit_licensed,'orientation_confirmed_by_author_pose':False,'uniform_orientation_control':model=='bert_support_uniform_orientation','local_halfplane_score':half,'baseline_halfplane_score':float(baseline_half),'hybrid_halfplane_score':half if half is not None else float(baseline_half),'reference_group':e['map']+':'+','.join(byX[rid]['reference_ids']),'hybrid_score':score if score is not None else bc,'hybrid_error':err if err is not None else baseline_err,'strict_state':p['strict_state'],'family_silver':p['family_silver']};cohort_records.append(rec)
 summary=[];grouped=collections.defaultdict(list)
 for r in cohort_records:grouped[(r['scheme'],r['cohort'],r['model'])].append(r)
 for key,rr in grouped.items():
  for mode in ['conditional_local','world_fallback_hybrid']:
   s=summarize(rr,local=mode=='conditional_local');s.update(scheme=key[0],cohort=key[1],model=key[2],mode=mode,metric=rr[0]['metric'],all_landmark_core_candidates=global_core,cohort_fraction_of_all_core=len(rr)/global_core,end_to_end_supported_fraction=s['covered']/global_core);summary.append(s)
 breakdown=[]
 for key,rr in grouped.items():
  for groupkey in ['word','map','parse_tier','family_silver']:
   sub=collections.defaultdict(list)
   for r in rr:sub[r[groupkey]].append(r)
   for val,q in sub.items():s=summarize(q,local=True);s.update(scheme=key[0],cohort=key[1],model=key[2],group_type=groupkey,group=val);breakdown.append(s)
 # Fixed-coverage geometry negative controls. No data-dependent randomization selection.
 null=[];rng=np.random.default_rng(SEED)
 for key,rr in grouped.items():
  if key[1]=='expanded_excluding_known_boat' or key[0]!='object_5fold' or key[2] not in {'strict_rules','semantic_candidates','bert_router'}:continue
  covered=[r for r in rr if r['local_score'] is not None]
  if not covered:continue
  obs=np.mean([r['local_score'] for r in covered]);rotated=[];permuted=[]
  for b in range(300):
   rotations=rng.uniform(0,360,len(covered));z=[]
   for r,rot in zip(covered,rotations):z.append(local_score(byP[r['occurrence_id']],r['weights'],byE[r['occurrence_id']],'roadside' if key[1]=='roadside_secondary' else 'sector',rotation=float(rot))[0])
   rotated.append(float(np.mean(z)))
   # Permute reference headings/labels AS A PACKAGE among eligible relations,
   # holding support and number/weights of hypotheses fixed.
   order=rng.permutation(len(covered));z=[]
   for j,r in enumerate(covered):
    donor=covered[order[j]];donorp=byP[donor['occurrence_id']];donorheads=[(c['heading_deg'],donor['weights'][c['candidate_id']]) for c in donorp['candidates'] if c['candidate_id'] in donor['weights']];e=byE[r['occurrence_id']];s=0.
    for h,w in donorheads:
     if key[1]=='roadside_secondary':
      a=np.deg2rad(h);left=np.array([-np.sin(a),np.cos(a)]);x=(np.array(e['target_polygon'])-np.array(e['roadside_center_xy']))@left;ok=min(x)>1 if e['word']=='left' else max(x)<-1
     else:ok=adiff(e['angle_world'],h+OFFSETS[e['word']])<=45
     s+=w*float(ok)
    z.append(s)
   permuted.append(float(np.mean(z)))
  for label,vals in [('independent_random_rotation',rotated),('permuted_reference_frame_packages',permuted)]:null.append({'scheme':key[0],'cohort':key[1],'model':key[2],'control':label,'covered':len(covered),'observed_conditional':float(obs),'null_mean':float(np.mean(vals)),'null95':np.quantile(vals,[.025,.975]).tolist(),'one_sided_MC_p':float((1+sum(v>=obs for v in vals))/(len(vals)+1)),'coverage_preserved':True,'B':len(vals),'caveat':'small-support and multiple-testing exploratory diagnostic, not causal proof'})
 writelines(out/'evaluation_predictions.jsonl.gz',cohort_records);pd.DataFrame([{k:v for k,v in r.items() if k!='weights'} for r in cohort_records]).to_csv(out/'evaluation_predictions.csv',index=False)
 writej(out/'summary.json',summary);pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);writej(out/'breakdown.json',breakdown);pd.DataFrame(breakdown).to_csv(out/'breakdown.csv',index=False)
 writej(out/'negative_controls.json',null);pd.DataFrame(null).to_csv(out/'negative_controls.csv',index=False);writej(out/'router_quality_silver.json',routing_quality);pd.DataFrame(router).to_csv(out/'router_predictions.csv',index=False);writelines(out/'folds.jsonl.gz',folds)
 # A deployment-facing output: frame type/state, candidate orientations and unknown;
 # route probabilities are not interpreted as calibrated author-frame confidence.
 publicout=[]
 predindex={(r['scheme'],r['model'],r['occurrence_id']):r for r in router}
 for inp,p in zip(X,P):
  q=predindex.get(('object_5fold','bert_router',inp['occurrence_id']));bert_ids=routing_ids(p,q['label']) if q and q['confidence']>=.60 else [];bert_weights=candidate_weights(p,bert_ids);bert_pool=[c for c in p['candidates'] if c['candidate_id'] in bert_weights];bert_clear=len(bert_pool)==1 and bert_pool[0]['level']=='explicit';publicout.append({'occurrence_id':inp['occurrence_id'],'subject_ids':inp['subject_ids'],'reference_ids':inp['reference_ids'],'word':inp['word'],'reference_frame_type_rule':p['family_silver'],'reference_frame_type_bert_crossfit':q['label'] if q else 'not_evaluated_noncore_word','bert_confidence_uncalibrated':q['confidence'] if q else None,'rule_state':p['strict_state'],'bert_state':'explicit_computable' if bert_clear else 'candidate_multiple' if bert_ids else 'unknown','bert_candidate_weights':bert_weights,'bert_heading_deg':bert_pool[0]['heading_deg'] if bert_clear else None,'bert_unknown_single_heading':not bert_clear,'rule_reference_heading_deg':next((c['heading_deg'] for c in p['candidates'] if c['candidate_id'] in p['strict_ids']),None),'candidate_frames':p['candidates'],'unknown_reference_heading':p['unknown_heading'],'GT_used':False})
 # Every named-landmark relation gets an output, even when binding is unknown.
 publicids={r['occurrence_id'] for r in publicout}
 for r in REL:
  if r['occurrence_id'] not in publicids:publicout.append({'occurrence_id':r['occurrence_id'],'subject_ids':r.get('subject_ids',[]),'reference_ids':r.get('reference_ids',[]),'reference_candidate_ids':r.get('reference_candidate_ids',[]),'word':r['semantic_word'],'rule_state':'unknown_binding_or_scope','bert_state':'unknown_binding_or_scope','rule_reference_heading_deg':None,'bert_heading_deg':None,'unknown_reference_heading':True,'candidate_frames':[],'reason':r['reason'],'GT_used':False})
 writelines(out/'reference_frame_outputs.jsonl.gz',publicout)
 writej(out/'frame_coverage.json',{'all_landmark_candidates':len(REL),'core_landmark_candidates':global_core,'bound_inputs':len(X),'core_bound_inputs':len(core),'full_relation_outputs':len(publicout),'rule_single_heading_all_bound':sum(bool(p['strict_ids']) for p in P),'rule_candidate_or_unknown_all_bound':sum(not bool(p['strict_ids']) for p in P),'unbound_unknown':len(REL)-len(X),'rule_levels':dict(collections.Counter(p['strict_state'] for p in P))})
 # Refit a small reusable router on all TRAIN silver labels; NOT used for CV scores.
 full=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=600,class_weight='balanced',random_state=SEED))
 with threadpool_limits(limits=1):full.fit(F[core],y[core])
 scaler=full.steps[0][1];lr=full.steps[1][1]
 np.savez_compressed(out/'silver_router_full_train.npz',mean=scaler.mean_,scale=scaler.scale_,coef=lr.coef_,intercept=lr.intercept_,classes=lr.classes_)
 writej(out/'silver_router_model.json',{'supervision':'text-derived silver family labels only','not_used_for_reported_crossfit_scores':True,'features':'frozen BERT 768 + reference category 5 one-hot + mention count /5','input_dim':int(F.shape[1]),'probability_gate':.60,'heading_inference':'independent public geometry only; multi-candidate signs never picked by target GT','weight_file_sha256':sha(out/'silver_router_full_train.npz')})
 writej(out/'evaluation_manifest.json',{'public_input_sha256':sha(out/'public_inputs.jsonl.gz'),'frozen_inference_sha256':sha(out/'inference_predictions.jsonl.gz'),'BERT_features_sha256':sha(out/'bert_features.npz'),'router_training_labels':'text-rule silver family only; NOT target-optimal orientations','BERT_fine_tuned':False,'router':'StandardScaler + LogisticRegression C=1 class_weight=balanced, 600 iterations; probability gate .60 fixed before evaluation','split_schemes':['subject_object_5fold','leave_map_out'],'raw_English_duplicate_train_rows_purged':True,'GT_used_for_local_selection':False,'GT_used_for_baseline':'shared global D4 dictionary fitted on OTHER groups only; v2 cohorts reuse exact frozen v2 baselines; roadside secondary independently fits its global world signed-halfplane dictionary on matching roadside training support','null_controls_B':300,'seed':SEED})
 print(routing_quality,flush=True);print('evaluation rows',len(cohort_records),flush=True)
if __name__=='__main__':run(BASE/'results')
