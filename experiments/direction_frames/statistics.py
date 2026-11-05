"""Additional TRAIN-only controls; does not change extraction or choose geometry."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import WORDS,OFFSETS,AXES,distance,summarize,cluster_ci,dump,polygon,SEED

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path(__file__).parent/'results');args=ap.parse_args();out=args.out
 R=json.loads((out/'occurrences.json').read_text());valid=[r for r in R if r['status']=='valid'];df=pd.DataFrame(valid)
 eps=json.loads((args.data_root/'original_citynav/citynav_train_seen.json').read_text());O=json.loads((args.data_root/'cityrefer_original/objects.json').read_text())
 targets={};ep_records=[]
 for i,e in enumerate(eps):
  m=f"{e['area']}_block_{e['block']}";oid=str(e['object_ids'][0]);o=O[m][oid];key=f'{m}:{oid}';targets[key]=o
  p=e['trajectory'][0];ep_records.append({'episode':i,'map':m,'target_id':oid,'ann_id':e['ann_ids'][0],'instruction':e['descriptions'][0],'gt_xyz':e['target_positions'][0],'target_polygon_key':key,'initial_pose':p})
 dump(out/'target_catalog_train.json',targets);pd.DataFrame(ep_records).to_csv(out/'episodes_train.csv',index=False)
 maps=sorted({x['map'] for x in ep_records})
 dump(out/'reference_catalog_train.json',{f'{m}:{i}':o for m in maps for i,o in O[m].items() if o['name']})
 assets=[]
 for m in maps:
  for ext in ['png','tif']:
   f=args.data_root/'rgbd'/f'{m}.{ext}';h=hashlib.sha256()
   with f.open('rb') as fp:
    for block in iter(lambda:fp.read(1<<20),b''):h.update(block)
   assets.append({'map':m,'path':str(f),'sha256':h.hexdigest(),'size_bytes':f.stat().st_size})
 dump(out/'raster_manifest.json',assets)
 # Record raw and normalized sentence agreement separately.
 mism=json.loads((out/'annotation_mismatches.json').read_text());audit=[]
 for z in mism:
  e=eps[z['episode']];o=O[z['map']][z['target_id']];txt=z['text'];a=z['ann_id'];n=lambda s:' '.join(s.lower().split())
  audit.append(dict(z,annotation_at_index=o['descriptions'][a] if a<len(o['descriptions']) else None,exact_match_elsewhere=[j for j,t in enumerate(o['descriptions']) if t==txt],normalized_match_at_index=a<len(o['descriptions']) and n(txt)==n(o['descriptions'][a])))
 dump(out/'annotation_alignment_audit.json',audit)
 # Full accepted-set text audit, independently inspectable; no human claim.
 pd.DataFrame([{'occurrence_id':r['occurrence_id'],'instruction':r['instruction'],'word':r['word'],'reference':r['ref_name'],'binding_rule':r['binding_rule'],'decision':'accepted_conservative_direct','auditor':'assistant_text_scope_check_not_human_or_independent_reviewer'} for r in valid]).to_csv(out/'accepted_scope_audit.csv',index=False)
 # The exact same spatial pair can carry opposite words across annotations.
 anti={'left':'right','right':'left','front':'behind','behind':'front','above':'below','below':'above','north':'south','south':'north','east':'west','west':'east','back':'front'}
 bypair=collections.defaultdict(list)
 for r in valid:bypair[(r['map'],r['target_id'],r['ref_id'])].append(r)
 contradictions=[]
 for key,rs in bypair.items():
  for i,r in enumerate(rs):
   for q in rs[i+1:]:
    if anti.get(r['semantic_word'])==q['semantic_word'] or anti.get(q['semantic_word'])==r['semantic_word']:
     contradictions.append({'map':key[0],'target_id':key[1],'ref_id':key[2],'ref_name':r['ref_name'],'first':r['occurrence_id'],'second':q['occurrence_id'],'word1':r['semantic_word'],'word2':q['semantic_word'],'angle1':r['angle_world'],'angle2':q['angle_world'],'instruction1':r['instruction'],'instruction2':q['instruction'],'polygon_distance_m':r['polygon_distance']})
 dump(out/'same_pair_opposite_descriptions.json',contradictions)
 # Error distribution under whole-corpus A axes; model comparison is on matched support.
 common=[];predictions=[]
 for w,sub in df.groupby('semantic_word'):
  if w not in OFFSETS:continue
  sub=sub[sub.angle_yaw.notna()&sub.angle_los.notna()].copy()
  folds=sub.cluster.map(lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%5)
  ac=np.zeros(len(sub),dtype=bool);ae=np.zeros(len(sub));eligible=np.zeros(len(sub),dtype=bool)
  for fold in range(5):
   fit=sub[folds!=fold];idx=np.where((folds==fold).values)[0]
   if len(fit) and len(idx):
    a=summarize(fit,'angle_world')['axis_deg'];err=distance(sub.iloc[idx].angle_world.values,a);ac[idx]=err<=45;ae[idx]=err;eligible[idx]=True
  sub=sub.iloc[np.where(eligible)[0]];ac=ac[eligible];ae=ae[eligible]
  if not len(sub):continue
  bc=distance(sub.angle_yaw.values,OFFSETS[w])<=45;cc=distance(sub.angle_los.values,OFFSETS[w])<=45
  for h,c in [('A_world_crossfit',ac),('B_initial_yaw',bc),('C_start_landmark_LOS',cc)]:
   base={'word':w,'hypothesis':h,'n_common':len(sub),'clusters':sub.cluster.nunique(),'consistency':float(c.mean()),'cluster95':cluster_ci(sub,c),'paired_improvement_over_A':float((c.astype(float)-ac).mean()),'paired_improvement95':cluster_ci(sub,c.astype(float)-ac)};common.append(base)
  for j,r in enumerate(sub.to_dict('records')):
   predictions.append({'occurrence_id':r['occurrence_id'],'word':w,'A_crossfit_correct':bool(ac[j]),'A_crossfit_error':ae[j],'B_correct':bool(bc[j]),'C_correct':bool(cc[j])})
 dump(out/'matched_hypotheses.json',common);pd.DataFrame(common).to_csv(out/'matched_hypotheses.csv',index=False);pd.DataFrame(predictions).to_csv(out/'crossfit_predictions.csv',index=False)
 # Per-map leave-one-map-out generalization of a learned word axis.
 loco=[]
 for (w,m),sub in df.groupby(['semantic_word','map']):
  fit=df[(df.semantic_word==w)&(df['map']!=m)]
  if len(fit):
   a=summarize(fit,'angle_world')['axis_deg'];z=summarize(sub,'angle_world',a);z.update(word=w,map=m,fit_n=len(fit));loco.append(z)
 dump(out/'leave_map_out.json',loco)
 # Height semantics are not inferred from object center Z sign alone.
 z=[]
 for w in ['above','below','top','bottom']:
  sub=[r for r in valid if r['semantic_word']==w]
  z.append({'word':w,'n':len(sub),'dz_values_m':[r['dz'] for r in sub], 'positive_dz_gt_1m':sum(r['dz']>1 for r in sub),'negative_dz_lt_minus_1m':sum(r['dz']< -1 for r in sub),'within_1m':sum(abs(r['dz'])<=1 for r in sub),'interpretation':'object center Z is not ground height, target position is not height-above-ground'})
 dump(out/'height_semantics.json',z)
 # Tolerance sensitivity, including geometry exclusion (not just projection).
 sensitivity=[]
 for tol in [.5,1.,2.]:
  for w in WORDS:
   rs=[r for r in R if r.get('semantic_word',r['word'])==w and 'intersection_area' in r and r['intersection_area']==0 and r['polygon_distance']>tol and r['distance_m']>tol]
   sensitivity.append({'word':w,'tolerance_m':tol,'n_disjoint_significant':len(rs)})
 pd.DataFrame(sensitivity).to_csv(out/'geometry_tolerance_sensitivity.csv',index=False)
 # Frame identifiability and parsing coverage have different denominators.
 corpus=json.loads((out/'corpus.json').read_text());corpus.update(valid_coverage_occurrences=len(valid)/len(R),unresolved_geometry_or_binding_fraction=1-len(valid)/len(R),valid_direction_episodes=len({r['episode'] for r in valid}),missing_author_camera_pose_among_valid=len(valid),opposite_description_pairs=len(contradictions))
 dump(out/'corpus.json',corpus)
 print('same-pair contradictions:',len(contradictions));print('matched comparisons',len(common))
if __name__=='__main__':main()
