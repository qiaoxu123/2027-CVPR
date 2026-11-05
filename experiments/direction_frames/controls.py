"""Inference controls, boundary confidence intervals and source-independent metrics."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from analyze import summarize,cluster_ci,wilson,distance,dump
p=Path(__file__).parent/'results'
r=json.loads((p/'occurrences.json').read_text());d=pd.DataFrame([x for x in r if x['status']=='valid'])
rows=[]
for w,sub in d.groupby('semantic_word'):
 for frame,col in [('world','angle_world'),('initial_yaw','angle_yaw'),('start_landmark_LOS','angle_los')]:
  q=sub[sub[col].notna()];z=summarize(q,col);z.update(word=w,frame=frame,interpretation='fitted_axis_descriptive_concentration_only');rows.append(z)
dump(p/'frame_concentration.json',rows);pd.DataFrame(rows).to_csv(p/'frame_concentration.csv',index=False)
b=pd.read_csv(p/'polygon_checks.csv');z=[]
for (w,h,t),sub in b.groupby(['word','hypothesis','tolerance_m']):
 usable=sub[sub.boundary_class!='ambiguous'];correct=usable.boundary_class=='consistent'
 clusters=d.set_index('occurrence_id').loc[usable.occurrence_id,'cluster'].values
 ci=cluster_ci(pd.DataFrame({'cluster':clusters}),correct.values) if len(usable) else [None,None]
 z.append({'word':w,'hypothesis':h,'tolerance_m':t,'all_disjoint_pairs':len(sub),'boundary_evaluable_n':len(usable),'ambiguous_n':len(sub)-len(usable),'coverage':len(usable)/len(sub),'consistent_n':int(correct.sum()),'consistency_if_evaluable':float(correct.mean()) if len(usable) else None,'wilson95':wilson(int(correct.sum()),len(usable)),'cluster95':ci})
dump(p/'polygon_hypotheses.json',z);pd.DataFrame(z).to_csv(p/'polygon_hypotheses.csv',index=False)
# Non-cardinal concentration: best arbitrary +/-45-degree sector is also not assumed.
free=[]
for w,q in d.groupby('semantic_word'):
 ang=q.angle_world.values;centers=np.unique(np.concatenate([ang-45,ang+45,ang]))%360
 counts=np.array([(distance(ang,a)<=45+1e-8).sum() for a in centers]);idx=np.argmax(counts)
 free.append({'word':w,'n':len(q),'max_arbitrary_90deg_sector_fraction':float(counts[idx]/len(q)),'fitted_center_deg':float(centers[idx]),'interpretation':'upper bound chosen on same TRAIN sample; optimistic, not independent proof'})
dump(p/'free_axis_upper_bound.json',free)
# Flatten exclusions for transparent machine aggregation.
exc=pd.DataFrame([{'word':x['word'],'reason':x['reason'],'scope':x['scope']} for x in r if x['status']!='valid']).groupby(['word','reason','scope']).size().reset_index(name='n');exc.to_csv(p/'exclusion_reasons.csv',index=False)
# Unique spatial pairs and unique text sensitivity (counting text duplicates separately otherwise).
ss=[]
for name,q in [('occurrence',d),('unique_pair_word',d.drop_duplicates(['map','target_id','ref_id','semantic_word'])),('unique_english_word',d.drop_duplicates(['instruction','semantic_word']))]:
 for w,g in q.groupby('semantic_word'):
  z=summarize(g,'angle_world');z.update(word=w,weighting=name);ss.append(z)
dump(p/'weighting_sensitivity.json',ss)
print('controls complete')
