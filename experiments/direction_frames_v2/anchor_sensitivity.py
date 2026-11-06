"""Change geometric anchor only, holding text binding and crossfit dictionary fixed."""
import gzip,json
from pathlib import Path
import numpy as np,pandas as pd
from analyze_frames import cluster_ci
from geometry_frames import adiff
from extract import dump
P=Path(__file__).parent/'results';pairs=[json.loads(x) for x in gzip.open(P/'geometry_pairs.jsonl.gz','rt')];by={p['occurrence_id']:p for p in pairs};Q=pd.read_csv(P/'crossfit_predictions.csv');Q=Q[Q.scheme=='target_5fold'];rows=[]
for (subset,w),sub in Q.groupby(['subset','word']):
 for key in ['angle_world','polygon_centroid_angle','nearest_reference_boundary_angle']:
  valid=[(r,by[r['occurrence_id']].get(key)) for r in sub.to_dict('records') if by[r['occurrence_id']].get(key) is not None]
  g=pd.DataFrame([r for r,a in valid]);a=np.array([a for r,a in valid]);err=adiff(a,g.expected_world_deg.values);ok=err<=45
  rows.append({'subset':subset,'word':w,'anchor':key,'n':len(g),'consistency':float(ok.mean()),'cluster95':cluster_ci(g,ok),'mean_abs_error_deg':float(err.mean()),'binding_changed':False,'dictionary_refitted':False,'interpretation':'geometric displacement sensitivity only; not a new independently confirmed frame'})
dump(P/'anchor_sensitivity.json',rows);pd.DataFrame(rows).to_csv(P/'anchor_sensitivity.csv',index=False)
print('anchor sensitivity complete')
