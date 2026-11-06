"""Version-pinned v4 utilities; new files never mutate v4 artifacts or thresholds."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;V4=HERE.parent/'direction_frames_v4'
sys.path.insert(0,str(V4))
from shared import *
RESULT=HERE/'results'
from infer import region,anchor
from roads import directed_field
from frames import derive
from evaluate import field_geometry,boundary_samples,wilson
import pandas as pd

def savecsv(name,rows):writej(RESULT/(name+'.json'),rows);pd.DataFrame(rows).to_csv(RESULT/(name+'.csv'),index=False)
def stats(rr):
 n=len(rr);u=[r for r in rr if r['state']=='unique_field'];s=[r for r in u if r['result'] in {'consistent','conflict'}];k=sum(r['result']=='consistent' for r in s);vals=[float(r['result']=='consistent') for r in s]
 return {'n':n,'bound':sum(r['bound'] for r in rr),'unique':len(u),'multiple':sum(r['state']=='multiple' for r in rr),'unknown':sum(r['state']=='unknown' for r in rr),'unique_coverage':len(u)/n if n else None,'scorable':len(s),'geometry_determinacy':len(s)/len(u) if u else None,'consistent':k,'conflict':len(s)-k,'ambiguous_unique':len(u)-len(s),'conditional_consistency':k/len(s) if s else None,'wilson95':wilson(k,len(s)),'object_cluster95':cluster_ci(s,vals),'road_cluster95':cluster_ci([{**r,'cluster':r['road_group']} for r in s],vals),'map_cluster95':cluster_ci([{**r,'cluster':r['map']} for r in s],vals),'confirmed_all':k/n if n else None,'confirmed_unique':k/len(u) if u else None,'independent_subjects':len({r['cluster'] for r in s}),'independent_roads':len({r['road_group'] for r in s}),'independent_maps':len({r['map'] for r in s})}
