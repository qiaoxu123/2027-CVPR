"""Post-freeze descriptive width/curvature strata, NEVER predictor tuning."""
from v5core import *
from shapely.geometry import Point

def run():
 X=readlines(RESULT/'public_inputs.jsonl.gz');F=readlines(RESULT/'road_fields.jsonl.gz');meta=[]
 for f in F:
  if not f.get('points_xy'):continue
  x=next(x for x in X if x['map']==f['key'].split(':')[0] and all(i in x['scene'] for i in f['road_ids']));p=region(x,f['road_ids']);xy=np.array(f['points_xy']);d=np.array(f['distance_from_source_m']);interior=(d>5)&(d<max(d)-5);width=[2*p.boundary.distance(Point(*v)) for v,ok in zip(xy,interior) if ok and p.covers(Point(*v))];h=np.array([np.nan if z is None else z for z in f['heading_deg']]);bends=[]
  for s,hh in zip(d,h):
   window=h[np.abs(d-s)<=5];window=window[np.isfinite(window)]
   if len(window) and np.isfinite(hh):bends.append(float(np.max(adiff(window,hh))))
  ratio=float(np.quantile(width,.9)/max(np.quantile(width,.1),1e-6)) if width else None;meta.append({'key':f['key'],'state':f['state'],'map':x['map'],'clearance_diameter_p10_m':float(np.quantile(width,.1)) if width else None,'clearance_diameter_p90_m':float(np.quantile(width,.9)) if width else None,'interior_width_proxy_ratio':ratio,'width_stratum':'unknown' if ratio is None else 'ratio_le_1.5' if ratio<=1.5 else 'ratio_1.5_to_2' if ratio<=2 else 'ratio_gt_2','bend_proxy_p90_deg_5m':float(np.quantile(bends,.9)) if bends else None,'bend_stratum':'low_bend_le15deg' if bends and np.quantile(bends,.9)<=15 else 'higher_bend_or_unavailable','branch_nodes':f['branch_nodes'],'endpoints':f['endpoints'],'component_fraction':f['component_fraction'],'definition':'2x polygon-boundary clearance, excluding end5m; NOT surveyed lane width','GT_used_for_feature':False,'subgroups_postfreeze_descriptive_only':True})
 savecsv('road_geometry_features',meta);by={m['key']:m for m in meta};Q=readlines(RESULT/'geometry_checks.jsonl.gz');rows=[]
 for key in ['width_stratum','bend_stratum']:
  values=sorted({m[key] for m in meta})
  for val in values:
   z=[r for r in Q if r['state']=='unique_field' and by[r['road_field_key']][key]==val];rows.append({'group_type':key,'group':val,**stats(z),'causal_error_attribution':False,'postfreeze_descriptive':True})
 savecsv('geometry_strata',rows);print('Geometry descriptors',len(meta))
if __name__=='__main__':run()
