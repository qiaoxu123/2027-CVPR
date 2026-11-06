"""Paired map bootstrap and proxy scale/world audit; no threshold selection."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.yolo_world_proxy_diagnostics import center_association
from graphnav.landmark_projection import frame_to_world

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);rows=[json.load(open(f)) for f in (Path(a.run)/'results').glob('*.json')];maps=sorted(set(r['map_name'] for r in rows));result={};rng=np.random.default_rng(42)
 for bank in ['base','color']:
  seq=[r for r in rows if r['bank']==bank];counts={};small={};ratios=[];ne=[];unique={};pair=[]
  for method in ['whole','tile']:
   matched_ids=set();obs_ids=set();smalln=smallhit=0
   for r in seq:
    z=center_association(r['methods'][method]['detections'],r['proxies'],'vehicle');counts[(r['map_name'],r['sample_id'],method)]=(z['associated'],z['proxies'])
    for i,g in enumerate(r['proxies']):
     if g['category']=='vehicle' and g['eligible']:obs_ids.add((r['map_name'],g['object_id']))
    smallids={i for i,g in enumerate(r['proxies']) if g['category']=='vehicle' and g['eligible'] and (g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1])*(640/r['width'])**2<32**2};smalln+=len(smallids);smallhit+=sum(g in smallids for d,g in z['pairs'])
    for d,g in z['pairs']:
     matched_ids.add((r['map_name'],r['proxies'][g]['object_id']))
     px=r['proxies'][g]['box'];center=np.array([[(px[0]+px[2])/2,(px[1]+px[3])/2]]);wc=frame_to_world(center,r['footprint'],(r['height'],r['width']))[0];dc=r['methods'][method]['detections'][d]['world_center'];ne.append((method,float(np.linalg.norm(wc-dc))))
   small[method]=dict(proxies=smalln,center_associated=smallhit,recall=smallhit/smalln if smalln else None);unique[method]=dict(visible_supported_unique=len(obs_ids),center_associated_unique=len(matched_ids))
  boot=[]
  for _ in range(2000):
   selected=rng.choice(maps,len(maps),replace=True);v={}
   for method in ['whole','tile']:
    q=[x for mp in selected for (map_,sid,m),x in counts.items() if map_==mp and m==method];v[method]=sum(x[0] for x in q)/sum(x[1] for x in q)
   boot.append(v['tile']-v['whole'])
  for r in seq:
   w=counts[(r['map_name'],r['sample_id'],'whole')][0];t=counts[(r['map_name'],r['sample_id'],'tile')][0];pair.append(t-w)
  result[bank]=dict(center_recall_delta_ci95=np.percentile(boot,[2.5,97.5]).tolist(),small_vehicle=small,unique_instances=unique,frames_more=sum(x>0 for x in pair),frames_equal=sum(x==0 for x in pair),frames_less=sum(x<0 for x in pair),world_proxy_center_distance={m:dict(mean=float(np.mean([n for mm,n in ne if mm==m])),median=float(np.median([n for mm,n in ne if mm==m]))) for m in ['whole','tile']})
 base=[r for r in rows if r['bank']=='base'];result['counts']=dict(unique_registered_objects=len({(r['map_name'],g['object_id']) for r in base for g in r['proxies']}),supported_unique_vehicle=len({(r['map_name'],g['object_id']) for r in base for g in r['proxies'] if g['category']=='vehicle' and g['eligible']}),full_supported=sum(g['coverage']>=.99 and g['eligible'] for r in base for g in r['proxies']),partial_supported=sum(g['coverage']<.99 and g['eligible'] for r in base for g in r['proxies']))
 result['tiles_total']=sum(len(r['methods']['tile']['tiles']) for r in base)
 result['sampling_ratio_range']=[min(r['render_sampling_ratio'] for r in base),max(r['render_sampling_ratio'] for r in base)]
 result['wall_seconds_config_to_last_result']=max(x.stat().st_mtime for x in (Path(a.run)/'results').glob('*.json'))-(Path(a.run)/'config.json').stat().st_mtime
 (root/'extended_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
