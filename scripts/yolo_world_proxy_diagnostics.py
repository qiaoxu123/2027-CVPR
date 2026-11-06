"""Posthoc proxy audit: coarse map contours cannot serve as tight vehicle boxes."""
import json,argparse
from pathlib import Path
from collections import Counter
import numpy as np
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from scipy.optimize import linear_sum_assignment
from scripts.yolo_world_citynav import family

def center_association(dets,proxies,category):
    ds=[(i,d) for i,d in enumerate(dets) if family(d['label'])==category]
    gs=[(i,g) for i,g in enumerate(proxies) if g['eligible'] and g['category']==category]
    hit=np.zeros((len(ds),len(gs)),bool)
    for j,(_,g) in enumerate(gs):
        poly=unary_union([Polygon(q) for q in g['contours']])
        for i,(_,d) in enumerate(ds):
            x,y,x2,y2=d['box'];hit[i,j]=poly.covers(Point((x+x2)/2,(y+y2)/2))
    pairs=[]
    if hit.size:
        rr,cc=linear_sum_assignment(-hit.astype(float))
        pairs=[(ds[i][0],gs[j][0]) for i,j in zip(rr,cc) if hit[i,j]]
    ratios=[(proxies[g]['box'][2]-proxies[g]['box'][0])*(proxies[g]['box'][3]-proxies[g]['box'][1])/max((dets[d]['box'][2]-dets[d]['box'][0])*(dets[d]['box'][3]-dets[d]['box'][1]),1e-9) for d,g in pairs]
    return dict(detections=len(ds),proxies=len(gs),associated=len(pairs),pairs=pairs,proxy_to_detection_bbox_area_ratios=ratios)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);rows=[json.load(open(f)) for f in (Path(a.run)/'results').glob('*.json')]
    result={};per=[]
    for r in rows:
        for m,v in r['methods'].items():
            for c in ['vehicle','parking','building','road']:
                z=center_association(v['detections'],r['proxies'],c);per.append(dict(sample_id=r['sample_id'],bank=r['bank'],method=m,category=c,**z))
    for b in ['base','color','parking']:
        for m in ['whole','tile']:
            for c in ['vehicle','parking','building','road']:
                seq=[r for r in per if r['bank']==b and r['method']==m and r['category']==c];n=sum(r['detections'] for r in seq);g=sum(r['proxies'] for r in seq);h=sum(r['associated'] for r in seq);rat=[v for r in seq for v in r['proxy_to_detection_bbox_area_ratios']]
                result[f'{b}/{m}/{c}']=dict(detections=n,proxies=g,center_associated=h,center_proxy_recall=h/g if g else None,center_proxy_precision=h/n if n else None,median_proxy_to_detection_bbox_area_ratio=float(np.median(rat)) if rat else None)
    result['label_counts']={f'{b}/{m}':dict(Counter(d['label'] for r in rows if r['bank']==b for d in r['methods'][m]['detections'])) for b in ['base','color','parking'] for m in ['whole','tile']}
    (out/'center_proxy_diagnostics.json').write_text(json.dumps(result,indent=2));(out/'center_proxy_per_image.json').write_text(json.dumps(per,separators=(',',':')))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
