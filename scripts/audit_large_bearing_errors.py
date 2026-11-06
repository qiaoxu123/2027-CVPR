"""Posthoc >90-degree audit of sealed TRAIN confirmation. No inference changes."""
import argparse,json,collections,re,html,unicodedata
from pathlib import Path
import numpy as np
import torch
from shapely.geometry import Polygon,Point
from shapely.ops import nearest_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.relation_bearing_strata import anchor_strata

def canonical(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())
def angle(a,b):return np.degrees(np.arctan2(*(b-a)[::-1]))%360

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--public-objects',type=Path,required=True);a=p.parse_args();out=Path('reports/relation_bearing/large_errors');out.mkdir(exist_ok=True)
 data=json.load(open(a.manifest))['records'];catalog=json.load(open(a.public_objects));parsed=[json.loads(s) for s in (a.run/'data/parsed.jsonl').read_text().splitlines()];pred=[json.loads(s) for s in (a.run/'confirmation/relations_angle_rows.jsonl').read_text().splitlines()];features=torch.load(a.run/'public_inputs.pt',map_location='cpu',weights_only=True);details=[];names={}
 for m,objs in catalog.items():
  d=collections.defaultdict(list)
  for oid,obj in objs.items():
   if obj.get('name'):d[canonical(obj['name'])].append(str(oid))
  names[m]=d
 for x in pred:
  i=x['index'];row=data[i];m=row['map_name'];ref=row['landmarks'][x['anchor']];obj=catalog[m][str(ref['id'])];poly=Polygon(obj['contour']);pts=np.array(obj['contour']);anchor=pts.mean(0);b=np.array(row['world_bounds']);gt=b[:2]+np.array(row['target_xy'])*(b[2:]-b[:2]);v=np.array(poly.centroid.coords[0]);gtangle=angle(anchor,gt);change=abs((angle(v,gt)-gtangle+180)%360-180);scene=features['scenes'][features['queries'][i]['scene']];j=x.get('object_pred');oid=scene['node_ids'][j] if j is not None and 0<=j<len(scene['node_ids']) else None;group=anchor_strata(parsed[i],ref);extent=np.ptp(pts,axis=0);targetid=row['episode_id'].split(':')[1];targetobj=catalog[m].get(targetid);targetpoly=Polygon(targetobj['contour']) if targetobj else None
  normalized=(pts-b[:2])/(b[2:]-b[:2]);stored=np.array(ref['contour']);binding_delta=float(np.max(np.abs(normalized-stored))) if stored.shape==normalized.shape else None
  details.append(dict(**x,instruction=row['instruction'],episode_id=row['episode_id'],map_name=m,anchor_id=ref['id'],anchor_name=ref['name'],anchor_type=obj.get('object_type'),anchor_polygon=pts.tolist(),anchor_vertex_mean=anchor.tolist(),anchor_area_m2=poly.area,anchor_extent_m=extent.tolist(),anchor_valid=poly.is_valid,anchor_centroid=v.tolist(),centroid_vs_vertex_bearing_difference_deg=change,GT_world_posthoc=gt.tolist(),GT_bearing_deg=gtangle,anchor_to_GT_m=float(np.linalg.norm(gt-anchor)),GT_to_anchor_polygon_m=poly.distance(Point(gt)),GT_inside_anchor=poly.covers(Point(gt)),same_name_public_ids=names[m].get(canonical(ref['name']),[]),same_name_ref_count=sum(canonical(z['name'])==canonical(ref['name']) for z in row['landmarks']),bound_normalized_contour_max_error=binding_delta,relation_group=group['group'],relative_reference_unresolved=group['unresolved_relative_frame'],selected_public_id=oid,selected_public_name=catalog[m].get(str(oid),{}).get('name'),target_public_id_posthoc=targetid,target_in_public_context=x['target_index_offline']>=0,target_ID_GT_distance_posthoc=targetpoly.distance(Point(gt)) if targetpoly else None,paths=[z for z in parsed[i]['paths'] if str(ref['id']) in z['public_ids']]))
 bad=[x for x in details if x['error']>90];good=[x for x in details if x['error']<=90]
 def counts(rr):return dict(anchor_pairs=len(rr),queries=len(set(x['index'] for x in rr)),physical_targets=len(set(x['physical_id'] for x in rr)))
 def table(key,fun=lambda x:x):
  t=collections.defaultdict(lambda:dict(total=0,bad=0))
  for x in details:k=str(fun(x[key]));t[k]['total']+=1;t[k]['bad']+=int(x['error']>90)
  return {k:{**v,'bad_rate':v['bad']/v['total']} for k,v in t.items()}
 stats=dict(total=counts(details),large_error=counts(bad),bad_rate=len(bad)/len(details),by_map=table('map_name'),by_type=table('anchor_type'),by_relation=table('relation_group'),by_relative_frame=table('relative_reference_unresolved'),by_target_context=table('target_in_public_context'),by_duplicate_name=table('same_name_public_ids',lambda x:len(x)>1),by_large_anchor=table('anchor_extent_m',lambda x:max(x)>100),by_anchor_distance=table('anchor_to_GT_m',lambda d:'<20m' if d<20 else '20-100m' if d<100 else '>=100m'),by_centroid_instability=table('centroid_vs_vertex_bearing_difference_deg',lambda x:x>45),GT_inside_anchor_bad=sum(x['GT_inside_anchor'] for x in bad),GT_inside_anchor_total=sum(x['GT_inside_anchor'] for x in details),binding_contour_error_max=max(x['bound_normalized_contour_max_error'] or 0 for x in details),invalid_anchor_total=sum(not x['anchor_valid'] for x in details))
 (out/'summary.json').write_text(json.dumps(stats,indent=2));(out/'all_large_errors.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in bad))
 # Diverse actual cases; no duplicate physical-target/anchor, no handpicked improvements.
 ids=[12810,14112,10272,12583,13299,10031,13714,13716,10056];chosen=[];seen=set()
 for i in ids:
  options=[x for x in bad if x['index']==i]
  if options:
   x=max(options,key=lambda z:z['error']);key=(x['physical_id'],x['anchor_id'])
   if key not in seen:chosen.append(x);seen.add(key)
 for x in sorted(bad,key=lambda x:-x['error']):
  key=(x['physical_id'],x['anchor_id'])
  if key in seen or x['anchor_type'] not in ['Building','TrafficRoad','Footpath']:continue
  chosen.append(x);seen.add(key)
  if len(chosen)>=12:break
 cards=[]
 for j,x in enumerate(chosen):
  m=x['map_name'];gt=np.array(x['GT_world_posthoc']);anchor=np.array(x['anchor_vertex_mean']);poly=np.array(x['anchor_polygon']);objects=catalog[m];fig,axes=plt.subplots(1,2,figsize=(12,5));ax=axes[0]
  for ob in objects.values():
   z=np.array(ob.get('contour',[]))
   if z.ndim==2 and len(z)>2 and np.linalg.norm(z.mean(0)-anchor)<300:ax.plot(*np.vstack([z,z[0]]).T,color='silver',lw=.5)
  ax.fill(*poly.T,color='orange',alpha=.5,label='Bound public Polygon');ax.scatter(*anchor,c='orange',s=40,label='Anchor vertex mean');ax.scatter(*gt,c='green',marker='*',s=120,label='TRAIN GT: posthoc only');v=np.array(x['anchor_centroid']);ax.scatter(*v,c='black',marker='x',label='Area centroid: diagnostic only')
  if x['selected_public_id'] in objects:
   z=np.array(objects[x['selected_public_id']]['contour']);ax.plot(*np.vstack([z,z[0]]).T,color='purple',lw=2,label='Latent selected public object')
  for theta,color,label in [(x['GT_bearing_deg'],'green','GT bearing'),(x['pred']*45,'red','Predicted 8-bin bearing')]:
   t=np.radians(theta);length=max(40,min(120,x['anchor_to_GT_m']));ax.arrow(*anchor,length*np.cos(t),length*np.sin(t),color=color,width=.8,label=label)
  span=max(70,np.max(abs(np.r_[poly-gt,anchor[None]-gt]))+20);center=(anchor+gt)/2;ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_aspect('equal');ax.legend(fontsize=6);ax.set(xlabel='World X (not assumed geographic east)',ylabel='World Y',title=f'{m}; {x["anchor_name"]} ({x["anchor_id"]})')
  axes[1].bar(np.arange(8)*45,x['probabilities'],width=30);axes[1].axvline(x['GT_bearing_deg'],color='green',label='GT bearing');axes[1].set(xlabel='World planar angle: 0=+X, CCW',ylabel='Probability',title=f'Error {x["error"]:.1f} deg; context target: {x["target_in_public_context"]}');axes[1].legend();fig.tight_layout();fn=f'case_{j:02d}.png';fig.savefig(out/fn,dpi=150);plt.close(fig)
  cards.append(f'<article><h2>{j+1}. Query {x["index"]}; {html.escape(x["anchor_name"])}; error {x["error"]:.1f}°</h2><p>{html.escape(x["instruction"])}</p><p>Group: {x["relation_group"]}; unresolved frame: {x["relative_reference_unresolved"]}; same-name polygons: {len(x["same_name_public_ids"])}; extent: {x["anchor_extent_m"]}; GT-to-region: {x["GT_to_anchor_polygon_m"]:.1f}m.</p><a href="{fn}"><img width="1100" src="{fn}"></a><details><summary>Parsed chains and original spans</summary><pre>{html.escape(json.dumps(x["paths"],ensure_ascii=False,indent=2))}</pre></details></article>')
 (out/'cases.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2));(out/'index.html').write_text('<meta charset="utf-8"><title>Large bearing error audit</title><h1>Sealed confirmation errors over 90°</h1><p><a href="REPORT_zh.md">中文报告</a> · <a href="summary.json">统计</a> · <a href="all_large_errors.jsonl">全部大误差指令与Polygon</a></p><p>Posthoc TRAIN diagnosis only. No predictions, binding or weights changed. No geographic north/east assertion.</p>'+''.join(cards));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
