"""Read-only TRAIN post-hoc audit. Never fits compass orientation from target GT."""
import argparse,csv,hashlib,html,json,sys
from collections import Counter
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from graphnav.renderer import CityNavObservedViewRenderer

UNITS={'north':(0,1),'south':(0,-1),'east':(1,0),'west':(-1,0)}
NOTES={2443:'目标在工地内；north of修饰工地，须保留目标→工地→地标链。',1096:'west of修饰第9排停车位；目标在该排内，需保留序数和关系链。',1161:'west of修饰第14排停车位；不能忽略排与实例约束。',5836:'facing the north side描述朝向/道路侧，且原文明确when viewing the road as a horizontal road；不宜作无歧义世界位置标签。',6517:'同一目标包含两个关系；Perry Barr footpath绑定到Perry Barr可能存在名称范围混淆，需核对公开对象类型。',6047:'north of修饰停车场；目标包含于停车场。',6099:'north of修饰停车场；目标包含于停车场。',6215:'south of修饰车排；须保留包含链。',6245:'south of修饰车排；须保留包含链。',6250:'south of修饰车排；须保留包含链。',6186:'原文是地标在车辆北侧；抽取成车辆在地标南侧，逆关系正确。',398:'Celine Grove为道路；整体质心不能代表指令指涉的局部路段。',449:'Celine Grove为道路；检查全片段、范围及局部作用点，不据GT挑选推理Anchor。'}
def interval_status(vertices,target,unit):
 s=np.asarray(vertices)@np.asarray(unit);t=float(np.asarray(target)@unit)
 return ('desired_side' if t>s.max() else 'opposite_side' if t<s.min() else 'within_anchor_extent'),float(t-s.min()),float(t-s.max())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--sbf-source',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('reports/direction_frame_audit'));a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True);(o/'frames').mkdir(exist_ok=True)
 base=Path('reports/direction_belief');inputs=[base/'evaluation/public_dev.json',base/'data_split/private_dev_labels.json',base/'evaluation/real_probabilities.npz'];before={str(x):sha(x) for x in inputs};rows=json.loads(inputs[0].read_text());labels={r['index']:r for r in json.loads(inputs[1].read_text())};probs=np.load(inputs[2])['direction_real'];raw=json.loads((a.data_root/'original_citynav/citynav_train_seen.json').read_text());catalog=json.loads((a.data_root/'cityrefer_original/objects.json').read_text());sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
 renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',o/'frames',384);results=[];raster_meta={};maxroundtrip=0
 for j,r in enumerate(rows):
  gt=np.array(labels[r['index']]['target_world'][:2]);polys=[np.array(x) for x in r['public_polygons']];geom=unary_union([Polygon(x).buffer(0) for x in polys]);center=np.array(geom.centroid.coords[0]);v=np.concatenate(polys);unit=np.array(UNITS[r['word']]);b=np.array(r['bounds']);xx,yy=np.meshgrid(b[0]+(np.arange(28)+.5)*r['span'][0]/28,b[1]+(np.arange(28)+.5)*r['span'][1]/28);an=np.array(r['anchor']);mean=np.array([(an*xx).sum()/an.sum(),(an*yy).sum()/an.sum()]);status,low,high=interval_status(v,gt,unit)
  objs=catalog[r['map_name']];bound=[objs[str(k)] for k in r['public_object_ids']];names=[x.get('name',x.get('object_name','')) for x in bound];types=[x.get('object_type') for x in bound]
  exact_ids=[str(k) for k,x in objs.items() if x.get('name','').strip().casefold()==r['landmark_name'].strip().casefold()];assert set(exact_ids)==set(r['public_object_ids']), 'Incomplete exact-name binding'
  target_ids=labels[r['index']]['target_ids'];targetpolys=[Polygon(objs[str(k)]['contour']).buffer(0) for k in target_ids];tgdist=float(unary_union(targetpolys).distance(Point(gt)))
  flags=[]
  if status=='within_anchor_extent':flags.append('anchor_extent_ambiguous')
  if len(polys)>1:flags.append('multi_polygon')
  if float((gt-center)@unit)<=0 and status=='opposite_side':flags.append('literal_direction_or_reference_frame_unresolved')
  if r['episode'] in NOTES:flags.append('text_scope_review')
  if r['episode']==5836:flags.append('position_relation_parse_not_supported')
  if r['episode']==6517 and r['landmark_name']=='Perry Barr':flags.append('anchor_name_scope_unverified')
  poses=raw[r['episode']]['trajectory'];native=[r['start_pose'],[float(x) for x in poses[-1][:3]]+[float(np.arctan2(poses[-1][4],poses[-1][3]))]]
  fig,axes=plt.subplots(1,4,figsize=(19,5));frameinfo=[]
  for k,pose in enumerate(native):
   if pose[2]<=GROUND_LEVEL[r['map_name']]:
    axes[k+1].text(.1,.5,'Original pose at/below ground: RGB unavailable',wrap=True);frameinfo.append(dict(pose=pose,footprint=None,path=None,actual_visibility='unknown',reason='nonpositive_original_altitude'));continue
   obs=renderer.current_rgbd(r['map_name'],pose,GROUND_LEVEL[r['map_name']]);fp=obs['footprint_world'];img=obs['bgr'];fname=f"frames/{j:02d}_{r['episode']}_{k}.png";cv2.imwrite(str(o/fname),img);ax=axes[k+1];ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));H=cv2.getPerspectiveTransform(fp.astype(np.float32),np.float32([[0,0],[383,0],[383,383],[0,383]]));inv=np.linalg.inv(H);project=lambda q:cv2.perspectiveTransform(np.asarray(q,np.float64).reshape(1,-1,2),H)[0];q=project(fp);back=cv2.perspectiveTransform(q.reshape(1,-1,2),inv)[0];maxroundtrip=max(maxroundtrip,float(np.abs(back-fp).max()))
   for poly in polys:
    q=project(np.vstack([poly,poly[0]]));ax.plot(q[:,0],q[:,1],'c-',lw=1)
   q=project([gt])[0];inside=bool(((q>=0)&(q<=383)).all());
   if inside:ax.scatter(*q,c='lime',marker='*',s=80)
   c=np.array(pose[:2]);q=project([c,c+[15,0],c+[0,15]])
   for n,col in [(1,'yellow'),(2,'magenta')]:ax.annotate('+X' if n==1 else '+Y',xy=q[n],xytext=q[0],color=col,arrowprops=dict(arrowstyle='->',color=col))
   ax.set(xlim=(0,383),ylim=(383,0),title=f"{'Initial' if k==0 else 'Human final (audit only)'} RGB\nyaw={np.degrees(pose[3]):.1f}; GT in footprint={inside}");frameinfo.append(dict(pose=pose,footprint=fp.tolist(),path=fname,gt_in_footprint=inside,actual_visibility='unknown'))
  rast=renderer._raster;raster_meta[r['map_name']]=dict(crs=str(rast.crs),transform=list(rast.transform),bounds=list(rast.bounds));ax=axes[0];ax.imshow(cv2.cvtColor(renderer._rgb,cv2.COLOR_BGR2RGB),extent=[rast.bounds.left,rast.bounds.right,rast.bounds.bottom,rast.bounds.top]);traj=np.array(poses);ax.plot(traj[:,0],traj[:,1],'w-',lw=.7)
  for poly in polys:ax.plot(*np.vstack([poly,poly[0]]).T,'c-',lw=1)
  ax.scatter(*gt,c='lime',marker='*',s=100);ax.scatter(*center,c='red',s=30);ax.annotate(r['word']+' (hypothesis)',xy=center+30*unit,xytext=center,color='orange',fontsize=7,arrowprops=dict(arrowstyle='->',color='orange'))
  for f in frameinfo:
   if f['footprint'] is not None:ax.plot(*np.vstack([f['footprint'],f['footprint'][0]]).T,'y--',lw=.6)
  ax.set(xlim=(b[0],b[2]),ylim=(b[1],b[3]),title='Source orthophoto | +Y up, +X right\nGreen GT; cyan public anchor; red centroid')
  ax=axes[3];ax.imshow(probs[j],origin='lower',extent=[b[0],b[2],b[1],b[3]],cmap='magma');ax.scatter(*gt,c='lime',marker='*',s=80)
  for poly in polys:ax.plot(*np.vstack([poly,poly[0]]).T,'c-',lw=.6)
  ax.set(title='Previously sealed Target Belief\nWorld axes, not a new prediction',xlim=(b[0],b[2]),ylim=(b[1],b[3]));fig.suptitle(f"{r['episode']} | {r['word']} of {r['landmark_name']} | {status}\nEngineering compass hypothesis only; geographic north not established",fontsize=10);fig.tight_layout();image=f'case_{j:02d}_{r["episode"]}.png';fig.savefig(o/image,dpi=100);plt.close(fig)
  results.append(dict(index=r['index'],episode=r['episode'],map_name=r['map_name'],instruction=r['instruction'],word=r['word'],landmark=r['landmark_name'],public_object_ids=r['public_object_ids'],catalog_names=names,object_types=types,polygon_count=len(polys),target=gt.tolist(),centroid=center.tolist(),grid_mean=mean.tolist(),centroid_direction_margin=float((gt-center)@unit),grid_direction_margin=float((gt-mean)@unit),interval_status=status,margin_from_min=low,margin_from_max=high,fragment_status=[interval_status(x,gt,unit)[0] for x in polys],target_to_instance_polygon_distance=tgdist,flags=flags,note=NOTES.get(r['episode'],'字面主客体未见明确倒置；语言参考系及真实地标身份仍须独立来源确认。'),confidence='geometry_computed; text_review_by_current_assistant; no_human_pixel_gold',source_relation=r['source_relation'],frames=frameinfo,image=image))
 assert before=={str(x):sha(x) for x in inputs}
 summary=dict(pairs=len(results),episodes=len(set(r['episode'] for r in results)),maps=len(raster_meta),interval_counts=dict(Counter(r['interval_status'] for r in results)),flags=dict(Counter(f for r in results for f in r['flags'])),centroid_positive=sum(r['centroid_direction_margin']>0 for r in results),grid_positive=sum(r['grid_direction_margin']>0 for r in results),target_outside_polygon=sum(r['target_to_instance_polygon_distance']>0 for r in results),roundtrip_max_m=maxroundtrip,raster=raster_meta,immutable_inputs=before,geographic_mapping='unestablished: no CRS or independently documented linguistic compass transform; no GT fitting performed',engineering_convention=UNITS)
 (o/'samples.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));(o/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 with (o/'samples.csv').open('w') as f:
  keys=['episode','word','landmark','polygon_count','centroid_direction_margin','grid_direction_margin','interval_status','target_to_instance_polygon_distance','flags','note'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in results)
 page='<meta charset="utf-8"><title>方向参考系审查</title><style>body{font:16px sans-serif;margin:20px}img{width:100%}article{border-top:1px solid #aaa}</style><h1>34关系 / 33指令逐样本审查</h1><p>GT、完整地图、末帧仅供离线审计；绿色不代表人工像素金标准。坐标箭头为工程XY，未证明地理北。</p><a href="REPORT_zh.md">中文报告</a>'
 for r in results:page+=f'<article><h2>{r["episode"]} {html.escape(r["landmark"])}</h2><p>{html.escape(r["instruction"])}</p><p>{html.escape(r["note"])} | {r["interval_status"]}</p><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}"></a></article>'
 (o/'index.html').write_text(page);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
