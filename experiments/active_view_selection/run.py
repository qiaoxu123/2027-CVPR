"""Sequential causal policies; privileged labels exist only in offline diagnostics."""
import hashlib,json,time,math
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import cv2,numpy as np,torch
from matplotlib.path import Path as Poly
from experiments.multiview_rgb_grounding.render import ROOT,OBJECTS,read_cloud,associate,render,regions,project
from experiments.multiview_rgb_grounding.geometry import world_grid
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from .sensor import render_culled
from .planner import boxes,legal,history_coverage,library,select,cost,View

R=Path('runs/active_view_selection');SEED=20261011

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def prepare():
 if (R/'cohort.pt').exists():
  manifest=json.loads((R/'cohort_manifest.json').read_text());assert digest(R/'cohort.pt')==manifest['cohort_sha256'];assert digest(Path(__file__).with_name('protocol.json'))==manifest['protocol_sha256'];return torch.load(R/'cohort.pt',weights_only=True)
 cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());objects=json.loads(OBJECTS.read_text());states=[];audit={};sources={}
 for split,n in [('train',4),('dev',6)]:
  source=Path('runs/instance_grounding/plan_'+split);p=torch.load(source/'plan.pt',weights_only=True);manifest=json.loads((source/'manifest.json').read_text());assert digest(source/'plan.pt')==manifest['files']['plan.pt'];sources[split]=digest(source/'plan.pt');groups=defaultdict(list);by_map=defaultdict(list);ground={v['map_name']:v['ground_level'] for v in p['views'].values()};rejected=defaultdict(int)
  for s in p['states']:groups[s['metadata']['family_id']].append(s)
  for family,ss in groups.items():
   ss.sort(key=lambda s:s['metadata']['cutoff_step']);s=ss[len(ss)//2];m=s['metadata'];pub=s['public'];name=m['map_name'];g=ground[name];bb=boxes(objects[name]);xyz=np.asarray(pub['pose_world'])[:3]
   if not legal(xyz,xyz,pub['bounds'],g,bb):rejected[name]+=1;continue
   hist=[]
   for v in p['views'].values():
    if v['episode_id']!=m['episode_id'] or v['source_step']>m['cutoff_step']:continue
    hist.append(dict(center_world=v['center_world'],source_step=v['source_step'],credible=bool(v['native_min_side']>=16 and v['depth_support']>=.9 and v['quality'][1]>=.7),confidence=1.))
   row=dict(split=split,metadata=m,public=pub,ground=float(g),history=hist,label=p['labels'][m['sample_id']]);by_map[name].append(row)
  names=sorted(by_map,key=lambda x:hashlib.sha256(('P3.5-map:'+x).encode()).hexdigest())
  for name in names:by_map[name].sort(key=lambda x:hashlib.sha256(('P3.5-family:'+x['metadata']['family_id']).encode()).hexdigest())
  chosen=[]
  for offset in range(max(map(len,by_map.values()),default=0)):
   for name in names:
    if offset<len(by_map[name]) and len(chosen)<n:chosen.append(by_map[name][offset])
  states+=chosen;audit[split]=dict(original_families=len(groups),legal_families=sum(map(len,by_map.values())),rejected_public_legality=dict(rejected),selected=[x['metadata']['sample_id'] for x in chosen])
 states.sort(key=lambda x:(x['metadata']['map_name'],x['metadata']['family_id']));R.mkdir(exist_ok=True);torch.save(states,R/'cohort.pt');dump(R/'cohort_manifest.json',dict(protocol_sha256=digest(Path(__file__).with_name('protocol.json')),sources=sources,audit=audit,cohort_sha256=digest(R/'cohort.pt'),GT_selects_cohort=False));return states

def visible_regions(frame,xyz,objects,gate):
 """Equivalent point-wise association, restricted to actual sensor winners."""
 win=frame['winner'];valid=win>=0;ids=np.unique(win[valid])
 if not len(ids):return [],np.full(win.shape,-1,np.int32)
 assignment,names,_,_=associate(xyz[ids],objects)
 local=np.full(win.shape,-1,np.int32);local[valid]=np.searchsorted(ids,win[valid]);sensor=dict(frame,winner=local)
 return regions(sensor,assignment,names,objects,gate)


class Simulator:
 def __init__(self,name,objects,gate):
  start=time.time();path=next(ROOT.glob('*/'+name+'.ply'));self.xyz,self.rgb,self.data=read_cloud(path);self.z_bounds=(float(self.xyz[:,2].min()),float(self.xyz[:,2].max()));self.names=sorted(objects,key=lambda x:int(x));self.ambiguous=None;self.objects=objects;self.gate=gate;self.cache={};self.seconds=time.time()-start;self.path=path;self.target_cache={};self.diagnostic_cache={}
 def exposure(self,view):
  key=hashlib.sha256(json.dumps([self.path.stem,view.key()]).encode()).hexdigest()[:20]
  if key in self.cache:return self.cache[key]
  existing=R/'frames'/key
  if (existing/'metadata.json').exists():
   meta=json.loads((existing/'metadata.json').read_text());assert meta['map_name']==self.path.stem and np.allclose(meta['camera_pose'],view.camera_pose(),atol=1e-6) and meta['view']['camera_tilt']==view.camera_tilt
   self.cache[key]=(meta,np.load(existing/'geometry.npz')['winner']);return self.cache[key]
  start=time.time();f=render_culled(self.xyz,self.rgb,view.camera_pose(),view.camera_tilt,self.z_bounds);regs,inst=visible_regions(f,self.xyz,self.objects,self.gate);d=R/'frames'/key;d.mkdir(parents=True,exist_ok=False);cv2.imwrite(str(d/'rgb.png'),f['rgb'][...,::-1]);cv2.imwrite(str(d/'visibility.png'),f['valid'].astype(np.uint8)*255);np.savez_compressed(d/'geometry.npz',depth=f['depth'],winner=f['winner'],instance=inst)
  yy,xx=np.where(f['valid']);hull=np.zeros((512,512),np.uint8)
  if len(xx)>=3:cv2.fillConvexPoly(hull,cv2.convexHull(np.stack([xx,yy],1).astype(np.int32)),1)
  valid=float(f['valid'].mean());meta=dict(frame_id=key,view=asdict(view),camera_pose=view.camera_pose(),valid_fraction=valid,inside_hull_valid=float(f['valid'].sum()/max(1,hull.sum())),frame_quality_pass=bool(valid>=.6 and f['reprojection_max']<=2.13),reprojection_max=f['reprojection_max'],regions=regs,render_seconds=time.time()-start,measured_winning_points=len(np.unique(f['winner'][f['valid']])),origin='new reached-view measured point-cloud simulator exposure; not original CityFlight RGB',map_name=self.path.stem)
  dump(d/'metadata.json',meta);self.cache[key]=(meta,f['winner']);return self.cache[key]
 def diagnostic(self,meta,win,target):
  # Privileged evaluator; planner never receives the target ID or this result.
  key=(meta['frame_id'],target)
  if key in self.diagnostic_cache:return self.diagnostic_cache[key]
  if target not in self.target_cache:
   o=self.objects[target];poly=np.asarray(o['contour']);pos=np.asarray(o['position']);dim=np.asarray(o['dimension']);lo=poly.min(0);hi=poly.max(0);xyz=self.xyz;ids=np.flatnonzero((xyz[:,0]>=lo[0])&(xyz[:,0]<=hi[0])&(xyz[:,1]>=lo[1])&(xyz[:,1]<=hi[1])&(abs(xyz[:,2]-pos[2])<=dim[2]/2+.5));ids=ids[Poly(poly).contains_points(xyz[ids,:2],radius=1e-7)];self.target_cache[target]=ids
  ids=self.target_cache[target];_,_,good=project(self.xyz[ids],meta['camera_pose'],meta['view']['camera_tilt']);projected=int(good.sum());visible=int(np.isin(np.unique(win[win>=0]),ids).sum());reg=next((r for r in meta['regions'] if r['object_id']==target),None);credible=bool(reg and reg['credible'] and meta['frame_quality_pass']);reason='credible' if credible else 'outside_fov' if projected==0 else 'occluded_or_no_winning_points' if visible==0 else 'weak_association_ambiguous' if reg is None else 'native_pixels_or_support' if not reg['credible'] else 'full_frame_quality'
  result=dict(fov=bool(projected),credible=credible,reason=reason,target_region=reg,target_source_points=len(ids),target_projected_points=projected,target_winning_points=visible,weak_annotation=True);self.diagnostic_cache[key]=result;return result

def reached_history(meta,objects,cutoff):
 return [dict(center_world=np.mean(objects[r['object_id']]['contour'],axis=0).tolist(),source_step=cutoff,credible=True,confidence=1.) for r in meta['regions'] if r['credible'] and meta['frame_quality_pass']]

def current_state(row):
 p=np.asarray(row['public']['pose_world']);return dict(xyz=p[:3].tolist(),uav_yaw=float(p[3]),camera_yaw=float(p[3]),camera_tilt=0,ground=row['ground'],instruction=row['metadata']['instruction'])

def summarize_trace(row,strategy,trace,baseline,sim):
 m=row['metadata'];target=row['label']['target_object_id'];diag=[sim.diagnostic(meta,win,target) for view,meta,win in trace];start=current_state(row);xy=z=rotation=0.
 for v,meta,win in trace:
  xy+=float(np.linalg.norm(np.asarray(v.xyz[:2])-start['xyz'][:2]));z+=abs(v.xyz[2]-start['xyz'][2]);rotation+=abs(math.degrees(math.atan2(math.sin(v.camera_yaw-start['camera_yaw']),math.cos(v.camera_yaw-start['camera_yaw']))))+abs(v.camera_tilt-start['camera_tilt']);start.update(xyz=list(v.xyz),camera_yaw=v.camera_yaw,camera_tilt=v.camera_tilt)
 credibleids={r['object_id'] for v,meta,win in trace for r in meta['regions'] if r['credible'] and meta['frame_quality_pass']};before={r['object_id'] for r in baseline[0]['regions'] if r['credible'] and baseline[0]['frame_quality_pass']};pose=np.asarray(row['public']['pose_world']);static=np.asarray(row['public']['static_logits']);wrong=np.linalg.norm(world_grid(row['public'])[np.argsort(static,kind='stable')[-1]]-row['label']['target_world'])>20
 actualpoints=set(np.unique(baseline[1][baseline[1]>=0]).tolist());newpoints=set()
 for v,meta,win in trace:newpoints.update(np.unique(win[win>=0]).tolist())
 return dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],split=row['split'],strategy=strategy,target_object_id=target,static_error=bool(wrong),source_cutoff=m['cutoff_step'],exposures=[dict(view=asdict(v),frame_id=meta['frame_id'],diagnostic=d) for (v,meta,win),d in zip(trace,diag)],metrics=dict(target_credible=float(any(d['credible'] for d in diag)),target_fov=float(any(d['fov'] for d in diag)),error_credible=float(wrong and any(d['credible'] for d in diag)),candidate_coverage=len(credibleids),new_credible_instances=len(credibleids-before),new_measured_points=len(newpoints-actualpoints),xy_m=xy,z_m=z,gimbal_degrees=rotation,extra_observations=len(trace),max_native_target_side=max((d['target_region']['native_min_side'] if d['target_region'] else 0) for d in diag),mean_valid_fraction=float(np.mean([meta['valid_fraction'] for _,meta,_ in trace])),inside_hull_valid=float(np.mean([meta['inside_hull_valid'] for _,meta,_ in trace])),render_seconds=sum(meta['render_seconds'] for _,meta,_ in trace)))

def run():
 assert not (R/'predictions.json').exists();start=time.time();states=prepare();cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());gate=json.loads(Path('experiments/multiview_rgb_grounding/protocol.json').read_text())['quality_gates'];objects=json.loads(OBJECTS.read_text());rows=[];oracle_audits=[];clouds={};sim=None;name=None
 for index,row in enumerate(states):
  if time.time()-start>35*60:break
  m=row['metadata'];pub=row['public'];mapname=m['map_name'];bb=boxes(objects[mapname])
  if mapname!=name:
   sim=Simulator(mapname,objects[mapname],gate);name=mapname;clouds[name]=dict(path=str(sim.path),bytes=sim.path.stat().st_size,load_association_seconds=sim.seconds,points=len(sim.xyz),ambiguous_points=sim.ambiguous)
  current=current_state(row);baseview=View(tuple(current['xyz']),current['uav_yaw'],0,'initial',current['uav_yaw']);baseline=sim.exposure(baseview);initialhistory=row['history']+reached_history(baseline[0],objects[mapname],m['cutoff_step']);coverage=history_coverage(pub,initialhistory,m['cutoff_step']);pool=library(pub,current,row['ground'],bb,coverage)
  # First actual oracle scan. This is privileged evaluation, never fed to selector.
  catalogue=[]
  for v in pool:
   meta,win=sim.exposure(v);catalogue.append([(v,meta,win),(v,meta,win)])
  traces={}
  for strategy in cfg['strategies']:
   if strategy=='view_oracle':continue
   cur=current_state(row);history=list(initialhistory);trace=[];rng=np.random.default_rng(SEED+int(hashlib.sha256((m['family_id']+strategy).encode()).hexdigest()[:8],16));prior=.5+.5*baseline[0]['valid_fraction']
   for used in range(2):
    c=history_coverage(pub,history,m['cutoff_step']);v=select(strategy,pub,cur,row['ground'],bb,c,used,rng,quality_prior=prior);assert legal(cur['xyz'],v.xyz,pub['bounds'],row['ground'],bb);meta,win=sim.exposure(v);trace.append((v,meta,win));history+=reached_history(meta,objects[mapname],m['cutoff_step']);prior=.5+.5*meta['valid_fraction'];cur.update(xyz=list(v.xyz),camera_yaw=v.camera_yaw,camera_tilt=v.camera_tilt)
   traces[strategy]=trace;catalogue.append(trace);rows.append(summarize_trace(row,strategy,trace,baseline,sim))
  # Exact best reachable two-exposure trace among this evaluated catalogue.
  candidates=[summarize_trace(row,'view_oracle',trace,baseline,sim) for trace in catalogue]
  chosen=max(candidates,key=lambda x:(x['metrics']['target_credible'],x['metrics']['max_native_target_side'],-x['metrics']['xy_m']-x['metrics']['z_m']));rows.append(chosen)
  oracle_audits.append(dict(sample_id=m['sample_id'],first_step_views=len(pool),evaluated_two_exposure_traces=len(catalogue),oracle_scope='exact best trace over legal first-step view library (repeat exposure) and six reached two-step policy traces; not global physical optimum',GT_usage='offline target scoring only; chosen view never enters nonoracle history or policy'))
  dump(R/'partial_predictions.json',rows);dump(R/'oracle_catalogue.json',oracle_audits);print(json.dumps(dict(family=index+1,total=len(states),map=mapname,unique_rendered_views=len(sim.cache),seconds=time.time()-start,oracle_credible=chosen['metrics']['target_credible'])),flush=True)
 dump(R/'predictions.json',rows);summary={};methods=cfg['strategies'];keys=list(rows[0]['metrics']) if rows else []
 for split in ['train','dev','all']:
  summary[split]={}
  for method in methods:
   selected=[r for r in rows if r['strategy']==method and (split=='all' or r['split']==split)];summary[split][method]={k:summarize_rows(selected,k,seed=SEED) for k in keys};summary[split][method]['independent_visible_errors']=len({r['family_id'] for r in selected if r['metrics']['error_credible']});summary[split][method]['independent_visible_error_objects']=len({(r['map_name'],r['target_object_id']) for r in selected if r['metrics']['error_credible']})
  for other in ['fixed_nadir','camera_only','move_nadir','random_legal','existing_belief_active']:
   a=[r for r in rows if r['strategy']==other and (split=='all' or r['split']==split)];b=[r for r in rows if r['strategy']=='active_view' and (split=='all' or r['split']==split)];summary[split]['active_minus_'+other]=paired_difference(a,b,'target_credible',seed=SEED)
 dump(R/'metrics.json',summary);dump(R/'execution.json',dict(seconds=time.time()-start,completed_families=len(rows)//7,planned_families=len(states),clouds=clouds,training=False,seen=False,unseen=False,stage_b_run=False,causal_policy=True,models_modified=False,protocol_sha256=digest(Path(__file__).with_name('protocol.json'))));print(json.dumps(dict(complete=True,seconds=time.time()-start,families=len(rows)//7)),flush=True)

if __name__=='__main__':run()
