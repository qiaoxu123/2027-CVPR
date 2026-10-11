"""Public-only view planner. No target labels, simulator point clouds or RGB lookup."""
from dataclasses import dataclass
import math
import numpy as np
from shapely.geometry import Point,LineString,Polygon
from experiments.multiview_rgb_grounding.geometry import world_grid
from experiments.multiview_rgb_grounding.render import project
from graphnav.active_observation import plan_active_observation,ActiveObservationConfig

@dataclass(frozen=True)
class View:
    xyz: tuple
    camera_yaw: float
    camera_tilt: int
    action: str
    uav_yaw: float
    def camera_pose(self):return [*self.xyz,self.camera_yaw]
    def key(self):return tuple(round(float(v),6) for v in (*self.xyz,self.camera_yaw,self.camera_tilt))

def boxes(objects):
    result=[]
    for oid,obj in objects.items():
        p=np.asarray(obj.get('contour',[]));pos=np.asarray(obj.get('position',[]));dim=np.asarray(obj.get('dimension',[]))
        if p.ndim!=2 or len(p)<3 or pos.shape!=(3,) or dim.shape!=(3,):continue
        lo=p.min(0);hi=p.max(0);z0,z1=pos[2]-dim[2]/2,pos[2]+dim[2]/2
        corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [z0,z1]])
        result.append(dict(id=oid,center=p.mean(0),corners=corners,roof=float(z1),polygon=p,type=obj.get('object_type')))
    return result

def legal(start,xyz,bounds,ground,public_boxes):
    a=np.asarray(start,float);b=np.asarray(xyz,float);bounds=np.asarray(bounds,float)
    if not np.isfinite(b).all() or np.linalg.norm(b[:2]-a[:2])>50+1e-6 or abs(b[2]-a[2])>10+1e-6:return False
    if not 10-1e-6<=b[2]-ground<=100+1e-6 or np.any(b[:2]<bounds[:2]) or np.any(b[:2]>bounds[2:]):return False
    # Public-building path risk proxy; no hidden terrain or point-cloud query.
    for o in public_boxes:
        if o['type']!='Building' or min(a[2],b[2])>o['roof']+2:continue
        poly=Polygon(o['polygon'])
        if not poly.is_valid:continue
        path=Point(b[:2]) if np.linalg.norm(b[:2]-a[:2])<1e-8 else LineString([a[:2],b[:2]])
        if poly.buffer(1).intersects(path):return False
    return True

def modes(public,coverage):
    grid=world_grid(public);s=np.asarray(public['static_logits'],float);ids=np.argsort(-s,kind='stable')[:8];w=np.exp(s[ids]-s[ids].max());w/=w.sum();groups=[]
    for i,prob in zip(ids,w):
        found=next((g for g in groups if np.linalg.norm(grid[i]-g['center'])<=25),None)
        if found is None:groups.append(dict(center=grid[i].copy(),mass=float(prob),need=float(prob*(1-coverage[i])),ids=[int(i)]))
        else:
            found['center']=(found['center']*found['mass']+grid[i]*prob)/(found['mass']+prob);found['mass']+=float(prob);found['need']+=float(prob*(1-coverage[i]));found['ids'].append(int(i))
    return sorted(groups,key=lambda g:-g['need'])[:2],ids,w

def history_coverage(public,observations,cutoff):
    grid=world_grid(public);c=np.zeros(len(grid),float)
    for r in observations:
        if r['source_step']>cutoff:raise ValueError('future evidence is forbidden')
        if r.get('credible',False):
            j=np.linalg.norm(grid-np.asarray(r['center_world']),axis=1).argmin();c[j]=max(c[j],float(r.get('confidence',1)))
    return c

def cost(view,current):
    a=np.asarray(current['xyz']);b=np.asarray(view.xyz);xy=float(np.linalg.norm(a[:2]-b[:2]));z=float(abs(a[2]-b[2]));yaw=abs(math.degrees(math.atan2(math.sin(view.camera_yaw-current['camera_yaw']),math.cos(view.camera_yaw-current['camera_yaw']))));tilt=abs(view.camera_tilt-current['camera_tilt'])
    return xy/50+z/10+(yaw+tilt)/180*.1

def library(public,current,ground,public_boxes,coverage):
    groups,_,_=modes(public,coverage);start=np.asarray(current['xyz']);bounds=np.asarray(public['bounds']);result={}
    for group in groups:
        goal=group['center'];d=goal-start[:2];distance=np.linalg.norm(d);unit=d/max(distance,1e-8)
        if distance<1e-8:unit=np.array([math.cos(current['uav_yaw']),math.sin(current['uav_yaw'])])
        left=np.array([-unit[1],unit[0]])
        poses=[('hold',start.copy()),('approach',np.r_[start[:2]+unit*min(50,distance),start[2]]),('left',np.r_[start[:2]+left*12,start[2]]),('right',np.r_[start[:2]-left*12,start[2]]),('descend',start+np.array([0,0,-10])),('ascend',start+np.array([0,0,10]))]
        for action,xyz in poses:
            if not legal(start,xyz,bounds,ground,public_boxes):continue
            d=goal-xyz[:2];yaw=math.atan2(d[1],d[0]) if np.linalg.norm(d)>1e-8 else current['camera_yaw']
            for tilt in [0,20,40,60]:
                v=View(tuple(xyz),float(yaw if tilt else current['uav_yaw']),tilt,action,current['uav_yaw']);result.setdefault(v.key(),v)
    if legal(start,start,bounds,ground,public_boxes):
        for tilt in [0,20,40,60]:
            v=View(tuple(start),current['uav_yaw'],tilt,'hold',current['uav_yaw']);result.setdefault(v.key(),v)
    return list(result.values())

def utility(view,public,current,coverage,public_boxes,quality_prior=.5):
    grid=world_grid(public);_,ids,prob=modes(public,coverage);value=0.
    for i,p in zip(ids,prob):
        nearby=[b for b in public_boxes if np.linalg.norm(b['center']-grid[i])<=30]
        if not nearby:
            xy=grid[i];corners=np.array([[xy[0]+a,xy[1]+b,current['ground']] for a in [-8,8] for b in [-8,8]])
            nearby=[dict(corners=corners)]
        best=0.
        for box in nearby:
            uv,z,_=project(box['corners'],view.camera_pose(),view.camera_tilt);front=z>.5
            if not front.any():continue
            q=uv[front];lo=q.min(0);hi=q.max(0);span=hi-lo;clipped=np.maximum(0,np.minimum(hi,512)-np.maximum(lo,0));visible=float(np.prod(clipped)/max(np.prod(span),1e-8));pixels=float(min(clipped));best=max(best,visible*min(1,pixels/16)*float(quality_prior))
        value+=float(p)*(1-float(coverage[i]))*best
    return value-.05*cost(view,current)

def select(strategy,public,current,ground,public_boxes,coverage,used,rng,quality_prior=.5):
    if used>=2:raise ValueError('observation budget exhausted')
    views=library(public,current,ground,public_boxes,coverage)
    if not views:raise ValueError('no legal views')
    hold=View(tuple(current['xyz']),current['uav_yaw'],0,'hold',current['uav_yaw'])
    if strategy=='fixed_nadir':return hold
    if strategy=='existing_belief_active':
        d=plan_active_observation(pose_xyzyaw=[*current['xyz'],current['uav_yaw']],map_bounds=public['bounds'],ground_level=ground,candidate_xy=public['grid_xy'],candidate_logits=public['static_logits'],observed_confidence=coverage,instruction=current['instruction'],step=used,used_actions=used,config=ActiveObservationConfig())
        if d is None:return hold
        v=View(tuple(d.xyz),current['uav_yaw'],0,d.action,current['uav_yaw'])
        return v if legal(current['xyz'],v.xyz,public['bounds'],ground,public_boxes) else hold
    if strategy=='camera_only':views=[v for v in views if np.linalg.norm(np.asarray(v.xyz)-current['xyz'])<1e-8]
    elif strategy=='move_nadir':views=[v for v in views if v.camera_tilt==0]
    elif strategy=='random_legal':return views[int(rng.integers(len(views)))]
    elif strategy!='active_view':raise ValueError(strategy)
    return max(views,key=lambda v:utility(v,public,current,coverage,public_boxes,quality_prior))
