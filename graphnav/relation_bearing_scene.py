"""Whole-public-Polygon scenes, selected without target coordinates or IDs."""
import math
from collections import defaultdict
import numpy as np
import shapely
from shapely.geometry import Polygon
from .polygon_neighborhood_bearing import TYPE_TO_ID,_describe_polygon


class RelationSceneBuilder:
    def __init__(self,catalog,*,max_nodes=64,radius_m=200.,samples=16):
        self.catalog=catalog;self.max_nodes=max_nodes;self.radius_m=float(radius_m);self.samples=samples;self._maps={};self._scenes={}
    def map_index(self,map_name):
        if map_name not in self._maps:
            rows={}
            for oid,obj in self.catalog[map_name].items():
                parsed=_describe_polygon(obj.get('contour',[]),self.radius_m)
                if parsed is None:continue
                center,summary=parsed;poly=Polygon(obj['contour'])
                pts=shapely.get_coordinates(shapely.line_interpolate_point(poly.exterior,np.linspace(0,1,self.samples,endpoint=False),normalized=True))
                typ=obj.get('object_type','Other');rows[str(oid)]=dict(id=str(oid),name=(typ+' '+obj.get('name','')).strip(),type=TYPE_TO_ID.get(typ,0),kind=typ,center=center,summary=summary,poly=poly,samples=pts)
            self._maps[map_name]=rows
        return self._maps[map_name]
    def scene(self,map_name,references):
        refids=sorted({str(r['id']) for r in references});key=(map_name,tuple(refids))
        if key in self._scenes:return self._scenes[key]
        rows=self.map_index(map_name);refs=[i for i in refids if i in rows]
        if len(refs)>self.max_nodes:raise ValueError('Never truncate named anchors')
        bucket=defaultdict(list)
        anchor_geoms=np.asarray([rows[i]['poly'] for i in refs],dtype=object)
        all_ids=list(rows)
        all_polys=np.asarray([rows[i]['poly'] for i in all_ids],dtype=object)
        distances=(shapely.distance(all_polys[:,None],anchor_geoms[None,:]).min(-1) if refs else np.full(len(all_ids),np.inf))
        for oid,dist in zip(all_ids,distances):
            row=rows[oid]
            if oid in refs or dist>self.radius_m:continue
            dist=float(dist)
            kind=row['kind'];group=0 if kind=='Building' else 1 if kind in {'TrafficRoad','Footpath','Bridge'} else 2 if kind in {'Parking','Ground','Water'} else 3 if kind=='Car' else 4
            bucket[group].append((dist,oid))
        for group in bucket:bucket[group].sort()
        ids=list(refs);positions=defaultdict(int)
        # Fixed type round-robin, independent of the target's class/identity.
        while len(ids)<self.max_nodes:
            changed=False
            for group in range(5):
                j=positions[group]
                if j<len(bucket[group]) and len(ids)<self.max_nodes:ids.append(bucket[group][j][1]);positions[group]+=1;changed=True
            if not changed:break
        n=len(ids);k=self.max_nodes;origin=np.mean([rows[i]['center'] for i in refs],axis=0) if refs else np.zeros(2)
        shape=np.zeros((k,self.samples*2+7),np.float32);types=np.zeros(k,np.int64);valid=np.zeros(k,bool);edges=np.zeros((k,k,10),np.float32)
        for j,oid in enumerate(ids):
            row=rows[oid];shape[j]=np.concatenate(((row['samples']-row['center']).ravel()/self.radius_m,row['summary'],(row['center']-origin)/self.radius_m));types[j]=row['type'];valid[j]=True
        if n:
            polys=np.asarray([rows[i]['poly'] for i in ids],dtype=object);centers=np.asarray([rows[i]['center'] for i in ids]);area=shapely.area(polys);delta=centers[:,None,:]-centers[None,:,:]
            distance=shapely.distance(polys[:,None],polys[None,:]);intersection=shapely.area(shapely.intersection(polys[:,None],polys[None,:]));iou=intersection/np.maximum(area[:,None]+area[None,:]-intersection,1e-9)
            bounds=np.asarray([rows[i]['poly'].bounds for i in ids]);overlap=(np.minimum(bounds[:,None,2:],bounds[None,:,2:])-np.maximum(bounds[:,None,:2],bounds[None,:,:2])>=0).all(-1)
            edges[:n,:n]=np.stack((delta[...,0]/self.radius_m,delta[...,1]/self.radius_m,np.linalg.norm(delta,axis=-1)/self.radius_m,distance/self.radius_m,iou,intersection/np.maximum(np.minimum(area[:,None],area[None,:]),1e-9),intersection/np.maximum(area[:,None],1e-9),intersection/np.maximum(area[None,:],1e-9),(distance<=2.).astype(float),overlap.astype(float)),axis=-1)
        else:valid[0]=True # padding-only no-anchor query: no supervised output
        scene=dict(map_name=map_name,node_ids=ids,node_names=[rows[i]['name'] for i in ids]+['']*(k-n),shape=shape,types=types,valid=valid,edges=edges,origin=origin)
        self._scenes[key]=scene;return scene
    def direction_kernel(self,scene,reference_ids):
        rows=self.map_index(scene['map_name']);r=len(reference_ids);k=self.max_nodes;out=np.full((r,k,8),1/8,np.float32);bins=np.arange(8)*math.pi/4
        for j,ref in enumerate(reference_ids):
            center=rows[str(ref)]['center']
            for i,oid in enumerate(scene['node_ids']):
                row=rows[oid];pts=np.vstack([row['center'][None],row['samples']]);delta=pts-center;keep=np.linalg.norm(delta,axis=-1)>1e-7
                if not keep.any():continue
                theta=np.arctan2(delta[keep,1],delta[keep,0]);z=8*np.cos(theta[:,None]-bins[None]);z-=z.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True);out[j,i]=p.mean(0)
        return out
