"""Public geometry whitelist and spatially sparse object graph, no GT fields."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
import torch

TYPES=('Building','Car','Footpath','Ground','HighVegetation','Water',
       'TrafficRoad','StreetFurniture','Parking','Wall','Bridge','Bike','Rail','Unknown')
PUBLIC_KEYS={'id','map_name','name','object_type','contour','position','dimension'}
FEATURE_DIM=len(TYPES)+12
EDGE_DIM=10


def public_catalog(raw):
    """P5 explicitly permits ALL object geometry, never private descriptions."""
    return {m:{str(i):{k:v for k,v in o.items() if k in PUBLIC_KEYS}
               for i,o in entries.items()} for m,entries in raw.items()}


def shape_of(obj):
    p=np.asarray(obj.get('contour',[]),float)
    if p.ndim==2 and p.shape[1:]==(2,) and len(p)>=3 and np.isfinite(p).all():
        q=Polygon(p)
        if q.is_valid and q.area>1e-10:return q,True
    xyz=np.asarray(obj.get('position',[0,0,0]),float)
    if len(xyz)<2 or not np.isfinite(xyz).all():raise ValueError('no finite public geometry')
    return Point(xyz[:2]),False


class ObjectGraph:
    def __init__(self,objects,bounds,*,near=25.,neighbors=16):
        # Order by actual geometry/type/name, never numeric ID or target key.
        pairs=[]
        for oid,obj in objects.items():
            g,valid=shape_of(obj)
            pairs.append((str(obj.get('object_type','Unknown')),float(g.centroid.x),float(g.centroid.y),
                          float(g.area),str(obj.get('name','')),str(oid),obj,g,valid))
        pairs.sort(key=lambda x:x[:5])
        self.ids=[x[5] for x in pairs];self.byid={i:j for j,i in enumerate(self.ids)}
        self.objects=[x[6] for x in pairs];self.shapes=np.array([x[7] for x in pairs],object)
        self.valid=np.array([x[8] for x in pairs],bool)
        self.centers=np.array([[g.centroid.x,g.centroid.y] for g in self.shapes],float)
        self.types=[o.get('object_type','Unknown') for o in self.objects]
        self.names=[str(o.get('name','')) for o in self.objects]
        self.bounds=np.asarray(bounds,float);self.tree=STRtree(self.shapes)
        span=np.maximum(self.bounds[2:]-self.bounds[:2],1.);self.features=[];self.areas=[];self.aspects=[]
        for i,(o,g) in enumerate(zip(self.objects,self.shapes)):
            area=float(g.area);perimeter=float(g.length);w,h=np.maximum(np.array(g.bounds[2:])-g.bounds[:2],0)
            angle=0.;aspect=1.
            if self.valid[i]:
                corners=np.array(g.minimum_rotated_rectangle.exterior.coords)[:4];delta=np.roll(corners,-1,axis=0)-corners;lengths=np.linalg.norm(delta,axis=1);a=int(lengths.argmax());angle=float(np.arctan2(delta[a,1],delta[a,0]));aspect=float(lengths.max()/max(lengths.min(),1e-8))
            xyz=np.asarray(o.get('position',[*self.centers[i],0]),float);dim=np.asarray(o.get('dimension',[w,h,0]),float)
            onehot=np.zeros(len(TYPES));onehot[TYPES.index(self.types[i]) if self.types[i] in TYPES else -1]=1
            geo=np.array([* ((self.centers[i]-self.bounds[:2])/span),np.log1p(area)/10,np.log1p(perimeter)/10,
                          w/span[0],h/span[1],np.log1p(aspect)/3,np.cos(2*angle),np.sin(2*angle),
                          float(xyz[2])/100,float(dim[2])/50,float(self.valid[i])])
            self.features.append(np.concatenate([onehot,geo]));self.areas.append(area);self.aspects.append(aspect)
        self.features=np.asarray(self.features,np.float32);self.areas=np.asarray(self.areas);self.aspects=np.asarray(self.aspects)
        edges={}
        for i,g in enumerate(self.shapes):
            candidates=self.tree.query(g.buffer(near));candidates=candidates[candidates!=i]
            if not len(candidates):continue
            distance=shapely.distance(self.shapes[candidates],g)
            keep=distance<=near; candidates=candidates[keep];distance=distance[keep]
            center_dist=np.linalg.norm(self.centers[candidates]-self.centers[i],axis=1)
            order=np.lexsort((center_dist,distance))[:neighbors]
            for k in order:
                j=int(candidates[k]);d=float(distance[k]);delta=self.centers[j]-self.centers[i]
                overlap=float(self.valid[i] and self.valid[j] and g.intersection(self.shapes[j]).area>1e-8)
                contains=float(g.contains(self.shapes[j]));within=float(self.shapes[j].contains(g))
                road=self.types[i] in ('TrafficRoad','Footpath','Bridge') and self.types[j] in ('TrafficRoad','Footpath','Bridge')
                dz=abs(float(self.objects[i].get('position',[0,0,0])[2])-float(self.objects[j].get('position',[0,0,0])[2]))
                road_touch=float(road and d<=1e-6 and dz<=2.5)
                attr=np.array([delta[0]/50,delta[1]/50,np.linalg.norm(delta)/50,d/25,float(d<=25),
                               float(d<=1),overlap,contains,within,road_touch],np.float32)
                edges[(i,j)]=attr;rev=attr.copy();rev[:2]*=-1;rev[7:9]=attr[[8,7]];edges[(j,i)]=rev
        for i in range(len(self.ids)):edges[(i,i)]=np.zeros(EDGE_DIM,np.float32)
        keys=sorted(edges);self.edge_index=np.asarray(keys,np.int64).T
        self.edge_attr=np.stack([edges[k] for k in keys])

    def local(self,centers,radius=100.,mandatory=(),limit=192):
        """Public predicted modes and referenced nodes; no target argument."""
        centers=np.asarray(centers,float).reshape(-1,2);selected=set(map(int,mandatory))
        for xy in centers:selected.update(map(int,self.tree.query(Point(xy).buffer(radius))))
        allids=np.array(sorted(selected),int)
        if len(allids)>limit:
            distance=np.linalg.norm(self.centers[allids,None]-centers[None],axis=2).min(1)
            keep=list(dict.fromkeys([*map(int,mandatory),*allids[np.argsort(distance,kind='stable')].tolist()]))[:limit]
            allids=np.array(keep,int)
        return allids

    def payload(self):
        return dict(ids=self.ids,objects=self.objects,bounds=self.bounds.tolist(),
                    features=torch.tensor(self.features),edge_index=torch.tensor(self.edge_index),edge_attr=torch.tensor(self.edge_attr),
                    stats=dict(nodes=len(self.ids),edges=self.edge_index.shape[1],valid_polygons=int(self.valid.sum()),
                               anonymous=sum(not n for n in self.names),type_counts=dict(Counter(self.types))))

    @classmethod
    def restore(cls,payload):
        obj=cls.__new__(cls);obj.ids=payload['ids'];obj.byid={i:j for j,i in enumerate(obj.ids)};obj.objects=payload['objects'];obj.bounds=np.array(payload['bounds'])
        pairs=[shape_of(o) for o in obj.objects];obj.shapes=np.array([x[0] for x in pairs],object);obj.valid=np.array([x[1] for x in pairs]);obj.centers=np.array([[g.centroid.x,g.centroid.y] for g in obj.shapes]);obj.types=[o.get('object_type','Unknown') for o in obj.objects];obj.names=[str(o.get('name','')) for o in obj.objects];obj.features=payload['features'].numpy();obj.edge_index=payload['edge_index'].numpy();obj.edge_attr=payload['edge_attr'].numpy();obj.areas=np.array([g.area for g in obj.shapes]);obj.aspects=np.expm1(obj.features[:,len(TYPES)+6]*3);obj.tree=STRtree(obj.shapes)
        return obj
