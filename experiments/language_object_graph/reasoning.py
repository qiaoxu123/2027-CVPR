"""Non-GNN joint relation factors, object posterior and continuous Goal prior."""
import itertools
import numpy as np
import shapely
from shapely.geometry import MultiPoint


def probability(scores):
    x=np.asarray(scores,float);x=np.exp(x-x.max());return x/max(float(x.sum()),1e-30)


def object_kernels(graph,world_grid):
    key=np.asarray(world_grid,dtype=np.float64).tobytes()
    if getattr(graph,'_kernel_key',None)==key:return graph._kernel_value
    points=shapely.points(np.asarray(world_grid));distance=shapely.distance(graph.shapes[:,None],points[None])
    kernels=np.exp(-.5*np.square(distance/10.));s=kernels.sum(1,keepdims=True)
    kernels=np.where(s>1e-24,kernels/np.maximum(s,1e-24),1/len(points))
    graph._kernel_key=key;graph._kernel_value=kernels.astype(np.float32)
    return graph._kernel_value


def unary(entity,graph):
    score=np.zeros(len(graph.ids),float);typ=entity['object_type']
    if typ is not None:score+=np.array([0. if t==typ else -5. for t in graph.types])
    named=entity['named_ids']
    if named:
        match=np.zeros(len(score),bool);match[list(named)]=True;score+=np.where(match,0.,-12.)
    same=np.array([t==typ for t in graph.types]) if typ else np.ones(len(score),bool)
    areas=graph.areas[same];q25,q75=np.quantile(areas,[.25,.75]) if len(areas) else (0,0)
    if entity['size'] in ('small','tiny'):score-=.5*np.log1p(graph.areas/max(q25,1))
    elif entity['size'] in ('large','big','huge'):score-=.5*np.log1p(max(q75,1)/np.maximum(graph.areas,1))
    if entity['shape']=='square':score-=.5*np.abs(np.log(np.maximum(graph.aspects,1)))
    elif entity['shape'] in ('long','narrow'):score+=.5*np.log(np.maximum(graph.aspects,1))
    # L/U/circular shapes and subtype/color have no trustworthy static labels.
    return score


def candidates(entity,graph,prior,maximum=12,visual=None):
    u=unary(entity,graph);score=u+np.log(np.maximum(prior,1e-30))
    if visual is not None:
        visual=np.asarray(visual,float)
        if visual.shape!=score.shape:raise ValueError('reference visual evidence shape')
        score+=np.clip(visual,-1,1)
    if entity['named_ids']:
        pool=np.array(entity['named_ids'],int)
    elif entity['object_type']:
        pool=np.array([i for i,t in enumerate(graph.types) if t==entity['object_type']],int)
    else:
        # Fully unresolved references are Unknown rather than a random anchor.
        return np.empty(0,int),np.empty(0,float)
    if not len(pool):return np.empty(0,int),np.empty(0,float)
    if len(pool)>maximum:pool=pool[np.argsort(score[pool],kind='stable')[::-1][:maximum]]
    weights=probability(score[pool]);return pool,weights


def directional(kind,xy,center,yaw):
    world={'east':np.array([1.,0]),'west':np.array([-1.,0]),'north':np.array([0.,1]),'south':np.array([0.,-1])}
    delta=xy-center
    if kind in world:return 1/(1+np.exp(np.clip(-(delta@world[kind])/10.,-60,60)))
    f=np.array([np.cos(yaw),np.sin(yaw)]);left=np.array([-f[1],f[0]])
    vectors={'left':[-world['east'],left,-left],
             'right':[world['east'],-left,left],
             'front':[world['north'],f,-f],
             'behind':[-world['north'],-f,f]}[kind]
    # Retain three frame hypotheses; never use target GT to select the frame.
    return np.mean([1/(1+np.exp(np.clip(-(delta@v)/10.,-60,60))) for v in vectors],axis=0)


def between(xy,centers):
    centers=np.asarray(centers,float)
    if len(centers)==2:
        a,b=centers;v=b-a;l=float(v@v)
        if l<1:return np.zeros(len(xy))
        t=np.clip((xy-a)@v/l,0,1);projection=a+t[:,None]*v;distance=np.linalg.norm(xy-projection,axis=1)
        # Penalize endpoints: between is not two independent Near terms.
        interval=np.maximum(0,1-2*np.abs(t-.5))
        return np.exp(-.5*(distance/25.)**2)*interval
    hull=MultiPoint(centers).convex_hull
    if hull.area<1:return between(xy,centers[:2])
    distance=shapely.distance(shapely.points(xy),hull)
    return np.exp(-.5*(distance/25.)**2)


def factor_field(factor,language,graph,xy,prior,yaw,maximum=12,reference_visual=None):
    reference_visual=reference_visual or {}
    groups=[candidates(language['entities'][i],graph,prior,maximum,reference_visual.get(i)) for i in factor['anchors']]
    if not groups or any(not len(ids) for ids,_ in groups):return None,[]
    kind=factor['kind'];hypotheses=[];field=np.zeros(len(xy));total=0.
    if kind=='between':
        combos=list(itertools.product(*[range(len(ids)) for ids,_ in groups]))
        combos.sort(key=lambda c:-float(np.prod([groups[k][1][j] for k,j in enumerate(c)])))
        for combo in combos[:144]:
            ids=[int(groups[k][0][j]) for k,j in enumerate(combo)]
            if len(set(ids))!=len(ids):continue
            weight=float(np.prod([groups[k][1][j] for k,j in enumerate(combo)]))
            value=between(xy,graph.centers[ids]);field+=weight*value;total+=weight
            hypotheses.append(dict(object_indices=ids,weight=weight))
    else:
        ids,weights=groups[0]
        for i,w in zip(ids,weights):
            i=int(i);w=float(w)
            if kind in ('near','along'):
                d=shapely.distance(shapely.points(xy),graph.shapes[i]);value=np.exp(-d/(40. if kind=='along' else 25.))
            else:value=directional(kind,xy,graph.centers[i],yaw)
            field+=w*value;total+=w;hypotheses.append(dict(object_indices=[i],weight=w))
    if total<=0:return None,[]
    return field/total,hypotheses


def infer(graph,language,world_grid,static_logits,*,relations=False,yaw=0.,visual_evidence=None,reference_visual=None):
    """Public-only forward. No label, target coordinate or object identity args."""
    base=probability(static_logits);kernels=object_kernels(graph,world_grid)
    prior=np.maximum(kernels@base,1e-30);score=unary(language['entities'][0],graph)+np.log(prior)
    fields=[];explanations=[];mandatory=[]
    if relations:
        for f in language['factors']:
            v,h=factor_field(f,language,graph,np.asarray(world_grid),prior,yaw,reference_visual=reference_visual)
            if v is not None:
                fields.append(v);explanations.append(dict(factor=f,hypotheses=h))
                for x in h:mandatory.extend(x['object_indices'])
    field=np.ones(len(base))
    for v in fields:field*=np.maximum(v,1e-3)
    if fields:score+=np.log(np.maximum(kernels@field,1e-12))
    if visual_evidence is not None:
        evidence=np.asarray(visual_evidence,float)
        if evidence.shape!=score.shape:raise ValueError('node evidence shape')
        # Evidence already abstains to0 for unsupported/unobserved attributes.
        score+=np.clip(evidence,-1,1)
    belief=probability(score);object_heat=belief@kernels
    # Continuous regions between objects survive even when no target object exists.
    continuous=base*field;continuous/=max(continuous.sum(),1e-30)
    heat=.5*object_heat+.5*continuous;heat/=heat.sum()
    return dict(object_belief=belief,heatmap=heat,kernels=kernels,
                relation_field=field,factors=explanations,mandatory=sorted(set(mandatory)),
                static=base,node_unary=score)


def mix(original_logits,posterior,weight):
    if not 0<=weight<=1:raise ValueError('mixture range')
    if weight==0:return np.asarray(original_logits).copy()
    original=probability(original_logits);new=(1-weight)*original+weight*np.asarray(posterior,float)
    if new.shape!=original.shape or not np.isfinite(new).all():raise ValueError('same finite original grid required')
    return np.log(np.maximum(new,1e-30))
