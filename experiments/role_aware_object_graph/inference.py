"""Role-sensitive sum-product factors and parse marginalization, no learned net."""
import numpy as np
import shapely
from .parser import parse
from experiments.language_object_graph.reasoning import probability,unary,object_kernels,candidates,between,directional,mix


def field(factor,language,graph,xy,pools,beliefs,yaw):
 anchors=factor['anchors'];kind=factor['kind'];groups=[(pools[a],beliefs[a]) for a in anchors]
 if not groups or any(not len(ids) for ids,_ in groups):return None
 if kind=='between':
  import itertools
  combinations=[]
  for indices in itertools.product(*[range(len(ids)) for ids,_ in groups]):
   ids=[int(groups[k][0][j]) for k,j in enumerate(indices)]
   if len(set(ids))<len(ids):continue
   weight=float(np.prod([groups[k][1][j] for k,j in enumerate(indices)]));combinations.append((weight,ids))
  combinations.sort(reverse=True);out=np.zeros(len(xy));total=0.
  for w,ids in combinations[:144]:out+=w*between(xy,graph.centers[ids]);total+=w
  return out/max(total,1e-30) if total else None
 ids,weights=groups[0];out=np.zeros(len(xy))
 for i,w in zip(ids,weights):
  if kind in ('near','along'):v=np.exp(-shapely.distance(shapely.points(xy),graph.shapes[int(i)])/(40 if kind=='along' else 25))
  else:v=directional(kind,xy,graph.centers[int(i)],yaw)
  out+=float(w)*v
 return out


def scoped(graph,language,world,static,*,method='A3',target=0,subjects=None,yaw=0.):
 base=probability(static);kernels=object_kernels(graph,world);prior=np.maximum(kernels@base,1e-30)
 entities=language['entities'];count=len(entities);pools=[];beliefs=[];scores=[]
 for i,e in enumerate(entities):
  if i==target:pool=np.arange(len(graph.ids));score=unary(e,graph)+np.log(prior)
  else:
   pool,_=candidates(e,graph,prior);score=unary(e,graph)[pool]+np.log(prior[pool])
  pools.append(pool);scores.append(score);beliefs.append(probability(score) if len(pool) else np.empty(0))
 factors=[]
 for j,f in enumerate(language['factors']):
  f=f.copy();f['subject']=target if method=='A2' else subjects[j] if subjects is not None else f['subject']
  if f['subject'] in f['anchors']:continue
  factors.append(f)
 # Reference-reference constraints affect reference marginals. Synchronous
 # updates use fixed unary each pass, not repeated multiplication of same edge.
 if method!='A2':
  for _ in range(2):
   incoming=[np.zeros(len(pool)) for pool in pools]
   for f in factors:
    s=f['subject']
    if s==target or not len(pools[s]) or target in f['anchors']:continue
    value=field(f,language,graph,graph.centers[pools[s]],pools,beliefs,yaw)
    if value is not None:incoming[s]+=np.log(np.maximum(value,1e-3))
   beliefs=[probability(score+inc) if len(pool) else np.empty(0) for pool,score,inc in zip(pools,scores,incoming)]
 fields=[];used=[]
 for f in factors:
  if f['subject']==target:
   value=field(f,language,graph,world,pools,beliefs,yaw)
   if value is not None:fields.append(value);used.append(f)
  elif target in f['anchors'] and len(f['anchors'])==1 and len(pools[f['subject']]):
   # Reverse a relation by averaging over observed PUBLIC subject candidates.
   # Relative frames remain marginalized; no target GT selects a direction.
   s=f['subject'];kind=f['kind'];out=np.zeros(len(world))
   inverse={'east':'west','west':'east','north':'south','south':'north','left':'right','right':'left','front':'behind','behind':'front'}
   for i,w in zip(pools[s],beliefs[s]):
    v=np.exp(-shapely.distance(shapely.points(world),graph.shapes[int(i)])/25) if kind in ('near','along') else directional(inverse[kind],world,graph.centers[int(i)],yaw)
    out+=float(w)*v
   fields.append(out);used.append(f)
 relation=np.ones(len(base))
 for v in fields:relation*=np.maximum(v,1e-3)
 node_score=scores[target].copy()
 if fields:node_score+=np.log(np.maximum(kernels@relation,1e-12))
 belief=probability(node_score);continuous=base*relation;continuous/=max(continuous.sum(),1e-30)
 heat=.5*(belief@kernels)+.5*continuous;heat/=heat.sum()
 supported=bool(fields) or bool(entities[target]['object_type'] or entities[target]['named_ids'])
 if not supported or language['confidence']==0:heat=base
 return dict(heatmap=heat,object_belief=belief,relation_field=relation,kernels=kernels,static=base,language=language,factors=used,reference_posteriors=[dict(entity=i,object_indices=pools[i].tolist(),probabilities=beliefs[i].tolist()) for i in range(count) if i!=target],supported=supported and language['confidence']>0)


def beams(language,limit=8):
 result=[dict(target=h['target'],subjects=[],probability=h['probability']) for h in language['hypotheses']]
 for f in language['factors']:
  new=[]
  for b in result:
   for h in f['subject_hypotheses']:new.append(dict(target=b['target'],subjects=b['subjects']+[h['subject']],probability=b['probability']*h['probability']))
  result=sorted(new,key=lambda b:-b['probability'])[:limit]
 total=sum(b['probability'] for b in result)
 # Missing beam mass is retained as Unknown/A0, never renormalized into certainty.
 if total<1-1e-12:result.append(dict(target=0,subjects=None,probability=1-total,null=True))
 return result


def forward(graph,instruction,public,*,method,weight,backend='rule',dependency=None):
 if set(public)!={'static_logits','grid_xy','bounds','pose_world'}:raise ValueError('strict public inputs; GT/annotations forbidden')
 bounds=np.asarray(public['bounds']);world=bounds[:2]+np.asarray(public['grid_xy'])*(bounds[2:]-bounds[:2]);language=parse(instruction,graph,backend=backend,dependency=dependency)
 if not language['entities']:
  k=object_kernels(graph,world);base=probability(public['static_logits']);belief=probability(np.log(np.maximum(k@base,1e-30)))
  return dict(language=language,logits=np.asarray(public['static_logits']).copy(),object_belief=belief,heatmap=base,factors=[],world_grid=world,parse_hypotheses=[],supported=False)
 configurations=beams(language) if method in ('A4','A4_damped_single') else [dict(target=0,subjects=None,probability=1.)]
 if method=='A4_damped_single':
  mass=sum(h['probability'] for h in configurations if not h.get('null'))
  configurations=[dict(target=0,subjects=None,probability=mass),dict(target=0,subjects=None,probability=1-mass,null=True)]
 outputs=[]
 for h in configurations:
  o=scoped(graph,language,world,public['static_logits'],method=method,target=h['target'],subjects=h['subjects'],yaw=float(public['pose_world'][3]))
  if h.get('null'):
   o['heatmap']=o['static'];o['supported']=False
  outputs.append(o)
 heat=sum(h['probability']*o['heatmap'] for h,o in zip(configurations,outputs));belief=sum(h['probability']*o['object_belief'] for h,o in zip(configurations,outputs));supported=any(o['supported'] for o in outputs)
 logits=mix(public['static_logits'],heat,weight) if supported else np.asarray(public['static_logits']).copy()
 return dict(language=language,logits=logits,object_belief=belief,heatmap=heat,factors=outputs[0]['factors'],world_grid=world,parse_hypotheses=configurations,reference_posteriors=outputs[0]['reference_posteriors'],supported=supported)
