"""Variable-conditioned observables; Unknown/unobserved always zero evidence."""
import numpy as np,torch
from .resources import regions


def template(e):
    # Texture/type wording is a feature, not a trusted fine semantic class.
    if e.get('named_ids'):return 'an object'
    return 'a '+(' '.join(e.get('colors',[]))+' ' if e.get('colors') else '')+(e.get('head_text') or e.get('object_type') or 'object').lower()


def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);return a@b/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b),1e-30)


def evidence(graph,language,state,features,templates,pixels,mode='real'):
    visible=regions(state);ids=sorted([graph.byid[o] for o in visible if o in graph.byid]);bank=features['bank'];black=np.asarray(features['pixel_zero']['image'],float);vectors={i:np.asarray(bank[visible[graph.ids[i]]['view_key']]['tight'],float) for i in ids};pixel_values={i:pixels.get(visible[graph.ids[i]]['view_key']) for i in ids}
    if mode=='shuffled':
        import hashlib
        seed=20261011+int(hashlib.sha256(state['metadata']['family_id'].encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed)
        for typ in sorted(set(graph.types[i] for i in ids)):
            group=[i for i in ids if graph.types[i]==typ]
            if len(group)<2:continue
            order=rng.permutation(group)
            if np.array_equal(order,group):order=np.roll(order,1)
            old=dict(vectors);oldp=dict(pixel_values)
            for a,b in zip(group,order):vectors[a]=old[int(b)]
            pgroup=[i for i in group if oldp[i] is not None]
            if len(pgroup)>1:
                porder=rng.permutation(pgroup)
                if np.array_equal(porder,pgroup):porder=np.roll(porder,1)
                for a,b in zip(pgroup,porder):pixel_values[a]=oldp[int(b)]
    if mode=='black':
        from .pixels import BLACK
        vectors={i:black for i in ids};pixel_values={i:({'color_scores':BLACK} if pixel_values[i] is not None else None) for i in ids}
    result={};coverage={}
    for v,e in enumerate(language['variables']):
        scores=np.zeros((len(graph.ids),2),np.float32);text=np.asarray(templates[template(e)],float);base=float(cosine(black,text));colors=e.get('colors',[])
        for i in ids:
            if e.get('named_ids'):continue
            r=visible[graph.ids[i]];quality=min(1.,r['native_min_side']/64)*r['depth_support'];match=float(cosine(vectors[i],text))-base
            # Bounded signed evidence is allowed for actually observed objects;
            # the same language conditioned black response is exactly removed.
            scores[i,0]=float(np.clip(match,-1,1))*quality
            p=pixel_values[i]
            if p is not None and len(colors)==1:scores[i,1]=(p['color_scores'].get(colors[0],0)-float(colors[0]=='black'))*quality
        result[v]=scores;coverage[v]=dict(observed=len(ids),pixel_scored=sum(p is not None for p in pixel_values.values()),unobserved=len(graph.ids)-len(ids))
    return result,coverage
