"""Same family/candidates/cutoff for geometry, HSV, frozen and RGB controls."""
import json,time
from pathlib import Path
import numpy as np,torch
from experiments.semantic_polygon_grounding.audit import save,sha
from .resources import OUT,baseline,public,primary_features
from .language import parse
from .visual import evidence
from .inference import prepare
from .factors import Geometry


def tensorize(r):
    result={k:r[k] for k in ('target_features','reference_features','log_phi','log_weights','support_mass','confidence','known_mass','assignments','language','unknown','reference_posteriors')}
    for k in ('target_features','reference_features','log_phi','log_weights'):result[k]=torch.tensor(result[k],dtype=torch.float16)
    return result


def main():
    start=time.time();torch.set_num_threads(1);dest=OUT/'observed';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():raise RuntimeError('preserve observed contrasts')
    a3,source,graphs,proof=baseline();lookup={(q['map_name'],q['instruction']):q for q in source['queries'] if q['split'] in ('train','dev')};templates=torch.load(OUT/'frozen_text/templates.pt',weights_only=True);pixels=torch.load(OUT/'pixels/bank.pt',weights_only=True);stats={};geometries={}
    for split in ('train','dev'):
        features,states,manifest=primary_features(split);prepared=[];labels=[]
        for j,state in enumerate(states):
            m=state['metadata'];q=lookup[(m['map_name'],m['instruction'])];assert q['split']==split
            g=graphs[m['map_name']];p=public(q,a3);assert np.allclose(p['bounds'],state['public']['bounds']) and np.allclose(p['grid_xy'],state['public']['grid_xy']);p['pose_world']=state['public']['pose_world'];language=parse(m['instruction'],g)
            key=m['map_name']
            if key not in geometries:
                b=np.array(p['bounds']);geometries[key]=Geometry(g,b[:2]+p['grid_xy']*(b[2:]-b[:2]))
            geom=geometries[key];versions={};cover={}
            geom_result=prepare(g,m['instruction'],p,language=language,geometry=geom);versions['geometry']=tensorize(geom_result)
            for mode in ('real','black','shuffled'):
                ev,cov=evidence(g,language,state,features,templates,pixels,mode);r=prepare(g,m['instruction'],p,visual=ev,language=language,geometry=geom);versions[mode]=tensorize(r);cover[mode]=cov
                if mode=='real':
                    hsv={i:values*np.array([0,1],np.float32) for i,values in ev.items()};r=prepare(g,m['instruction'],p,visual=hsv,language=language,geometry=geom);versions['hsv']=tensorize(r)
                    onlyref={i:(values if i!=0 else np.zeros_like(values)) for i,values in ev.items()};r=prepare(g,m['instruction'],p,visual=onlyref,language=language,geometry=geom);versions['references_only']=tensorize(r)
                    onlytarget={i:(values if i==0 else np.zeros_like(values)) for i,values in ev.items()};r=prepare(g,m['instruction'],p,visual=onlytarget,language=language,geometry=geom);versions['target_only']=tensorize(r)
                    sig={i:values*np.array([1,0],np.float32) for i,values in ev.items()};r=prepare(g,m['instruction'],p,visual=sig,language=language,geometry=geom);versions['siglip']=tensorize(r)
            prepared.append(dict(metadata=m,query_index=q['query_index'],a0=a3['logits'][q['query_index']],versions=versions,coverage=cover))
            # Labels remain in a separate sealed payload; never given to prepare.
            lab=features['plan']['labels'][m['sample_id']];labels.append(dict(sample_id=m['sample_id'],target_object_id=str(lab['target_object_id']),target_world=lab['target_world'],source='original weak referring identity; no human attribute labels'))
            if j%64==0:print('observed',split,j,round(time.time()-start,1),flush=True)
        torch.save(prepared,dest/f'{split}_public.pt');torch.save(labels,dest/f'{split}_labels.pt');stats[split]=dict(families=len(prepared),public_sha=sha(dest/f'{split}_public.pt'),labels_sha=sha(dest/f'{split}_labels.pt'),feature_sha=manifest['feature_sha256'])
        del features
    save(dest/'complete.json',dict(stats=stats,seconds=time.time()-start,backbones_frozen=True,candidate_protocol='all target objects; identical observed support for controls; matched proposal algorithm; Unknown mass retained',causality_checked=True));print('OBSERVED COMPLETE',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
