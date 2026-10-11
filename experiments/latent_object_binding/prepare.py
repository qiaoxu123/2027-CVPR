"""Full public Train/Dev geometry preparation; GT joined only by the writer."""
import collections,concurrent.futures,json,multiprocessing,time
from pathlib import Path
import numpy as np,torch
from experiments.semantic_polygon_grounding.audit import save,sha
from .resources import OUT,baseline,public
from .inference import prepare
from .factors import Geometry
from .visual import template
GG=None;GE=None;T=None

def init():
    global GG,GE,T
    from experiments.language_object_graph.objects import ObjectGraph
    torch.set_num_threads(1);GG={p.stem:ObjectGraph.restore(torch.load(p,weights_only=True)) for p in Path('runs/language_object_graph/graphs').glob('*.pt')};GE={};T=time.time()

def worker(item):
    name,text,p=item;g=GG[name];key=(name,tuple(p['bounds']))
    if key not in GE:
        b=np.array(p['bounds']);GE[key]=Geometry(g,b[:2]+p['grid_xy']*(b[2:]-b[:2]))
    geom=GE[key]
    if len(geom.between_cache)>512:geom.between_cache.clear()
    r=prepare(g,text,p,geometry=geom)
    # Candidate/public data only. Labels are not accepted by this worker.
    kept={k:r[k] for k in ('target_features','reference_features','log_phi','log_weights','support_mass','confidence','known_mass','assignments','language','unknown','reference_posteriors')}
    for k in ('target_features','reference_features','log_phi','log_weights'):kept[k]=np.asarray(kept[k],dtype=np.float16)
    return kept

def main():
    start=time.time();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);dest=OUT/'full_geometry';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():raise RuntimeError('preserve completed full preparation')
    source_hashes={f:sha(Path(__file__).with_name(f)) for f in ('language.py','factors.py','inference.py')}
    a3,source,graphs,proof=baseline();queries=[q for q in source['queries'] if q['split'] in ('train','dev')];save(OUT/'protocol.json',json.loads(Path(__file__).with_name('protocol.json').read_text()));save(OUT/'a3_proof.json',proof)
    chunks=[];labels=[];templates=set();stats=collections.defaultdict(collections.Counter);graphmeta={}
    for name,g in graphs.items():
        # Bounds are taken from actual public query, not GT or target extents.
        q=next((q for q in queries if q['map_name']==name),None)
        if q is None:continue
        b=np.array(q['public_query']['world_bounds']);geom=Geometry(g,b[:2]+a3['grid_xy'].numpy()*(b[2:]-b[:2]));graphmeta[name]=dict(kernels=torch.tensor(geom.kernels),grid_xy=a3['grid_xy'],world=torch.tensor(geom.world),bounds=b.tolist(),ids=g.ids)
    torch.save(graphmeta,dest/'maps.pt');source_public=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        for offset in range(0,len(queries),64):
            qs=queries[offset:offset+64];path=dest/f'chunk_{offset:06d}.pt'
            if path.exists():
                with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,np.dtypes.Float64DType,np.dtypes.Int64DType]):prepared=torch.load(path,weights_only=True,map_location='cpu')
                for p in prepared:
                    for key in ('support_mass','confidence','known_mass'):p[key]=float(p[key])
                torch.save(prepared,path)
            else:
                tasks=[(q['map_name'],q['instruction'],public(q,a3)) for q in qs];prepared=list(pool.map(worker,tasks,chunksize=4))
            batch=[]
            for q,p in zip(qs,prepared):
                g=graphs[q['map_name']]
                for key in ('target_features','reference_features','log_phi','log_weights'):
                    if not torch.is_tensor(p[key]):p[key]=torch.from_numpy(p[key])
                p.update(query_index=q['query_index'],map_name=q['map_name'],split=q['split'],instruction=q['instruction'],family_ids=q['family_ids'],a0=a3['logits'][q['query_index']]);batch.append(p)
                variables=p['language'].get('variables',p['language'].get('entities',[]));templates.update(template(e) for e in variables)
                positive=list(dict.fromkeys(g.byid[str(x['target_object_id'])] for x in q['labels']));xy=np.array([x['target_world'] for x in q['labels']]);world=graphmeta[q['map_name']]['world'].numpy();distance=np.linalg.norm(world[:,None]-xy[None],axis=-1).min(1);cells=distance<=20
                if not cells.any():raise ValueError('GT has no unchanged success-radius candidate; report rather than change protocol')
                labels.append(dict(query_index=q['query_index'],positive_objects=positive,positive_cells=torch.tensor(cells),offline_labels=q['labels']))
                stats[q['split']]['queries']+=1;stats[q['split']]['latent_assignments']+=len(p['assignments']);stats[q['split']]['zero_known_mass']+=int(p['known_mass']<1e-8);stats[q['split']]['multi_positive_queries']+=int(len(positive)>1)
                source_public.append(dict(query_index=q['query_index'],split=q['split'],map_name=q['map_name'],instruction=q['instruction'],family_ids=q['family_ids']))
            if not path.exists():torch.save(batch,path)
            chunks.append(path.name)
            if offset%640==0:print('full_geometry',offset,len(queries),'seconds',round(time.time()-start,1),flush=True)
            if time.time()-start>900:raise TimeoutError('predeclared 15min preparation budget; preserve partial results')
    assert source_hashes=={f:sha(Path(__file__).with_name(f)) for f in source_hashes}, 'inference code changed during preparation'
    torch.save(labels,dest/'labels.pt');save(dest/'queries_public.json',source_public);save(OUT/'templates.json',sorted(templates));save(dest/'complete.json',dict(chunks=chunks,stats={s:dict(c) for s,c in stats.items()},seconds=time.time()-start,queries=len(queries),candidate_recall=1.,nodes=sum(len(g.ids) for g in graphs.values()),edges=sum(g.edge_index.shape[1] for g in graphs.values()),map_sha=sha(dest/'maps.pt'),labels_sealed_sha=sha(dest/'labels.pt'),protocol_sha=sha(Path(__file__).with_name('protocol.json')),reference_instance_truth=None,public_whitelist=True,source_hashes=source_hashes));print('PREPARED',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
