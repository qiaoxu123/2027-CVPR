"""Resumable GeomBelief map grounding experiment. Archived runs are read-only."""
import json, os, sys, time, fcntl, shutil, subprocess, traceback
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.geombelief_inputs import ARMS,SPATIAL,mask_geometry,shuffle_manifest
from scripts import polygon_ablation_engine as engine
from scripts.polygon_ablation_plan import OUT as OLD, HERE,canonical_job

OUT=HERE/'reports/paper_experiments/geombelief_remaining'

def init():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((OLD/'frozen_plan.json').read_text())
    assert not plan['args'].get('relative_position',False)
    for p,h in plan['input_sha256'].items():assert file_sha256(p)==h,p
    plan.update(experiment='GeomBelief language-grounded map geometry',parent_commit='34d297425ac5c48c036055af62f325f20126964b',new_arms=list(ARMS),correct_geometry_reuse=str(OLD),
        candidate_position='Common absolute 28x28 lattice; relative_position=False, no landmark-dependent positional encoding',
        shuffle='Fixed 20261017 permutation of entire spatial bundles within split/TrainDev partition/map/entity-mask/node-type schema; names/language stay recipient. Singleton strata remain unchanged and are reported. Exact spatial marginals, including rule fields, retained; tests spatial grounding jointly, not only landmark-name binding.',
        distance='unsigned centroid Euclidean distance only, pair channel2; no signed boundary/inside/extent',direction='unit centroid-to-candidate vector only, pair channels3,4; no dx/dy magnitude',
        all_geometry_baseline='Correct retains all original geometry paths; simple arms zero every other spatial path. This tests total geometry utility, not just pair descriptors.',
        navigation_decision='After all static results: evaluate all five new arms, all three seeds on Seen128/Unseen128 then full splits regardless of outcome; fixed transfer head. Correct reuses archived identical arm.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if (OUT/'frozen_plan.json').exists():
        saved=json.loads((OUT/'frozen_plan.json').read_text());plan['source_commit']=saved['source_commit'];assert saved==plan
    else:atomic_json(plan,OUT/'frozen_plan.json')
    for name in ('train_dev_manifest.json','val_seen_query_manifest.json','val_unseen_query_manifest.json','train_seen_query_manifest.json','navigation_data_audit.json'):
        dst=OUT/name
        if dst.exists():assert file_sha256(dst)==file_sha256(OLD/name)
        else:shutil.copyfile(OLD/name,dst)
    return plan

class ControlledStore:
    def __init__(self,base,split):
        self.base=base;self.metadata=base.metadata;self.n=base.n
        members=json.loads((OUT/'train_dev_manifest.json').read_text())
        partitions=[r['split'] for r in members] if split=='train_seen' else [split]*self.n
        d=base.tensors
        manifest=shuffle_manifest(base.metadata,d['node_types'].cpu(),d['node_valid'].cpu(),d['entity_valid'].cpu(),partitions)
        path=OUT/f'{split}_shuffle_manifest.json'
        if path.exists():assert json.loads(path.read_text())==manifest
        else:atomic_json(manifest,path)
        self.donors=torch.tensor([r['donor_index'] for r in manifest],device='cuda')
    def batch(self,ids,scope,rep):
        x,t,m,n,y,size=self.base.batch(ids,'descriptor_only','full_polygon')
        if rep=='shuffled':
            other,*_=self.base.batch(self.donors[ids.to('cuda')],'descriptor_only','full_polygon')
            x=dict(x,**{k:other[k] for k in SPATIAL})
        else:x=mask_geometry(x,rep)
        return x,t,m,n,y,size

def load_store(plan,split):
    # Read validated pre-existing geometry/BERT caches; never create/modify OLD.
    saved=engine.OUT;engine.OUT=OLD
    try:
        p=OLD/f'.cache/{split}_geometry.pt';payload=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        assert payload['signature']['input_sha256']==file_sha256(plan['paths'][split])
        assert payload['signature']['geometry_source_sha256']==file_sha256(HERE/'graphnav/polygon_ablation.py')
        return ControlledStore(engine.Store(payload,plan),split)
    finally:engine.OUT=saved

def main():
    plan=init();lock=(OUT/'map_run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engine.OUT=OUT;torch.set_num_threads(1)
    atomic_json(dict(pid=os.getpid(),command=sys.argv,started=time.time()),OUT/'map_process.json')
    if (OUT/'map_complete.json').exists():return
    train=load_store(plan,'train_seen')
    for seed in (0,1,2):
        reference=json.loads((OLD/'training'/f'shared_full_polygon__full_polygon__seed{seed}'/'complete.json').read_text())
        for arm in ARMS:
            engine.train(plan,train,'map_geometry',arm,seed)
            done=json.loads((OUT/'training'/canonical_job('map_geometry',arm,seed)/'complete.json').read_text())
            assert done['initial_state_sha256']==reference['initial_state_sha256']
            oldhist=json.loads((OLD/'training'/f'shared_full_polygon__full_polygon__seed{seed}'/'history.json').read_text())
            hist=json.loads((OUT/'training'/canonical_job('map_geometry',arm,seed)/'history.json').read_text())
            assert [h['data_order_sha256'] for h in hist]==[h['data_order_sha256'] for h in oldhist]
    del train;torch.cuda.empty_cache()
    atomic_json(dict(status='Done',new_training_runs=15,correct_reused=3),OUT/'all_training_frozen.json')
    for split in ('val_seen','val_unseen'):
        store=load_store(plan,split)
        for seed in (0,1,2):
            for arm in ARMS:
                job=canonical_job('map_geometry',arm,seed);folder=OUT/'localization'/job;folder.mkdir(parents=True,exist_ok=True)
                cp=OUT/'training'/job/'best.pt';dest=folder/f'{split}.json'
                if dest.exists():continue
                model=engine.model_for(plan);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True)
                metrics=engine.evaluate(model,store,'map_geometry',arm,output=folder/f'{split}.localization.jsonl')
                atomic_json(dict(status='Done',arm=arm,seed=seed,split=split,checkpoint_sha256=file_sha256(cp),metrics=metrics),dest)
                del model;torch.cuda.empty_cache()
        del store;torch.cuda.empty_cache()
    atomic_json(dict(status='Done',new_conditions=15,correct_reused=3),OUT/'all_static_complete.json')
    from scripts.geombelief_map_navigation import run_all
    run_all(plan)
    atomic_json(dict(status='Done',new_training_runs=15,new_navigation_split_runs=30),OUT/'map_complete.json')

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);atomic_json(dict(status='Blocked',error=traceback.format_exc()),OUT/'map_failure.json');raise
