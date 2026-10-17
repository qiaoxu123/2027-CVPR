"""Fixed-controller latent-distribution audit on predeclared 128-query subsets."""
import json,time,csv
import torch
import numpy as np
from torch.nn import functional as F
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.geombelief_map_experiments import OUT,OLD,load_store
from scripts.geombelief_results import map_paths,csvwrite
from scripts import polygon_ablation_engine as engine
from graphnav.geombelief_inputs import ARMS

@torch.no_grad()
def collect(model,store,ids,arm):
    top=[];dense=[];entropy=[]
    for start in range(0,len(ids),16):
        part=torch.tensor(ids[start:start+16])
        x,t,m,n,y,size=store.base.batch(part,'descriptor_only','full_polygon') if arm=='correct' else store.batch(part,'map_geometry',arm)
        with torch.autocast('cuda',dtype=torch.bfloat16):field=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
        top.append(field.candidate_context[:,0].float().cpu());dense.append(field.dense_context.float().cpu())
        p=field.belief_logits.float().softmax(-1);entropy.append((-(p*p.clamp_min(1e-10).log()).sum(-1)).cpu())
    return torch.cat(top),torch.cat(dense),torch.cat(entropy)

def main():
    while not (OUT/'all_static_complete.json').exists():time.sleep(30)
    plan=json.loads((OUT/'frozen_plan.json').read_text());torch.set_num_threads(1);engine.OUT=OUT
    refpath=plan['sources']['reference']['path'];ref=engine.model_for(plan)
    ref.load_state_dict(torch.load(refpath,map_location='cpu',weights_only=True)['model'],strict=True);ref.eval()
    result=[];samples=[]
    for split in ('val_seen','val_unseen'):
        store=load_store(plan,split);ids=np.linspace(0,store.n-1,128,dtype=int).tolist()
        rt,rd,re=collect(ref,store,ids,'correct')
        for arm in ('correct',*ARMS):
            for seed in (0,1,2):
                _,_,cp=map_paths(arm,seed,split,'localization');model=engine.model_for(plan)
                model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
                top,dense,entropy=collect(model,store,ids,arm)
                same=(dense-rd).norm(dim=-1).mean(-1)/rd.norm(dim=-1).mean(-1).clamp_min(1e-9)
                cosine=F.cosine_similarity(top,rt,dim=-1)
                result.append(dict(split=split,arm=arm,seed=seed,queries=len(ids),context_mean=float(top.mean()),context_std=float(top.std()),top1_context_norm=float(top.norm(dim=-1).mean()),top1_context_cosine_to_head_training=float(cosine.mean()),same_grid_context_relative_L2=float(same.mean()),belief_entropy=float(entropy.mean()),reference_checkpoint_sha256=file_sha256(refpath),checkpoint_sha256=file_sha256(cp)))
                for j,i in enumerate(ids):samples.append(dict(split=split,arm=arm,seed=seed,query_id=store.metadata[i]['query_id'],map_name=store.metadata[i]['map_name'],top1_cosine=float(cosine[j]),same_grid_relative_L2=float(same[j])))
                del model;torch.cuda.empty_cache()
        del store;torch.cuda.empty_cache()
    csvwrite('controller_context_distribution.csv',result);atomic_json(samples,OUT/'controller_context_samples.json')
    atomic_json(dict(status='Done',samples_per_split=128,selection='Uniform fixed index subset, no outcome selection',reference=refpath,limitation='Latent differences are descriptive; neural coordinates are not identifiable across independent training. Raw cosine/L2 quantify the actual fixed head input shift, not semantic inferiority or calibrated performance.'),OUT/'controller_context_audit.json')

if __name__=='__main__':main()
