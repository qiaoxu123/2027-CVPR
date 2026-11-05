"""Fixed TRAIN-only privileged PGVB experiment; foreign checkpoint cannot navigate."""
import argparse, json, random, time, hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256, rng_state, restore_rng
from graphnav.full_relation_training import load_full_relation_model
from graphnav.complete_landmark_rgb import refine_views

FORMAT='unified-pgvb-privileged-visual-train-v1'

class Cache:
    def __init__(self,root):self.root=Path(root)
    @lru_cache(maxsize=96)
    def group(self,key):return torch.load(self.root/'groups'/f'{key}.pt',map_location='cpu',weights_only=True)['observed']
    def batch(self,ids,arm):
        rows=[torch.load(self.root/'queries'/f'{i:05d}.pt',map_location='cpu',weights_only=True) for i in ids]
        packs=[]
        for row in rows:
            p=row['current'] if arm=='current' else self.group(row['group']) if row['group'] else None
            if p is not None and arm=='whole':p={k:v[:1] for k,v in p.items()}
            packs.append(p)
        return torch.stack([r['base'] for r in rows]).cuda(),torch.stack([r['dense'] for r in rows]).cuda().float(),packs


def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ('inventory','cache','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--resume',action='store_true');p.add_argument('--smoke-updates',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);np.random.seed(42);a.output.mkdir(parents=True,exist_ok=True)
    complete=json.load(open(a.cache/'complete.json')) if (a.cache/'complete.json').exists() else {}
    assert complete.get('complete') or a.smoke_updates,'All-TRAIN cache must finish before full training'
    qs=json.load(open(a.inventory/'queries_public.json'));labels={x['index']:x for x in json.load(open(a.inventory/'fit_labels_private.json'))};groups=json.load(open(a.inventory/'groups.json'))
    ids=[q['index'] for q in qs if q['subset']=='fit'];assert len(ids)==14259
    if a.smoke_updates:
        ids=[i for i in ids if (a.cache/'queries'/f'{i:05d}.pt').exists()]
        assert ids,'No fit-city cache available yet; refuse empty smoke'
    fingerprint={k:file_sha256(v) for k,v in dict(cache=a.cache/'manifest.json',source=a.checkpoint,inventory=a.inventory/'manifest.json',code=Path(__file__),refine=Path('graphnav/complete_landmark_rgb.py')).items()}
    config=dict(seed=42,epochs=a.epochs,batch_size=a.batch_size,arm='multi',pgvb_lr=3e-5,joint_lr=1e-4,weight_decay=.01,clip=1.,preservation_KL=.2,target_sigma_m=15.,supervision_support_radius_m=20.,selection='final epoch, no Dev selection',total_logit_cap=.5)
    model=load_full_relation_model(torch.load(a.checkpoint,map_location='cpu',weights_only=True));model.requires_grad_(False)
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(('pgvb.','joint_logits.')))
    initial_frozen=state_digest({k:v for k,v in model.state_dict().items() if not k.startswith(('pgvb.','joint_logits.'))})
    initial_train=state_digest({k:v for k,v in model.state_dict().items() if k.startswith(('pgvb.','joint_logits.'))})
    optimizer=torch.optim.AdamW([dict(params=model.pgvb.parameters(),lr=3e-5),dict(params=model.joint_logits.parameters(),lr=1e-4)],weight_decay=.01)
    epoch=cursor=step=effective_updates=supported_examples=0;history=[];start=time.perf_counter();last=a.output/'last.pt'
    if a.resume and last.exists():
        s=torch.load(last,map_location='cpu',weights_only=True);assert s['fingerprint']==fingerprint and s['config']==config
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);restore_rng(s['rng']);epoch=s['epoch'];cursor=s['cursor'];step=s['step'];history=s['history'];effective_updates=s['effective_updates'];supported_examples=s['supported_examples']
    cache=Cache(a.cache);grid=(torch.stack(torch.meshgrid(torch.arange(28),torch.arange(28),indexing='ij'),-1)[...,[1,0]].reshape(-1,2).cuda()+.5)/28
    atomic_json(dict(config=config,fingerprint=fingerprint,trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),frozen_SHA256=initial_frozen,fit_queries=len(ids),Dev_used_for_selection=False),a.output/'config.json')
    def save(e,c):
        atomic_save(dict(format=FORMAT,privileged=True,normal_navigation_allowed=False,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),epoch=e,cursor=c,step=step,effective_updates=effective_updates,supported_examples=supported_examples,history=history,fingerprint=fingerprint,config=config),last)
    for e in range(epoch,a.epochs):
        order=ids.copy();random.Random(42+e).shuffle(order);model.train();model.geometry.eval();model.full_relation.eval()
        for c in range(cursor if e==epoch else 0,len(order),a.batch_size):
            chosen=order[c:c+a.batch_size];base,dense,packs=cache.batch(chosen,'multi');optimizer.zero_grad(set_to_none=True)
            logits,covered=refine_views(model,base,dense,packs,joint=True)
            target=torch.tensor([labels[i]['target_normalized'] for i in chosen],device='cuda');bounds=torch.tensor([groups[qs[i]['group']]['world_bounds'] if qs[i]['group'] else [0,0,1,1] for i in chosen],device='cuda');size=bounds[:,2:]-bounds[:,:2]
            dist=((grid[None]-target[:,None])*size[:,None]).square().sum(-1);soft=(-dist/(2*15**2)).softmax(-1)
            supported=(covered & (dist<=20**2)).any(-1);ce=-(soft*logits.log_softmax(-1)).sum(-1)
            fit=(ce*supported).sum()/supported.sum().clamp_min(1);kl=F.kl_div(logits.log_softmax(-1),base.softmax(-1),reduction='batchmean');loss=fit+.2*kl
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            # No supported observations => differentiable zero, never forced fake positives.
            if covered.any() and loss.requires_grad:
                loss.backward();gn=torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.)
                pg=sum(float(x.grad.norm()) for x in model.pgvb.parameters() if x.grad is not None);jg=sum(float(x.grad.norm()) for x in model.joint_logits.parameters() if x.grad is not None)
                optimizer.step();effective_updates+=1
            else:gn=pg=jg=0.
            step+=1;supported_examples+=int(supported.sum())
            if step==1 or step%100==0:
                torch.cuda.synchronize();row=dict(epoch=e+1,step=step,loss=float(loss.detach()),target_CE=float(fit.detach()),KL=float(kl.detach()),supported=int(supported.sum()),gradient=float(gn),pgvb_gradient=pg,joint_gradient=jg,seconds=time.perf_counter()-start,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30);history.append(row);print(json.dumps(row),flush=True)
            if step%500==0:save(e,c+len(chosen))
            if a.smoke_updates and step>=a.smoke_updates:save(e,c+len(chosen));atomic_json(dict(smoke=True,step=step,history=history),a.output/'complete.json');return
        save(e+1,0);cursor=0
    assert initial_frozen==state_digest({k:v for k,v in model.state_dict().items() if not k.startswith(('pgvb.','joint_logits.'))}), 'Frozen modules changed'
    changed=initial_train!=state_digest({k:v for k,v in model.state_dict().items() if k.startswith(('pgvb.','joint_logits.'))})
    assert changed and step>0,'No parameter update; refuse training success'
    atomic_json(dict(complete=True,step=step,effective_updates=effective_updates,supported_examples=supported_examples,epochs=a.epochs,seconds=time.perf_counter()-start,weights_changed=changed,frozen_SHA256=initial_frozen,checkpoint_SHA256=file_sha256(last),history=history),a.output/'complete.json')

if __name__=='__main__':main()
