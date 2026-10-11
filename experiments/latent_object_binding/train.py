"""Full Train target-supervised latent matching, no reference labels/backbones."""
import argparse,json,time
import numpy as np,torch
from experiments.semantic_polygon_grounding.audit import save,sha
from .resources import OUT
from .model import Matcher
from .inference import posterior
from .evaluate import materialize,full_eval


def overrides(split,version):
    data=torch.load(OUT/'observed'/f'{split}_public.pt',weights_only=True,mmap=True);chosen={}
    for d in data:
        i=d['query_index']
        if i not in chosen or d['metadata']['cutoff_step']>chosen[i]['metadata']['cutoff_step']:chosen[i]=d
    return {i:d['versions'][version] for i,d in chosen.items()}


def run(name,resume=False):
    start=time.time();torch.manual_seed(20261011);np.random.seed(20261011);torch.set_num_threads(1);device='cuda';dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=True)
    model=Matcher(rgb=name=='rgb').cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);maps=torch.load(OUT/'full_geometry/maps.pt',weights_only=True);manifest=json.loads((OUT/'full_geometry/complete.json').read_text());labels={l['query_index']:l for l in torch.load(OUT/'full_geometry/labels.pt',weights_only=True)}
    for m in maps.values():m['kernels']=m['kernels'].cuda()
    train_extra=overrides('train','real' if name=='rgb' else 'geometry');dev_extra=overrides('dev','real' if name=='rgb' else 'geometry');history=[];first=1;best=-1.;best_epoch=None
    if resume:
        ck=torch.load(dest/'last.pt',weights_only=True,map_location='cuda');model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);first=ck['epoch']+1;history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch']
    elif (dest/'last.pt').exists():raise RuntimeError('use --resume instead of overwriting')
    for epoch in range(first,4):
        rng=np.random.default_rng(20261011+epoch);order=rng.permutation(len(manifest['chunks']));total=0.;n=0;object_correct=0;goal_correct=0;batch_count=0;optimizer.zero_grad();epochstart=time.time()
        for chunk_index in order:
            data=torch.load(OUT/'full_geometry'/manifest['chunks'][int(chunk_index)],weights_only=True,mmap=True)
            train=[d for d in data if d['split']=='train']
            for j in rng.permutation(len(train)):
                row=train[int(j)];label=labels[row['query_index']];selected=train_extra.get(row['query_index'],row);p=materialize(selected,maps[row['map_name']],row['a0'],device);r=posterior(p,model.weights());positive=torch.tensor(label['positive_objects'],device=device);cells=label['positive_cells'].to(device)
                objloss=-torch.log(r['object_belief'][positive].sum().clamp(min=1e-30));goalloss=-torch.log(r['heatmap'][cells].sum().clamp(min=1e-30));loss=objloss+.5*goalloss
                if not torch.isfinite(loss):raise ValueError('nonfinite matching loss')
                (loss/32).backward();total+=float(loss.detach());n+=1;batch_count+=1;object_correct+=int(r['object_belief'].argmax().item() in label['positive_objects']);goal_correct+=int(cells[r['heatmap'].argmax()].item())
                if batch_count==32:torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad();batch_count=0
            del data
        if batch_count:
            # Correct scaling of final short batch, same effective budget.
            for param in model.parameters():
                if param.grad is not None:param.grad*=32/batch_count
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad()
        with torch.no_grad():dev_rows,dev_metrics=full_eval(model.weights().detach(),methods=('LearnedGeom',),split='dev',device=device,write=False,overrides=dev_extra)
        score=dev_metrics['LearnedGeom']['goal_r1']['mean'];stats=dict(epoch=epoch,loss=total/max(n,1),train_queries=n,train_online_object_r1=object_correct/max(n,1),train_online_goal_r1=goal_correct/max(n,1),dev=dev_metrics['LearnedGeom'],seconds=time.time()-epochstart,weights=model.weights().detach().cpu().tolist(),peak_gpu_bytes=torch.cuda.max_memory_allocated());history.append(stats)
        if score>best:best=score;best_epoch=epoch;torch.save(dict(model=model.state_dict(),weights=model.weights().detach().cpu(),epoch=epoch,seed=20261011,name=name,dev_metric=score),dest/'best.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,history=history,best=best,best_epoch=best_epoch),dest/'last.pt');save(dest/'history.json',history);print('epoch',name,epoch,'loss',round(stats['loss'],4),'devR1',round(score,5),'seconds',round(stats['seconds'],1),flush=True)
        if time.time()-start>900 and epoch<3:raise TimeoutError('15min training budget; preserve checkpoint and report partial pilot')
    save(dest/'complete.json',dict(name=name,epochs=len(history),best_epoch=best_epoch,best_dev_goal_r1=best,seconds=time.time()-start,train_queries=18895,train_observed_query_overrides=len(train_extra),dev_queries=2285,dev_observed_query_overrides=len(dev_extra),parameters=9,effective_parameters=9 if name=='rgb' else 7,all_backbones_frozen=True,reference_instance_labels_used=False,checkpoint_sha=sha(dest/'best.pt'),known_mass_gain_unchanged=True,history=history));print('TRAIN COMPLETE',name,round(time.time()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['geometry','rgb'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.name,a.resume)
