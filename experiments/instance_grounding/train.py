"""Three epochs maximum, frozen feature backbones, internal-Dev selection."""
import argparse,json,time
from pathlib import Path
import torch
from experiments.bidirectional_core.stats import summarize_rows
from graphnav.checkpointing import file_sha256
from .model import InstanceVEN,loss
from .dataset import city_rows,external_rows,collate,training_unit
from .data import write

@torch.inference_mode()
def score(model,rows,device='cuda',control='real'):
    records=[];model.eval()
    for off in range(0,len(rows),4):
        chunk=rows[off:off+4];p,_,_=collate(chunk,control);p={k:v.to(device) for k,v in p.items()};s,prob,tri=model(p)
        for i,r in enumerate(chunk):
            n=len(r['public']['roi_valid']);records.append(dict(score=s[i,:n].cpu(),prob=prob[i,:n].cpu(),tri=tri[i,:n].cpu()))
    return records

def evaluate(rows,predictions,identity):
    families={}
    for r,p in zip(rows,predictions):
        lab=r['labels'];pos=lab['identity_positive'] if identity else lab['positive_roi_mask'];valid=r['public']['roi_valid']
        # City primary gallery ALL actual observed objects. External attribute
        # protocol retains judged-only gallery; these are different objectives.
        eligible=valid if identity else (lab['semantic_valid_mask']&valid)
        if not (pos&eligible).any() or eligible.sum()<2:continue
        best=p['score'].masked_fill(~eligible,-torch.inf).argmax();f=r['metadata']['family_id'];families.setdefault(f,[]).append(float(pos[best]))
    return sum(sum(v)/len(v) for v in families.values())/len(families) if families else 0,len(families)

def run(a):
    output=Path(a.output)
    if output.exists():raise ValueError('no overwrite')
    output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(20261011);start=time.time()
    reader=city_rows if a.city else external_rows;train=reader(a.train);dev=reader(a.dev)
    if a.city:
        assert not ({r['metadata']['map_name'] for r in train}&{r['metadata']['map_name'] for r in dev});train=training_unit(train)
    else:assert not ({r['metadata']['family_id'] for r in train}&{r['metadata']['family_id'] for r in dev})
    model=InstanceVEN(relative_attention=a.city).to('cuda')
    if a.initialize:
        state=torch.load(a.initialize,weights_only=True,map_location='cpu')
        if 'instance_model' in state:
            missing,unexpected=model.load_state_dict(state['instance_model'],strict=False)
            if unexpected or any(not k.startswith('relation_coefficients.') for k in missing):raise ValueError('unexpected initialized architecture mismatch')
        else:model.base.load_state_dict(state['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);best=-1;history=[];steps=0
    for epoch in range(1,4):
        model.train();order=torch.randperm(len(train));losses=[]
        for off in range(0,len(order),8):
            batch=[train[j] for j in order[off:off+8]];p,pos,judged=collate(batch);p={k:v.to('cuda') for k,v in p.items()};pos=pos.cuda();judged=judged.cuda()
            s,prob,tri=model(p);value=loss(s,tri,pos,judged,p['valid'],p['quality'],class_balanced=a.city)
            # Clearly corrupted evidence teaches abstention; no claim that a
            # shuffled actual image is a semantic negative without its labels.
            zp,_,_=collate(batch,'zero');z={k:v.to('cuda') for k,v in zp.items()}
            _,_,ztri=model(z)
            if p['valid'].any():value=value+.05*torch.nn.functional.cross_entropy(ztri[p['valid']],torch.full((int(p['valid'].sum()),),2,dtype=torch.long,device='cuda'))
            if not torch.isfinite(value):raise ValueError('nonfinite real loss')
            optimizer.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();steps+=1;losses.append(float(value.detach()))
        preds=score(model,dev);metric,n=evaluate(dev,preds,a.city);item=dict(epoch=epoch,steps=steps,loss=sum(losses)/len(losses),dev_family_r1=metric,dev_eligible_families=n,seconds=time.time()-start);history.append(item)
        state=dict(instance_model={k:v.cpu() for k,v in model.state_dict().items()},epoch=epoch,seed=20261011,relative_attention=a.city,train_manifest_sha256=file_sha256(Path(a.train)/'manifest.json'),dev_manifest_sha256=file_sha256(Path(a.dev)/'manifest.json'))
        torch.save(state,output/f'epoch{epoch:02}.pt')
        if metric>best:best=metric;torch.save(state,output/'best_dev.pt')
        print(json.dumps(item),flush=True)
    write(output/'training_complete.json',dict(complete=True,epochs=3,steps=steps,training_units=len(train),dev_rows=len(dev),trainable_parameters=sum(p.numel() for p in model.parameters()),parameters_with_gradient=sum(p.numel() for p in model.parameters() if p.grad is not None),seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),backbones_frozen=True,class_balanced_judgment=a.city,relative_attention=a.city,history=history,checkpoint_sha256=file_sha256(output/'best_dev.pt'),args=vars(a),selection='family-balanced Train internal Dev original identity' if a.city else 'external multi-positive judged-region matching, not CityNav identity',unknown_policy='absent target Unknown; ambiguous candidates Unknown; corrupted evidence Unknown',unknown_training_control='actual encoded black pixels' if a.city else 'zero feature sentinel (external cache has no black embeddings)'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--dev',required=True);p.add_argument('--output',required=True);p.add_argument('--initialize');p.add_argument('--city',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
