"""Bounded cached-feature head training, internal Dev checkpoint selection only."""
import argparse
import json
from pathlib import Path
import time
import torch
from .evaluate_features import load,metrics
from .head import VisualEvidenceNetwork,grounding_loss
from graphnav.checkpointing import file_sha256


def collate(rows,representation,language):
    b=len(rows);c=max(len(r['public']['roi_valid']) for r in rows);d=rows[0]['texts'][language].shape[-1]
    words=max(len(r['language_tokens']) for r in rows);patches=rows[0]['patch_tokens'].shape[1]
    public=dict(text=torch.stack([r['texts'][language] for r in rows]),images=torch.zeros(b,c,d),patch_tokens=torch.zeros(b,c,patches,d),language_tokens=torch.zeros(b,words,d),language_valid=torch.zeros(b,words,dtype=torch.bool),quality=torch.zeros(b,c,3),valid=torch.zeros(b,c,dtype=torch.bool))
    pos=torch.zeros(b,c,dtype=torch.bool);judged=pos.clone()
    for i,r in enumerate(rows):
        n=len(r['public']['roi_valid']);l=len(r['language_tokens']);public['images'][i,:n]=r['features'][representation];public['patch_tokens'][i,:n]=r['patch_tokens'];public['language_tokens'][i,:l]=r['language_tokens'];public['language_valid'][i,:l]=r['language_valid'];public['quality'][i,:n]=r['public']['roi_quality'];public['valid'][i,:n]=r['public']['roi_valid'];pos[i,:n]=r['labels']['positive_roi_mask'];judged[i,:n]=r['labels']['semantic_valid_mask']
    return public,pos,judged


@torch.inference_mode()
def evaluate(model,rows,representation,language,device):
    records=[];model.eval()
    for off in range(0,len(rows),8):
        chunk=rows[off:off+8];public,_,_=collate(chunk,representation,language)
        evidence=model({k:v.to(device) for k,v in public.items()})
        for i,r in enumerate(chunk):
            n=len(r['public']['roi_valid']);m=r['metadata'];value=metrics(evidence.score[i,:n].cpu(),r)
            records.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],direction=m['direction'],metrics=value,score=evidence.score[i,:n].cpu().tolist(),confidence=evidence.confidence[i,:n].cpu().tolist()))
    return records


def run(args):
    output=Path(args.output)
    if output.exists():raise ValueError('refusing checkpoint overwrite')
    output.mkdir(parents=True);started=time.time();torch.manual_seed(args.seed);device=torch.device(args.device)
    train,dev=load(args.train),load(args.dev)
    if {r['metadata']['family_id'] for r in train}&{r['metadata']['family_id'] for r in dev}:
        raise ValueError('training/development identity leakage')
    dimension=train[0]['texts'][args.language].shape[-1]
    model=VisualEvidenceNetwork(dimension=dimension,token_attention=args.token_attention).to(device)
    if args.initialize:
        state=torch.load(args.initialize,map_location='cpu',weights_only=True)
        if state['dimension']!=dimension:raise ValueError('initial head dimension mismatch')
        if state['token_attention']==args.token_attention:
            model.load_state_dict(state['model'],strict=True)
        elif args.token_attention and not state['token_attention']:
            missing,unexpected=model.load_state_dict(state['model'],strict=False)
            allowed=('image_tokens.','word_tokens.','attention.','token_score.')
            if unexpected or any(not key.startswith(allowed) for key in missing):raise ValueError('unexpected initialized projection mismatch')
            # New zero-initialized token score preserves the pretrained scorer
            # initially; no claim that missing attention weights were trained.
        else:raise ValueError('initial head architecture mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    best=-1;history=[];steps=0
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch));losses=[];epoch_start=time.time()
        for off in range(0,len(order),args.batch_size):
            rows=[train[j] for j in order[off:off+args.batch_size]];public,positive,judged=collate(rows,args.representation,args.language)
            public={k:v.to(device) for k,v in public.items()};positive=positive.to(device);judged=judged.to(device)
            evidence=model(public);loss=grounding_loss(evidence,positive,judged,public['quality'])
            if not torch.isfinite(loss['total']):raise ValueError('nonfinite real training loss')
            optimizer.zero_grad(set_to_none=True);loss['total'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite actual gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();steps+=1;losses.append(float(loss['total'].detach()))
        predictions=evaluate(model,dev,args.representation,args.language,device)
        # Predeclared forward language semantics; reverse geometry is secondary.
        eligible=[r['metrics']['forward_r1'] for r in predictions if r['metrics']['forward_r1'] is not None]
        selection=sum(eligible)/len(eligible) if eligible else sum(r['metrics']['r1'] for r in predictions)/len(predictions)
        item=dict(epoch=epoch,steps=steps,loss=sum(losses)/len(losses),internal_dev_selection_r1=selection,seconds=time.time()-epoch_start,dev_semantic_states=len(eligible));history.append(item)
        state=dict(model={k:v.cpu() for k,v in model.state_dict().items()},dimension=dimension,token_attention=args.token_attention,epoch=epoch,steps=steps,representation=args.representation,language=args.language,source=dict(train_manifest_sha256=file_sha256(Path(args.train)/'manifest.json'),dev_manifest_sha256=file_sha256(Path(args.dev)/'manifest.json')))
        torch.save(state,output/f'epoch{epoch:02}.pt')
        if selection>best:
            best=selection;torch.save(state,output/'best_dev.pt')
            with (output/'best_dev_predictions.jsonl').open('w') as h:
                for r in predictions:h.write(json.dumps(r)+'\n')
        print(json.dumps(item),flush=True)
    result=dict(complete=True,epochs=args.epochs,steps=steps,training_states=len(train),internal_dev_states=len(dev),trainable_parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,backbones_trained=False,selection='Train internal Dev forward semantic R1, first highest tie',args=vars(args),history=history,best_checkpoint_sha256=file_sha256(output/'best_dev.pt'))
    (output/'training_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--dev',required=True);p.add_argument('--output',required=True);p.add_argument('--representation',default='polygon_masked');p.add_argument('--language',default='phrase');p.add_argument('--token-attention',action='store_true');p.add_argument('--initialize');p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda');run(p.parse_args())


if __name__=='__main__':main()
