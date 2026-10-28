"""Real TRAIN-only revision CUDA backward and measured iteration timing."""
import json,time
from pathlib import Path
import torch
from .paths import save,sha
from .collected_snapshot import BASE
from .revision_data import RevisionViews
from .revision_model import RevisionNetwork
from .revision_loss import revision_loss
from .revision_train import code_hashes
from .batching import collate_views,to_device


def meaningful_view(row):
    y=row['labels'];return bool((y['ranking']>0).any() and (y['rank_label_mask'] & (y['ranking']<=0)).any()
                                 and y['candidate_trust'].any() and row['public']['network']['link_valid'].any())


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    dataset=RevisionViews(BASE/'collected_snapshot','train');rows=[];names=[]
    for i,e in enumerate(dataset.entries):
        row=dataset[i]
        if meaningful_view(row):rows.append(row);names.append(e['filename'])
        if len(rows)==8:break
    if len(rows)<4:raise ValueError('Insufficient real revision TRAIN contrasts')
    model=RevisionNetwork().cuda().train();batch=to_device(collate_views(rows),'cuda')
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=3e-4)
    torch.cuda.synchronize();start=None
    for iteration in range(6):
        if iteration==2:torch.cuda.synchronize();start=time.perf_counter()
        out=model(**batch['network']);loss,stats=revision_loss(out,batch,refine=True)
        optimizer.zero_grad();loss.backward()
        if not torch.isfinite(loss):raise ValueError('Nonfinite revision CUDA loss')
        grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite revision gradients')
        for prefix in ('base.language.','base.scorer.rank_head.','rank_context.','trust_context.','benefit.','category.'):
            if not any(v>0 for n,v in grads.items() if n.startswith(prefix)):raise ValueError(f'Missing revision gradient {prefix}')
        optimizer.step()
    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/4
    report=dict(status='revision_real_train_cuda_backward_passed',device=torch.cuda.get_device_name(0),
                source_rows=names,cache_signature_sha256=sha(BASE/'collected_snapshot/signature.json'),
                reserve_sha256=dataset.reserve_sha256,code_sha256=code_hashes(),loss=float(loss.detach()),statistics=stats,
                gradients=grads,seconds_per_batch8=seconds,smoke_steps=6,measured_steps=4,batch_boxes=int(batch['network']['box_rgb'].shape[1]),formal_training=False,
                max_cuda_memory_gb=torch.cuda.max_memory_allocated()/2**30)
    save(report,BASE/'revision_cuda_smoke.json');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
