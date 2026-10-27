"""Real informative TRAIN-contrast CUDA backward gate, separate from full training."""
import argparse
from pathlib import Path
import torch
from .paths import ROOT,save,sha
from .model import LandmarkBeliefNetwork,supervision_loss
from .batching import collate_views,to_device


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=ROOT/'runs/a2_waldo_landmark_20261027/cache_smoke')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(0)
    rows=[];paths=[]
    signature=sha(args.cache/'signature.json')
    for path in sorted(args.cache.glob('train_*.pt')):
        row=torch.load(path,map_location='cpu',weights_only=False)
        if row['signature_sha256']!=signature:
            raise ValueError('Smoke cache signature mismatch')
        y=row['labels']
        if (y['ranking']>0).any() and (y['rank_label_mask'] & (y['ranking']<=0)).any() and row['public']['network']['link_valid'].any():
            rows.append(row);paths.append(str(path))
        if len(rows)>=4:
            break
    if not rows:
        raise ValueError('No informative TRAIN contrast: auxiliary-only smoke cannot authorize ranking training')
    batch=to_device(collate_views(rows),'cuda')
    model=LandmarkBeliefNetwork().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    out=model(**batch['network'])
    loss,stats=supervision_loss(out,batch['labels'],refine=True)
    optimizer.zero_grad();loss.backward()
    gradients={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(value>0 for name,value in gradients.items() if name.startswith('scorer.rank_head'))
    assert any(value>0 for name,value in gradients.items() if name.startswith('language.'))
    assert any(value>0 for name,value in gradients.items() if name.startswith('match.'))
    optimizer.step()
    save(dict(status='informative_train_cuda_backward_passed',scope='One optimizer smoke step; not formal 20+5 training',
              source_rows=paths,signature_sha256=signature,loss=float(loss.detach()),statistics=stats,
              gradients=gradients,max_cuda_memory_gb=torch.cuda.max_memory_allocated()/2**30),
         ROOT/'reports/a2_execution_20261027/informative_cuda_smoke.json')
    print('Informative TRAIN CUDA backward passed',stats,flush=True)


if __name__=='__main__':
    main()
