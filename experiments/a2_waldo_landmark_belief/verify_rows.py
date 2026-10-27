"""Real informative TRAIN-contrast CUDA backward gate, separate from full training."""
import argparse
from pathlib import Path
import torch
from .paths import ROOT,save,sha
from .model import LandmarkBeliefNetwork,supervision_loss
from .batching import collate_views,to_device


def informative_refinement_view(row):
    y=row['labels']
    return bool((y['ranking']>0).any() and (y['rank_label_mask'] & (y['ranking']<=0)).any()
                and row['public']['network']['link_valid'].any()
                and (y['offset_valid'] & y['rank_label_mask'] & (y['ranking']>0)).any())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=ROOT/'runs/a2_waldo_landmark_20261027/cache_smoke')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(0)
    rows=[];paths=[]
    signature=sha(args.cache/'signature.json')
    from .collected_snapshot import cache_source
    from .train import CachedViews
    import json
    if (args.cache/'manifest.json').exists():
        manifest=json.loads((args.cache/'manifest.json').read_text())
        dataset=CachedViews(args.cache,manifest,cache_source(args.cache),'train')
        iterator=((args.cache/entry['filename'],dataset[i]) for i,entry in enumerate(dataset.entries))
    else:
        iterator=((path,torch.load(path,map_location='cpu',weights_only=False)) for path in sorted(args.cache.glob('train_*.pt')))
    for path,row in iterator:
        if row['signature_sha256']!=signature:
            raise ValueError('Smoke cache signature mismatch')
        if informative_refinement_view(row):
            rows.append(row);paths.append(str(path))
        if len(rows)>=4:
            break
    if not rows:
        raise ValueError('No informative TRAIN contrast with reachable offset supervision: cannot authorize full training')
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
    assert any(value>0 for name,value in gradients.items() if name.startswith('offset.'))
    optimizer.step()
    save(dict(status='informative_train_cuda_backward_passed',scope='One optimizer smoke step; not formal 20+5 training',
              source_rows=paths,signature_sha256=signature,loss=float(loss.detach()),statistics=stats,
              device='cuda',model_config=model.config,
              preflight_code_sha256=sha(Path(__file__)),
              runtime_code_sha256={name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)
                                   for name in ('model.py','core.py','observed_geometry.py','batching.py')},
              gradients=gradients,max_cuda_memory_gb=torch.cuda.max_memory_allocated()/2**30),
         ROOT/'reports/a2_execution_20261027/informative_cuda_smoke.json')
    print('Informative TRAIN CUDA backward passed',stats,flush=True)


if __name__=='__main__':
    main()
