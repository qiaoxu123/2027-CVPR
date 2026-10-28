"""Reproducible real-TRAIN CPU backward check, never a CUDA/training gate."""
from collections import Counter
import json,time
import torch
from experiments.a2_waldo_landmark_belief.relation_v3_data import RelationV3Views
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from experiments.a2_waldo_landmark_belief.relation_v3_loss import relation_v3_loss
from experiments.a2_waldo_landmark_belief.batching import collate_views
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.paths import ROOT,sha,save


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(0)
    dataset=RelationV3Views('train');rows=[];counts=Counter();identities=[]
    for i,entry in enumerate(dataset.entries):
        source=entry['source_kind']
        if counts[source]>=2:continue
        row=dataset[i];y=row['labels'];pos=y['localization_utility']&y['utility_valid'];neg=~pos&y['utility_valid']
        if pos.any() and neg.any():
            rows.append(row);counts[source]+=1
            identities.append({k:entry[k] for k in ('source_kind','filename','sha256','episode','step')})
        if len(rows)==6:break
    if len(rows)!=6 or len(counts)!=3:raise ValueError('Two informative TRAIN rows from each source family required')
    batch=collate_views(rows);model=RelationV3Network();parent=BASE/'revision_v1/training/best.pt'
    if sha(parent)!='b5f7ae68659d1550c77e6d1cb4c32447433e9c2d459b2e4b2ac221ef45a2fc9d':raise ValueError('Frozen parent changed')
    model.warmstart(torch.load(parent,map_location='cpu',weights_only=False))
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=.0003);history=[]
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss,stats=relation_v3_loss(model,model(**batch['network']),batch,refine=step>=3)
        if not torch.isfinite(loss):raise ValueError('Nonfinite real TRAIN CPU loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),5.,error_if_nonfinite=True)
        if not norm>0:raise ValueError('No real TRAIN gradient')
        if step>=3 and not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.risk.parameters()):raise ValueError('No real risk gradient')
        optimizer.step();history.append(dict(step=step,refine=step>=3,loss=float(loss),gradient_norm=float(norm),stats=stats))
    report=dict(status='real_TRAIN_CPU_backward_only_not_CUDA_or_formal_training',sources=dict(counts),rows=6,updates=6,
        history=history,observations=identities,roles_sha256=dataset.roles_sha256,parent_sha256=sha(parent),
        code_sha256={n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in ('relation_v3_data.py','relation_v3_model.py','relation_v3_fusion.py','relation_v3_loss.py','batching.py')},
        smoke_script_sha256=sha(ROOT/'scripts/a2_relation_v3_cpu_smoke.py'),seconds=time.perf_counter()-start)
    save(report,BASE/'revision_v3/real_cpu_backward.json')
    save(report,ROOT/'reports/a2_execution_20261027/RELATION_V3_CPU_BACKWARD.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
