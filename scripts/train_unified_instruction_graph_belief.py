"""Full TRAIN experiment: original Unified V2 + optional 2-layer graph Belief.

Use the ALREADY TRAINED complete-instruction Unified V2 checkpoint. DO NOT
retrain BERT, original Geometry, Relation or PGVB in this first diagnostic.
Train only the new 2-layer graph -> candidate/Polygon residual on ALL 21,878
static TRAIN tasks; no RGB/Val/Test/closed-loop here.

Birmingham is part of TRAIN here; report is a fitting diagnostic only, not
an independent held-out validation result. Use paired Val-Seen closed-loop
only after leakage/scope/graph value checks on TRAIN.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,Subset

from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.full_relation_training import FORMAT,load_full_relation_model,full_relation_inputs
from graphnav.static_localization import cross_entropy_goal
from graphnav.token_language import FrozenBERTTokens
from graphnav.unified_instruction_graph_belief import UnifiedInstructionGraphBelief
from graphnav.unified_instruction_graph_data import (
    graph_input_batch,read_graphs,
)

GRAPH_FORMAT="unified-v2-graph-belief-residual-v1"


class GraphIndexedDataset(Dataset):
    def __init__(self,base):
        self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        return dict(**self.base[index],graph_index=int(index))


def build_source(train_path,checkpoint):
    cfg=checkpoint["config"]["geometry_args"]
    dataset=FullCandidateDataset(
        train_path,grid_size=int(cfg["grid_size"]),
        max_landmarks=int(cfg["max_landmarks"]),
        max_roads=int(cfg["max_roads"]),
        max_relations=int(cfg["max_relations"]),
        prior_sigma_m=float(cfg["prior_sigma_m"]),
        rule_sigma_m=float(cfg["rule_sigma_m"]),
        deduplicate_queries=False)
    if len(dataset)!=21878:
        raise ValueError("Full CityNav TRAIN must contain all 21,878 tasks")
    if {r.get("dataset_version") for r in dataset.records}!={"revised"}:
        raise ValueError("Only Revised TRAIN is supported")
    return dataset


def forward_pair(core,head,bert,batch,graphs,records,device):
    # Important: a frozen Unified V2 core caches Geometry between inference
    # calls. Every different TRAIN query minibatch needs a cache reset.
    core.reset_episode()
    with torch.no_grad():
        encoded=full_relation_inputs(batch,bert,device)
        baseline=core.forward_full_relation(**encoded,observed=None)
    source_ids=batch["graph_index"].tolist()
    graph_batch=graph_input_batch(
        [graphs[i] for i in source_ids],
        [records[i] for i in source_ids],bert,device,
        entity_capacity=encoded["static_inputs"]["pair_geometry"].shape[2])
    field=baseline["relation_field"]
    result=head(
        packed=graph_batch["packed"],
        node_text=graph_batch["node_text"],
        edge_text=graph_batch["edge_text"],
        dense_context=field.dense_context.detach(),
        pair_geometry=encoded["static_inputs"]["pair_geometry"].detach(),
        node_anchor=graph_batch["node_anchor"],
        base_logits=field.belief_logits.detach(),
    )
    return field.belief_logits.detach().float(),result


def distances(pred,batch):
    q=pred.argmax(-1)
    grid=batch["grid_xy"].to(pred.device).float()
    candidate=grid.gather(1,q[:,None,None].expand(-1,1,2)).squeeze(1)
    target=batch["target_xy"].to(pred.device).float()
    sizes=batch["map_sizes"].to(pred.device).float()
    return ((candidate-target)*sizes).norm(dim=-1)


def _counter():
    return dict(N=0,hit_base=0,hit_graph=0,ne_base=0.,ne_graph=0.,
                fixes=0,breaks=0,changed=0,supported=0)


def _accumulate(counter,base,refined,supported,batch):
    e0=distances(base,batch).detach()
    e1=distances(refined,batch).detach()
    g0=e0<=20.;g1=e1<=20.
    counter["N"]+=len(e0)
    counter["hit_base"]+=int(g0.sum());counter["hit_graph"]+=int(g1.sum())
    counter["ne_base"]+=float(e0.sum());counter["ne_graph"]+=float(e1.sum())
    counter["fixes"]+=int((~g0&g1).sum());counter["breaks"]+=int((g0&~g1).sum())
    counter["changed"]+=int((base.argmax(-1)!=refined.argmax(-1)).sum())
    counter["supported"]+=int(supported.sum())


def _finalize(counter):
    n=max(1,counter["N"])
    return dict(N=counter["N"],R20_base=counter["hit_base"]/n,
                R20_graph=counter["hit_graph"]/n,
                NE_base_m=counter["ne_base"]/n,NE_graph_m=counter["ne_graph"]/n,
                fixes=counter["fixes"],breaks=counter["breaks"],
                net=counter["fixes"]-counter["breaks"],
                changed_top1=counter["changed"],graph_supported=counter["supported"])


@torch.no_grad()
def evaluate(core,head,bert,base,graphs,device,batch_size):
    head.eval()
    selection=[i for i,r in enumerate(base.records)
               if r["map_name"].lower().startswith("birmingham_")]
    if not selection:
        raise ValueError("No Birmingham TRAIN queries")
    loader=DataLoader(Subset(GraphIndexedDataset(base),selection),
        batch_size=batch_size,shuffle=False,collate_fn=collate_full)
    stats={name:_counter() for name in
           ("all","direct","indirect","both","no_target_landmark_path")}
    for batch in loader:
        src_indices=batch["graph_index"].tolist()
        baseline,res=forward_pair(core,head,bert,batch,graphs,
                                  base.records,device)
        scores=res["logits"]
        # Graph input is WEAK parser output, not a verified classification.
        types=[]
        for i in src_indices:
            paths=graphs[i].get("paths",[])
            direct=any(p.get("hops")==1 for p in paths)
            indirect=any(p.get("hops",0)>1 for p in paths)
            types.append("both" if direct and indirect else "direct"
                         if direct else "indirect" if indirect
                         else "no_target_landmark_path")
        _accumulate(stats["all"],baseline,scores,res["supported"],batch)
        for cat in stats:
            if cat=="all":continue
            ids=[j for j,t in enumerate(types) if t==cat]
            if not ids:continue
            idx=torch.tensor(ids,device=device)
            selected=dict(grid_xy=batch["grid_xy"].index_select(0,idx.cpu()),
                          target_xy=batch["target_xy"].index_select(0,idx.cpu()),
                          map_sizes=batch["map_sizes"].index_select(0,idx.cpu()))
            _accumulate(stats[cat],baseline.index_select(0,idx),
                        scores.index_select(0,idx),
                        res["supported"].index_select(0,idx),selected)
    return {k:_finalize(v) for k,v in stats.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl",type=Path,required=True)
    p.add_argument("--graph-jsonl",type=Path,required=True)
    p.add_argument("--existing-full-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--kl-weight",type=float,default=.05)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-batches",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.seed!=42 or a.epochs<1 or a.batch_size<1 or a.max_batches<0:
        raise ValueError("Invalid deterministic TRAIN settings")
    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires local CityNav GPU")
    if a.output.exists() and not a.resume:
        raise FileExistsError("Do not overwrite an existing experiment")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda")
    src=torch.load(a.existing_full_checkpoint,map_location="cpu",weights_only=True)
    if (src.get("format")!=FORMAT or src.get("smoke")
            or src.get("completed_epochs",0)!=src.get("planned_epochs",0)
            or src.get("completed_epochs",0)<1):
        raise ValueError("Use a COMPLETE existing Unified V2 full-text checkpoint")
    if src["source_sha256"]["train_jsonl"]!=file_sha256(a.train_jsonl):
        raise ValueError("Baseline checkpoint and TRAIN source do not match")
    graph_meta=json.loads(a.graph_jsonl.with_suffix(".manifest.json").read_text())
    if (graph_meta["source_sha256"]!=file_sha256(a.train_jsonl) or
            graph_meta["graph_sha256"]!=file_sha256(a.graph_jsonl) or
            graph_meta["target_GT_used"] is not False):
        raise ValueError("Graph provenance/content mismatch")
    base=build_source(a.train_jsonl,src)
    graphs=read_graphs(a.graph_jsonl,base.records)
    core=load_full_relation_model(src,device).eval().requires_grad_(False)
    head=UnifiedInstructionGraphBelief(hidden=128,layers=2).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=.01)
    cfg=src["config"]["geometry_args"]
    bert=FrozenBERTTokens(cfg["text_backbone"],device=device,
                          max_length=int(cfg["max_text_length"]))
    cache=a.output/"bert_cache.pt"
    if a.resume and cache.exists():bert.load_cache(cache)
    manifest=dict(protocol=GRAPH_FORMAT,seed=42,epochs=a.epochs,batch_size=a.batch_size,
        lr=a.lr,kl_weight=a.kl_weight,n=len(base),max_batches=a.max_batches,
        full_source_sha256=file_sha256(a.existing_full_checkpoint),
        source_sha256=file_sha256(a.train_jsonl),graphs_sha256=file_sha256(a.graph_jsonl),
        model_sha256=file_sha256("graphnav/unified_instruction_graph_belief.py"),
        data_sha256=file_sha256("graphnav/unified_instruction_graph_data.py"),
        train_code_sha256=file_sha256(__file__),
        core_frozen=True,BERT_frozen=True,RGB_disabled=True,GT_forward=False,
        graph_is_weak_unreviewed=True,val_seen_unaccessed=True)
    if a.resume:
        path=a.output/"manifest.json"
        if not path.exists() or json.loads(path.read_text())!=manifest:
            raise ValueError("Refuse mismatched resume data/code/config")
        state=torch.load(a.output/"last.pt",map_location="cpu",weights_only=True)
        head.load_state_dict(state["adapter"]);optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start=int(state["epoch"]);history=state["history"]
    else:
        atomic_json(manifest,a.output/"manifest.json")
        start=0;history=[]
    for epoch in range(start,a.epochs):
        order=np.random.default_rng(42+epoch).permutation(len(base)).tolist()
        loader=DataLoader(Subset(GraphIndexedDataset(base),order),
                          batch_size=a.batch_size,shuffle=False,
                          collate_fn=collate_full,num_workers=0)
        head.train()
        seen=used=0;sum_loss=0.
        for batch_idx,batch in enumerate(loader):
            if a.max_batches and batch_idx>=a.max_batches:break
            baseline,result=forward_pair(core,head,bert,batch,graphs,
                                         base.records,device)
            target=batch["target_xy"].to(device).float()
            loss_ce=cross_entropy_goal(result["logits"],target)
            loss_kl=F.kl_div(F.log_softmax(result["logits"],-1),
                F.softmax(baseline,-1),reduction="batchmean")
            loss=loss_ce+a.kl_weight*loss_kl
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite graph loss")
            if loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                used+=int(result["supported"].sum())
            seen+=len(target);sum_loss+=float(loss.detach())*len(target)
            if batch_idx%200==0:
                print(json.dumps(dict(epoch=epoch+1,batch=batch_idx,seen=seen,
                    loss=float(loss.detach()),supported=int(result["supported"].sum()))),
                    flush=True)
        if not a.max_batches and seen!=len(base):
            raise RuntimeError(f"TRAIN query coverage {seen}!={len(base)}")
        history.append(dict(epoch=epoch+1,train_seen=seen,
                            supported_steps=used,mean_loss=sum_loss/max(1,seen)))
        atomic_save(dict(format=GRAPH_FORMAT,epoch=epoch+1,adapter=head.state_dict(),
                         optimizer=optimizer.state_dict(),manifest=manifest,
                         history=history,torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all()),
                    a.output/"last.pt")
        atomic_json(history,a.output/"history.json")
        bert.save_cache(cache)
        print(json.dumps(history[-1]),flush=True)
    if not a.max_batches:
        scores=evaluate(core,head,bert,base,graphs,device,a.batch_size)
        atomic_json(dict(manifest=manifest,fit_diagnostic=scores,
            note="Birmingham subset is part of full TRAIN; NOT a held-out split or navigation SR",
            baseline_retrained=False),a.output/"report.json")
        print(json.dumps(scores,indent=2),flush=True)


if __name__=="__main__":
    main()
