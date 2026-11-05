"""Train frozen-Geometry/Relation + candidate-specific Polygon-language attention.

NO RGB, no language pseudo-labels, no target GT in model forward. Revised TRAIN
only. Fit Cambridge, evaluate Birmingham TRAIN-Dev exactly once after last
epoch. Frozen backbone previously trained on ALL TRAIN, so this is a probe
of the new residual's held-out-map fitting, NOT whole-system unseen-map proof.

Three matched Dev arms: frozen static, real language, zeroed language, and
cross-query swapped language. All include identical frozen geometry/Polygon.
Attention weights are diagnostics, NOT proof of token causality.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset

from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.polygon_language_attention import PolygonGuidedLanguageAttention
from graphnav.static_localization import cross_entropy_goal
from graphnav.token_language import FrozenBERTTokens

FORMAT="polygon-guided-language-attention-v1"


def make_dataset(path,geometry_args):
    ds=FullCandidateDataset(
        path,grid_size=28,
        max_landmarks=int(geometry_args["max_landmarks"]),
        max_roads=int(geometry_args["max_roads"]),
        max_relations=int(geometry_args["max_relations"]),
        prior_sigma_m=float(geometry_args["prior_sigma_m"]),
        rule_sigma_m=float(geometry_args["rule_sigma_m"]),
        deduplicate_queries=False)
    if ({r.get("dataset_version") for r in ds.records}!={"revised"} or
            {r.get("match_mode") for r in ds.records}!={"fuzzy"}):
        raise ValueError("Expected complete Revised/fuzzy TRAIN only")
    fit=[i for i,r in enumerate(ds.records) if r.get("map_name","").startswith("cambridge_")]
    dev=[i for i,r in enumerate(ds.records) if r.get("map_name","").startswith("birmingham_")]
    if len(fit)+len(dev)!=len(ds):
        raise ValueError("Unexpected map not assigned to Cambridge/Birmingham")
    if not fit or not dev:
        raise ValueError("Must have nonempty map-disjoint TRAIN-Fit/Dev")
    return ds,fit,dev


def frozen_features(model,batched,bert,device):
    with torch.no_grad():
        inputs=full_relation_inputs(batched,bert,device)
        result=model.forward_full_relation(**inputs,relation_enabled=True)
    base=result["relation_field"].belief_logits.float().detach()
    dense=result["geometry_field"].dense_context.float().detach()
    pair=inputs["static_inputs"]["pair_geometry"].float()
    return dict(
        base_logits=base,dense_context=dense,
        pair_geometry=pair,
        entity_valid=inputs["static_inputs"]["entity_valid"].bool(),
        node_emb=inputs["node_emb"].float(),
        tokens=inputs["tokens"].float(),
        token_mask=inputs["token_mask"].bool(),
    )


@torch.no_grad()
def paired_score(base,variant,batch):
    grid=batch["grid_xy"].to(base.device).float()
    gt=batch["target_xy"].to(base.device).float()
    size=batch["map_sizes"].to(base.device).float()
    ix=torch.stack((base.argmax(-1),variant.argmax(-1)),-1)
    pred=grid.gather(1,ix[...,None].expand(-1,-1,2))
    ne=torch.linalg.vector_norm((pred-gt[:,None])*size[:,None],dim=-1)
    return ne.cpu().numpy()


def summarize(arm):
    count=arm["n"]
    return dict(n=count,R20=arm["hit"]/max(1,count),
                NE_m=arm["ne"]/max(1,count),
                fixes=arm["fix"],breaks=arm["breaks"],
                net=arm["fix"]-arm["breaks"],
                changed_top1=arm["changed"])


def accumulate(stats,ne,changed):
    before=ne[:,0]<=20
    after=ne[:,1]<=20
    stats["n"]+=len(ne);stats["hit"]+=int(after.sum())
    stats["ne"]+=float(ne[:,1].sum())
    stats["fix"]+=int((~before&after).sum())
    stats["breaks"]+=int((before&~after).sum())
    stats["changed"]+=int(changed)


@torch.no_grad()
def evaluate(model,adapter,dataset,indices,bert,device,batch_size,max_batches=0):
    model.eval();adapter.eval()
    rng=np.random.default_rng(42)
    loader=DataLoader(Subset(dataset,indices),batch_size=batch_size,
                      shuffle=False,num_workers=0,collate_fn=collate_full)
    names=("static","real","zero_text","swapped_text")
    stats={x:dict(n=0,hit=0,ne=0.,fix=0,breaks=0,changed=0) for x in names}
    explanation=[]
    for num,batch in enumerate(loader):
        if max_batches and num>=max_batches:
            break
        x=frozen_features(model,batch,bert,device)
        base=x["base_logits"]
        outputs={"static":base}
        out=adapter(**x,return_attention=len(explanation)<20)
        outputs["real"]=out["logits"]
        outputs["zero_text"]=adapter(**{**x,"tokens":torch.zeros_like(x["tokens"])})["logits"]
        if len(batch["text"])>1:
            shifted=torch.arange(len(batch["text"]),device=device).roll(1)
            outputs["swapped_text"]=adapter(**{**x,
                "tokens":x["tokens"].index_select(0,shifted),
                "token_mask":x["token_mask"].index_select(0,shifted)})["logits"]
        else:
            outputs["swapped_text"]=base
        for name,pred in outputs.items():
            ne=paired_score(base,pred,batch)
            accumulate(stats[name],ne,int((pred.argmax(-1)!=base.argmax(-1)).sum()))
        if "token_weights" in out and len(explanation)<20:
            top=base.topk(min(5,base.shape[-1]),dim=-1).indices
            top_weights=out["token_weights"].gather(
                1,top[...,None].expand(-1,-1,out["token_weights"].shape[-1])
            ).mean(1).cpu()
            for i,text in enumerate(batch["text"]):
                ids=bert.tokenizer(
                    text,truncation=True,max_length=bert.max_length)["input_ids"]
                names_of_tokens=bert.tokenizer.convert_ids_to_tokens(ids)
                size=min(len(names_of_tokens),top_weights.shape[1])
                words=[(names_of_tokens[k],float(top_weights[i,k]))
                       for k in range(size) if names_of_tokens[k] not in
                       ("[CLS]","[SEP]","[PAD]")]
                words.sort(key=lambda z:z[1],reverse=True)
                explanation.append(dict(
                    instruction=text,
                    top_weighted_tokens=[dict(token=w,attention=round(v,5))
                                         for w,v in words[:8]],
                    note="Attention weight is NOT causal phrase importance"))
                if len(explanation)>=20:
                    break
    return dict(arms={name:summarize(x) for name,x in stats.items()},
                example_token_attention=explanation,
                evaluated_queries=stats["real"]["n"],
                privileged_images_used=False,
                benchmark_split="TRAIN-Dev only")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl",type=Path,required=True)
    p.add_argument("--relation-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--kl-weight",type=float,default=.1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-train-batches",type=int,default=0)
    p.add_argument("--max-dev-batches",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<2 or a.max_train_batches<0 or a.max_dev_batches<0:
        raise ValueError("Invalid training/eval controls")
    if not torch.cuda.is_available():
        raise RuntimeError("Use local CityNav GPU environment")
    if a.output.exists() and not a.resume:
        raise FileExistsError("Refusing to overwrite an existing experiment")
    a.output.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4)
    device=torch.device("cuda")
    ckpt=torch.load(a.relation_checkpoint,map_location="cpu",weights_only=True)
    if ckpt.get("format")!="unified-pgvb-full-relation-v1" or ckpt.get("completed_epochs",0)<1:
        raise ValueError("Expected completed frozen full-relation checkpoint")
    model=load_full_relation_model(ckpt,device).eval().requires_grad_(False)
    args=ckpt["config"]["geometry_args"]
    ds,fit,dev=make_dataset(a.train_jsonl,args)
    bert=FrozenBERTTokens(args["text_backbone"],device=device,
                          max_length=int(args["max_text_length"]))
    if a.resume and (a.output/"bert_cache.pt").exists():
        bert.load_cache(a.output/"bert_cache.pt")
    adapter=PolygonGuidedLanguageAttention().to(device)
    opt=torch.optim.AdamW(adapter.parameters(),lr=a.lr,weight_decay=.01)
    manifest=dict(protocol=FORMAT,source_sha=file_sha256(a.relation_checkpoint),
        train_sha=file_sha256(a.train_jsonl),fit_queries=len(fit),dev_queries=len(dev),
        fit_maps=sorted({ds.records[i]["map_name"] for i in fit}),
        dev_maps=sorted({ds.records[i]["map_name"] for i in dev}),
        seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,
        kl_weight=a.kl_weight,max_train_batches=a.max_train_batches,
        max_dev_batches=a.max_dev_batches,
        model_code_sha=file_sha256("graphnav/polygon_language_attention.py"),
        script_sha=file_sha256(__file__),
        GT_in_forward=False,relation_geometry_frozen=True,
        special_tokens_excluded_from_attention=True,
        full_val_seen_access=False,
        core_pretrained_on_all_TRAIN=True,
        smoke=bool(a.max_train_batches or a.max_dev_batches))
    exists=a.output/"manifest.json"
    if a.resume:
        if not exists.exists() or json.loads(exists.read_text())!=manifest:
            raise ValueError("Cannot resume changed inputs/code/checkpoint")
        state=torch.load(a.output/"last.pt",map_location="cpu",weights_only=True)
        adapter.load_state_dict(state["adapter"])
        opt.load_state_dict(state["optimizer"])
        start=int(state["epoch"])
        history=state["history"]
    else:
        atomic_json(manifest,exists);start=0;history=[]
    for epoch in range(start,a.epochs):
        # Deterministic per-epoch training subset permutations (seed 42 only).
        order=np.random.default_rng(a.seed+epoch).permutation(fit).tolist()
        loader=DataLoader(Subset(ds,order),batch_size=a.batch_size,
                          shuffle=False,num_workers=0,collate_fn=collate_full)
        adapter.train();seen=steps=0;loss_sum=0.
        for batch_no,batch in enumerate(loader):
            if a.max_train_batches and batch_no>=a.max_train_batches:break
            x=frozen_features(model,batch,bert,device)
            out=adapter(**x)
            gold=batch["target_xy"].to(device)
            task_loss=cross_entropy_goal(out["logits"],gold)
            preserve=F.kl_div(F.log_softmax(out["logits"],-1),
                              x["base_logits"].softmax(-1),reduction="batchmean")
            loss=task_loss+a.kl_weight*preserve
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Polygon-language loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,
                                               error_if_nonfinite=True)
            opt.step()
            seen+=len(batch["text"]);steps+=1;loss_sum+=float(loss.detach())*len(batch["text"])
            if batch_no%250==0:
                print(json.dumps(dict(epoch=epoch+1,batch=batch_no,seen=seen,
                                      loss=float(loss.detach()),grad_norm=float(norm))),
                      flush=True)
        if not a.max_train_batches and seen!=len(fit):
            raise RuntimeError(f"TRAIN-Fit coverage incomplete {seen}/{len(fit)}")
        history.append(dict(epoch=epoch+1,fit_seen=seen,steps=steps,
                            train_loss=loss_sum/max(1,seen)))
        atomic_save(dict(format=FORMAT,epoch=epoch+1,adapter=adapter.state_dict(),
                         optimizer=opt.state_dict(),manifest=manifest,history=history),
                    a.output/"last.pt")
        atomic_json(history,a.output/"history.json")
        bert.save_cache(a.output/"bert_cache.pt")
        print(json.dumps(history[-1]),flush=True)
    report=evaluate(model,adapter,ds,dev,bert,device,a.batch_size,
                    max_batches=a.max_dev_batches)
    report.update(manifest=manifest,completed_epochs=a.epochs,
                  warning=("Smoke-only; not full TRAIN-Dev" if manifest["smoke"]
                           else "TRAIN-Dev, not Val/Test, no navigation SR"))
    atomic_json(report,a.output/"report.json")
    bert.save_cache(a.output/"bert_cache.pt")
    print(json.dumps(report["arms"],indent=2),flush=True)


if __name__=="__main__":
    main()
