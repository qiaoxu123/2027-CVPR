"""Full TRAIN: teach an angle-only modality without retraining full Unified V2.

True goal XY appears ONLY in direction/goal LOSS and offline ORACLE DIAGNOSTIC.
Deployment uses the PREDICTED bearing distribution, never teacher bearings.

Original full-text BERT, Geometry+Relation, Public Polygon, PGVB and controller
are FROZEN. Angular head accepts only original BERT, public name embedding
and public candidate ANGLE. No range, dx/dy vector or target coordinate to
the new model forward. All 21878 Revised TRAIN tasks stay in denominator.

Birmingham is part of TRAIN, not independent validation. Oracle diagnostics
use privileged GT and MUST NOT be called a navigation result.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,Subset

from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.full_relation_training import FORMAT,full_relation_inputs,load_full_relation_model
from graphnav.static_localization import cross_entropy_goal
from graphnav.token_language import FrozenBERTTokens
from graphnav.unified_angle_only_belief import (
    AngleOnlyBelief,bearing_supervision,named_anchor_mask,public_candidate_bearings,
)

MODEL_FORMAT="unified-v2-predicted-angle-only-v1"


class IndexedDataset(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):return dict(**self.base[i],source_index=int(i))


def build_data(path,state,*,expected=21878):
    cfg=state["config"]["geometry_args"]
    ds=FullCandidateDataset(
        path,grid_size=int(cfg["grid_size"]),
        max_landmarks=int(cfg["max_landmarks"]),
        max_roads=int(cfg["max_roads"]),
        max_relations=int(cfg["max_relations"]),
        prior_sigma_m=float(cfg["prior_sigma_m"]),
        rule_sigma_m=float(cfg["rule_sigma_m"]),
        deduplicate_queries=False)
    if len(ds)!=expected or {r.get("dataset_version") for r in ds.records}!={"revised"}:
        raise ValueError("Requires ALL expected Revised TRAIN queries")
    if {r.get("match_mode") for r in ds.records}!={"fuzzy"}:
        raise ValueError("Use matched complete TRAIN source")
    return ds


def forward_frozen(core,head,bert,batch,records,device):
    """No GT access in this routine. All new geometry is angle-only."""
    core.reset_episode()
    with torch.no_grad():
        encoded=full_relation_inputs(batch,bert,device)
        baseline=core.forward_full_relation(**encoded,observed=None)
    field=baseline["relation_field"]
    pair=encoded["static_inputs"]["pair_geometry"]
    counts=torch.tensor([len(records[i]["landmarks"])
                         for i in batch["source_index"].tolist()],
                        device=device,dtype=torch.long)
    named=named_anchor_mask(encoded["static_inputs"]["entity_valid"],counts)
    candidate_angles=public_candidate_bearings(pair)
    x=dict(
        base_logits=field.belief_logits.detach(),
        candidate_bearings=candidate_angles.detach(),
        landmark_valid=named,
        tokens=encoded["tokens"].detach(),
        token_mask=encoded["token_mask"],
        node_emb=encoded["node_emb"].detach(),
    )
    out=head(**x)
    return x,out


def labels_after_forward(batch,static,head,device):
    """This function may read GT; it is NEVER called by runtime."""
    gt=batch["target_xy"].to(device)
    scale=batch["map_sizes"].to(device)
    centers=static["nodes"][:,:,:2]
    label,mask=bearing_supervision(
        gt,centers,scale,static["named_mask"],bins=head.bins)
    return label,mask


def target_error(logits,batch):
    ids=logits.argmax(-1)
    grids=batch["grid_xy"].to(logits.device).float()
    chosen=grids.gather(1,ids[:,None,None].expand(-1,1,2)).squeeze(1)
    target=batch["target_xy"].to(logits.device).float()
    extent=batch["map_sizes"].to(logits.device).float()
    return ((chosen-target)*extent).norm(-1)


def summarize_scores(storage):
    n=len(storage["baseline"])
    def stat(key):
        e=torch.cat(storage[key])
        return dict(R20=float((e<=20.).float().mean()),NE_m=float(e.mean()))
    original=stat("baseline");predicted=stat("predicted")
    oracle=stat("oracle_GT_direction_OFFLINE_ONLY")
    a=torch.cat(storage["baseline"])<=20.
    b=torch.cat(storage["predicted"])<=20.
    return dict(N=len(a),baseline=original,predicted=predicted,
                oracle_GT_direction_OFFLINE_ONLY=oracle,
                fixes=int((~a&b).sum()),breaks=int((a&~b).sum()),
                net=int((~a&b).sum()-(a&~b).sum()),
                note="Oracle reads target GT direction. Its R20 is a privileged diagnostic, not fair inference.")


@torch.no_grad()
def evaluate(core,head,bert,ds,device,batch_size,*,max_batches=0):
    head.eval()
    subset=[i for i,r in enumerate(ds.records)
            if r["map_name"].lower().startswith("birmingham_")]
    if not subset:raise ValueError("Missing Birmingham TRAIN subset")
    loader=DataLoader(Subset(IndexedDataset(ds),subset),
                      batch_size=batch_size,shuffle=False,
                      collate_fn=collate_full,num_workers=0)
    storage={x:[] for x in ("baseline","predicted","oracle_GT_direction_OFFLINE_ONLY")}
    direction_hit=direction_total=0
    supported=0;seen=0
    for step,batch in enumerate(loader):
        if max_batches and step>=max_batches:break
        x,out=forward_frozen(core,head,bert,batch,ds.records,device)
        static=dict(nodes=batch["nodes"].to(device),
                    named_mask=x["landmark_valid"])
        gt_bins,valid=labels_after_forward(batch,static,head,device)
        # Oracle evaluation is isolated here and consumes GT *only* offline.
        hard=F.one_hot(gt_bins,num_classes=head.bins).float()
        oracle=head.apply_angle(
            base_logits=x["base_logits"],
            candidate_bearings=x["candidate_bearings"],
            landmark_valid=x["landmark_valid"]&valid,
            bearing_probabilities=hard,
        )["logits"]
        errors=dict(
            baseline=target_error(x["base_logits"],batch),
            predicted=target_error(out["logits"],batch),
            oracle_GT_direction_OFFLINE_ONLY=target_error(oracle,batch),
        )
        for key,e in errors.items():storage[key].append(e.cpu())
        hits=(out["angle_logits"].argmax(-1)==gt_bins)&valid
        direction_hit+=int(hits.sum());direction_total+=int(valid.sum())
        supported+=int(out["supported"].sum());seen+=len(batch["text"])
    if not seen:raise ValueError("No evaluation tasks")
    report=summarize_scores(storage)
    report.update(direction_class_accuracy=direction_hit/max(1,direction_total),
                  direction_labeled_anchors=direction_total,
                  angular_supported_queries=supported,
                  evaluated_queries=seen)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl",type=Path,required=True)
    p.add_argument("--existing-full-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--direction-weight",type=float,default=.5)
    p.add_argument("--kl-weight",type=float,default=.05)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-batches",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.seed!=42 or a.epochs<1 or a.batch_size<1 or a.max_batches<0:
        raise ValueError("Seed=42 and positive TRAIN settings required")
    if not torch.cuda.is_available():
        raise RuntimeError("Run on existing CityNav GPU workstation")
    if a.output.exists() and not a.resume:
        raise FileExistsError("Refuse to overwrite existing run")
    a.output.mkdir(parents=True,exist_ok=True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    device=torch.device("cuda")
    source=torch.load(a.existing_full_checkpoint,map_location="cpu",weights_only=True)
    if (source.get("format")!=FORMAT or source.get("smoke") or
            source.get("completed_epochs",0)!=source.get("planned_epochs",0) or
            source.get("completed_epochs",0)<1):
        raise ValueError("Use completed previously trained full-text Unified V2")
    if source["source_sha256"]["train_jsonl"]!=file_sha256(a.train_jsonl):
        raise ValueError("Full-text baseline checkpoint trained on a different TRAIN inventory")
    ds=build_data(a.train_jsonl,source)
    core=load_full_relation_model(source,device).eval().requires_grad_(False)
    head=AngleOnlyBelief(bins=8).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=.01)
    cfg=source["config"]["geometry_args"]
    bert=FrozenBERTTokens(cfg["text_backbone"],device=device,
                          max_length=int(cfg["max_text_length"]))
    cache=a.output/"bert_cache.pt"
    if a.resume and cache.exists():bert.load_cache(cache)
    manifest=dict(
        protocol=MODEL_FORMAT,seed=42,epochs=a.epochs,batch_size=a.batch_size,
        lr=a.lr,direction_weight=a.direction_weight,kl_weight=a.kl_weight,
        max_batches=a.max_batches,n=len(ds),
        source_sha256=file_sha256(a.train_jsonl),
        baseline_sha256=file_sha256(a.existing_full_checkpoint),
        model_code_sha256=file_sha256("graphnav/unified_angle_only_belief.py"),
        train_code_sha256=file_sha256(__file__),
        new_modality=["predicted_bearing_8_bins"],
        forbidden=["target_GT_at_runtime","dx_dy_modality","distance_bin",
                   "bearing_magnitude","target_position_feature"],
        geometry_and_full_BERT_frozen=True,GT_only_in_loss=True,
        oracle_only_in_offline_evaluation=True)
    if a.resume:
        if json.loads((a.output/"manifest.json").read_text())!=manifest:
            raise ValueError("Refuse resume after model/data/config change")
        state=torch.load(a.output/"last.pt",map_location="cpu",weights_only=True)
        head.load_state_dict(state["head"]);opt.load_state_dict(state["optimizer"])
        start=int(state["epoch"]);history=state["history"]
        torch.set_rng_state(state["rng_cpu"]);torch.cuda.set_rng_state_all(state["rng_cuda"])
    else:
        atomic_json(manifest,a.output/"manifest.json")
        start=0;history=[]
    for epoch in range(start,a.epochs):
        indices=np.random.default_rng(42+epoch).permutation(len(ds)).tolist()
        loader=DataLoader(Subset(IndexedDataset(ds),indices),
                          batch_size=a.batch_size,shuffle=False,
                          collate_fn=collate_full,num_workers=0)
        head.train();seen=0;sampled_angles=0;total_loss=0.
        for j,batch in enumerate(loader):
            if a.max_batches and j>=a.max_batches:break
            x,out=forward_frozen(core,head,bert,batch,ds.records,device)
            static=dict(nodes=batch["nodes"].to(device),
                        named_mask=x["landmark_valid"])
            labels,valid=labels_after_forward(batch,static,head,device)
            if valid.any():
                direction_loss=F.cross_entropy(out["angle_logits"][valid],labels[valid])
            else:
                direction_loss=out["angle_logits"].sum()*0.
            goal_loss=cross_entropy_goal(out["logits"],batch["target_xy"].to(device))
            prior=F.kl_div(F.log_softmax(out["logits"],-1),
                           F.softmax(x["base_logits"],-1),reduction="batchmean")
            loss=goal_loss+a.direction_weight*direction_loss+a.kl_weight*prior
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite angular modality objective")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            seen+=len(batch["text"]);sampled_angles+=int(valid.sum())
            total_loss+=float(loss.detach())*len(batch["text"])
            if j%200==0:
                print(json.dumps(dict(epoch=epoch+1,batch=j,seen=seen,
                                      loss=float(loss.detach()),
                                      supervised_angles=sampled_angles,
                                      gain=float(head.fusion_gain.detach()))),flush=True)
        if not a.max_batches and seen!=len(ds):
            raise ValueError("Incomplete full TRAIN coverage")
        if sampled_angles==0:raise RuntimeError("No supervised public landmark bearings")
        history.append(dict(epoch=epoch+1,fit_seen=seen,
                            supervised_landmark_angles=sampled_angles,
                            loss=total_loss/max(1,seen),
                            learned_gain=float(head.fusion_gain.detach())))
        atomic_save(dict(format=MODEL_FORMAT,head=head.state_dict(),
                         epoch=epoch+1,optimizer=opt.state_dict(),
                         history=history,manifest=manifest,
                         rng_cpu=torch.get_rng_state(),
                         rng_cuda=torch.cuda.get_rng_state_all()),
                    a.output/"last.pt")
        atomic_json(history,a.output/"history.json")
        bert.save_cache(cache)
        print(json.dumps(history[-1]),flush=True)
    if not a.max_batches:
        report=dict(manifest=manifest,completed_epochs=a.epochs,
                    TRAIN_Birmingham_diagnostic=evaluate(core,head,bert,ds,device,
                                                          a.batch_size),
                    warning="Birmingham is part of TRAIN; Oracle is privileged; only predicted-bearing closedloop is fair.")
        atomic_json(report,a.output/"report.json")
        print(json.dumps(report["TRAIN_Birmingham_diagnostic"],indent=2),flush=True)
    bert.save_cache(cache)


if __name__=="__main__":
    main()
