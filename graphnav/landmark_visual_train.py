"""P2 weakly-supervised landmark RGB pretraining with SigLIP2 LARGE patches.

Prerequisites:
 - P0 projected overlay QA inspected by a human;
 - ROI pair has two DISTINCT past/current observation frames of same source ID;
 - datasets from same Revised CityNav asset revision, disjoint train/unseen maps.

This model learns image correspondence and type-language alignment, NOT goal
localization/Stop. Do NOT interpret source map polygon as a verified RGB mask.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .landmark_visual_pretrain import (
    FrozenSigLIP2Patches, PairedROIs, VisualLandmarkAligner,
    roi_collate, match_loss, retrieval_metrics, SameMapTypeBatchSampler
)
from .token_language import FrozenBERTTokens


def encode_twoview(batch,backbone,net,device):
    # Frozen vision runs under inference_mode. Clone outside that context:
    # trainable patch projections must be able to save their input for backward.
    a,va=backbone.encode(batch["first"])
    b,vb=backbone.encode(batch["second"])
    a,b=a.clone(),b.clone()
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(device.type=="cuda")):
        return net(a,va).float(),net(b,vb).float()


def run(args):
    if not args.acknowledge_geometry_pseudo_labels:
        raise ValueError(
            "Inspect P0 ROI overlay images first, then pass "
            "--acknowledge-geometry-pseudo-labels. Projection != semantic GT."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    train=PairedROIs(args.train,min_visible=args.min_visible,seed=args.seed)
    dev_base=PairedROIs(args.train,min_visible=args.min_visible,seed=args.seed,
                        deterministic=True)
    unseen=PairedROIs(args.unseen,min_visible=args.min_visible,seed=args.seed,
                      deterministic=True)
    if train.maps & unseen.maps:
        raise ValueError("P2 train/unseen map split leakage")
    order=list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    cut=int(len(order)*.9)
    if cut<2 or len(order)-cut<2:
        raise ValueError("need >=20 paired landmarks to create train/dev contrastive splits")
    loader_args=dict(batch_size=args.batch_size,num_workers=args.workers,
                     pin_memory=device.type=="cuda",collate_fn=roi_collate)
    train_subset=Subset(train,order[:cut])
    if args.random_negative_batches:
        train_loader=DataLoader(train_subset,shuffle=True,drop_last=True,**loader_args)
    else:
        sampler=SameMapTypeBatchSampler(
            train,order[:cut],batch_size=args.batch_size,seed=args.seed
        )
        train_loader=DataLoader(
            train_subset,batch_sampler=sampler,num_workers=args.workers,
            collate_fn=roi_collate,pin_memory=device.type=="cuda"
        )
    dev_loader=DataLoader(Subset(dev_base,order[cut:]),shuffle=False,**loader_args)
    unseen_loader=DataLoader(unseen,shuffle=False,**loader_args)
    backbone=FrozenSigLIP2Patches(
        name=args.image_backbone,device=args.device,
        microbatch=args.image_microbatch,cache_limit=args.cache_size
    )
    text=FrozenBERTTokens(name=args.text_backbone,device=args.device,max_length=32)
    type_text=text.encode([f"aerial image of a {typ}" for typ in train.types]).clone()
    model=VisualLandmarkAligner(
        vision_dim=backbone.hidden_size,hidden=args.hidden,heads=args.heads
    ).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    history=[]
    best=-1.
    for epoch in range(args.epochs):
        model.train()
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        losses=[]
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            a,b=encode_twoview(batch,backbone,model,device)
            loss,parts=match_loss(
                model,a,b,batch["type"].to(device),type_text,
                type_weight=args.type_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        dev=retrieval_metrics(
            model,dev_loader,backbone,device,max_groups=args.eval_max_groups
        )
        log=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=elapsed,
                 dev=dev,peak_gpu_gib=(
                     torch.cuda.max_memory_allocated(device)/2**30
                     if device.type=="cuda" else None
                 ))
        history.append(log)
        print(json.dumps(log),flush=True)
        if dev["R@1"]>best:
            best=dev["R@1"]
            torch.save({
                "model":model.state_dict(), "epoch":epoch+1,
                "image_backbone":args.image_backbone,
                "vision_hidden":backbone.hidden_size,
                "hidden":args.hidden,"heads":args.heads,
                "train_types":train.types,
                "train_maps":sorted(train.maps),
                "training_stage":"P2_visual_landmark_cross_view",
                "pseudo_label_warning":True,
            },out/"visual_best.pt")
    state=torch.load(out/"visual_best.pt",map_location=device,weights_only=True)
    model.load_state_dict(state["model"])
    unseen_score=retrieval_metrics(
        model,unseen_loader,backbone,device,max_groups=args.eval_max_groups
    )
    report=dict(
        stage="P2_patch_contrastive",trained_instances=len(train),
        unseen_instances=len(unseen),train_maps=sorted(train.maps),
        unseen_maps=sorted(unseen.maps),
        best_epoch=state["epoch"],history=history,
        unseen=unseen_score,
        warning="Geometric projected pseudo instances, NOT verified semantic region masks",
    )
    (out/"visual_pretrain_report.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8"
    )
    print(json.dumps({"unseen":unseen_score,"best_epoch":state["epoch"]}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True,
                   help="P0 train_seen/landmark_rois.jsonl (with saved ROIs)")
    p.add_argument("--unseen",required=True,
                   help="P0 val_unseen/landmark_rois.jsonl")
    p.add_argument("--acknowledge-geometry-pseudo-labels",action="store_true")
    p.add_argument("--output",default="runs/p2_visual_landmarks")
    p.add_argument("--image-backbone",default="google/siglip2-large-patch16-384")
    p.add_argument("--text-backbone",default="bert-base-uncased")
    p.add_argument("--image-microbatch",type=int,default=8)
    p.add_argument("--cache-size",type=int,default=2048)
    p.add_argument("--batch-size",type=int,default=16,
                   help="ROI identity pairs (2 x this many images)")
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--random-negative-batches",action="store_true",
                   help="disable default same-map/type hard visual negatives")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--hidden",type=int,default=256)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--type-weight",type=float,default=.25)
    p.add_argument("--min-visible",type=float,default=.8)
    p.add_argument("--eval-max-groups",type=int,default=1500)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    run(p.parse_args())


if __name__=="__main__":
    main()
