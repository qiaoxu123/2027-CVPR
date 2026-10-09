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
from .checkpointing import atomic_save,input_fingerprint,provenance,rng_state,restore_training


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
                     pin_memory=device.type=="cuda",collate_fn=roi_collate,
                     persistent_workers=args.workers > 0)
    train_subset=Subset(train,order[:cut])
    if args.random_negative_batches:
        train_loader=DataLoader(train_subset,shuffle=True,drop_last=True,**loader_args)
    else:
        sampler=SameMapTypeBatchSampler(
            train,order[:cut],batch_size=args.batch_size,seed=args.seed
        )
        train_loader=DataLoader(
            train_subset,batch_sampler=sampler,num_workers=args.workers,
            collate_fn=roi_collate,pin_memory=device.type=="cuda",
            persistent_workers=args.workers > 0
        )
    dev_loader=DataLoader(Subset(dev_base,order[cut:]),shuffle=False,**loader_args)
    unseen_loader=DataLoader(unseen,shuffle=False,**loader_args)
    backbone=FrozenSigLIP2Patches(
        name=args.image_backbone,device=args.device,
        microbatch=args.image_microbatch,cache_limit=args.cache_size,
        cache_directory=args.patch_cache_directory
    )
    text=FrozenBERTTokens(name=args.text_backbone,device=args.device,max_length=32)
    type_text=text.encode([f"aerial image of a {typ}" for typ in train.types]).clone()
    model=VisualLandmarkAligner(
        vision_dim=backbone.hidden_size,hidden=args.hidden,heads=args.heads
    ).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    out=Path(args.output)
    resume=getattr(args,'resume',None)
    if out.exists() and not resume:
        raise ValueError('refusing to overwrite existing P2 experiment; use new output or --resume')
    out.mkdir(parents=True,exist_ok=True)
    fingerprint=input_fingerprint(args,('train','unseen'))
    origin=provenance()
    history=[]
    best=-1.
    start_epoch=0
    if resume:
        saved=restore_training(resume,model,optimizer,args,fingerprint)
        start_epoch=saved['epoch'];history=saved['history'];best=saved['best']
        if not (out/'visual_best.pt').is_file() or not (out/'visual_initial.pt').is_file():
            raise ValueError('resume requires preserved initial and best checkpoints in the same output')
    else:
        atomic_save({"model":model.state_dict()}, out/"visual_initial.pt")
    for epoch in range(start_epoch,args.epochs):
        train.set_epoch(epoch)
        if not args.random_negative_batches:sampler.epoch=epoch
        model.train()
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        losses, count = [], 0
        encoded_start = backbone.encoded_images
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
            n = len(batch["first"])
            count += n
            losses.append((float(loss.detach()), n))
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        if count == 0:
            raise ValueError("No train batches with same-city/type hard negatives")
        peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None
        validation_start=time.perf_counter()
        dev=retrieval_metrics(
            model,dev_loader,backbone,device,max_groups=args.eval_max_groups
        )
        log=dict(epoch=epoch+1,loss=sum(v*n for v,n in losses)/count,seconds=elapsed,
                 train_pairs=count,pairs_per_second=count/elapsed,
                 images_per_second=2*count/elapsed,
                 train_instance_fraction=count/len(train_subset),
                 newly_encoded_images=backbone.encoded_images-encoded_start,
                 dev=dev,peak_gpu_gib=peak,validation_seconds=time.perf_counter()-validation_start)
        history.append(log)
        print(json.dumps(log),flush=True)
        if dev["R@1"]>best:
            best=dev["R@1"]
            atomic_save({
                "model":model.state_dict(), "epoch":epoch+1,
                "image_backbone":args.image_backbone,
                "vision_hidden":backbone.hidden_size,
                "hidden":args.hidden,"heads":args.heads,
                "train_types":train.types,
                "train_maps":sorted(train.maps),
                "training_stage":"P2_visual_landmark_cross_view",
                "pseudo_label_warning":True,
                "args": vars(args),
                "input_fingerprint":fingerprint,"provenance":origin,
            },out/"visual_best.pt")
        atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "epoch":epoch+1,"args":vars(args),"rng":rng_state(),
                    "history":history,"best":best,"input_fingerprint":fingerprint,
                    "provenance":origin},out/"visual_last.pt")
        (out/"history.json").write_text(json.dumps(history,indent=2))
    state=torch.load(out/"visual_best.pt",map_location=device,weights_only=True)
    model.load_state_dict(state["model"])
    unseen_score=retrieval_metrics(
        model,unseen_loader,backbone,device,max_groups=args.eval_max_groups,
        prediction_path=out/"unseen_predictions.jsonl"
    )
    # Baselines are evaluated on holdout only AFTER selection/training.
    initial=VisualLandmarkAligner(vision_dim=backbone.hidden_size,
                                 hidden=args.hidden,heads=args.heads).to(device)
    initial.load_state_dict(torch.load(out/"visual_initial.pt",map_location=device,weights_only=True)["model"])
    initial_score=retrieval_metrics(initial,unseen_loader,backbone,device,max_groups=args.eval_max_groups)
    class MeanPatches(torch.nn.Module):
        def forward(self, patches, valid):
            weights=valid.to(patches.dtype).unsqueeze(-1)
            return torch.nn.functional.normalize((patches*weights).sum(1)/weights.sum(1).clamp_min(1),dim=-1)
    frozen_mean_score=retrieval_metrics(MeanPatches().to(device),unseen_loader,backbone,device,
                                       max_groups=args.eval_max_groups)
    report=dict(
        stage="P2_patch_contrastive",eligible_train_instances=len(train),
        trained_instances=cut,dev_instances=len(order)-cut,
        unseen_instances=len(unseen),train_maps=sorted(train.maps),
        unseen_maps=sorted(unseen.maps),
        best_epoch=state["epoch"],history=history,
        unseen=unseen_score,
        unseen_initial_head=initial_score,unseen_frozen_mean_patch=frozen_mean_score,
        args=vars(args),vision_revision=backbone.protocol[1],
        newly_encoded_images_total=backbone.encoded_images,disk_cache_hits=backbone.disk_hits,
        warning="Geometric projected pseudo instances, NOT verified semantic region masks",
        provenance=origin,input_fingerprint=fingerprint,
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
    p.add_argument('--resume',default=None)
    p.add_argument("--image-backbone",default="google/siglip2-large-patch16-384")
    p.add_argument("--text-backbone",default="bert-base-uncased")
    p.add_argument("--image-microbatch",type=int,default=8)
    p.add_argument("--cache-size",type=int,default=2048)
    p.add_argument("--patch-cache-directory",default=None)
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
