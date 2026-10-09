"""P1 geometry-first reversible language localization with paired consistency.

--train: GENERATED pairs from train_seen maps only.
--unseen: GENERATED pairs from disjoint val_unseen maps, evaluated LAST.
No future trajectories, no RGB, no hidden target ID or GT passed forward.
Train/dev selection is within train map pairs; unseen not used for tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from .bidirectional import INVERSE
from .candidate_geometry import FullCandidateDataset, collate_full
from .full_localization import FullTokenGeometryLocalizer
from .static_localization import cross_entropy_goal, static_recall
from .token_language import FrozenBERTTokens
from .checkpointing import atomic_save,input_fingerprint,provenance,rng_state,restore_training


class PairedGeometryDataset(Dataset):
    def __init__(self, path, grid_size=28, max_relations=8):
        self.samples = FullCandidateDataset(
            path, grid_size=grid_size, max_landmarks=1,
            max_roads=0, max_relations=max_relations,
        )
        grouped={}
        for i, record in enumerate(self.samples.records):
            grouped.setdefault(record.get("pair_id"), {})[record.get("pair_role")]=i
        if None in grouped:
            raise ValueError("all synthetic rows must declare pair_id and pair_role")
        self.pairs=[]
        for pair, ids in sorted(grouped.items()):
            if set(ids)!={"forward","reverse"}:
                raise ValueError(f"incomplete reversible pair {pair}")
            f,r=(self.samples.records[ids[k]] for k in ("forward","reverse"))
            if (f["source_map_name"] != r["source_map_name"]
                or f["source_anchor_id"] != r["hidden_target_id"]
                or r["source_anchor_id"] != f["hidden_target_id"]
                or INVERSE[f["relation_direction"]] != r["relation_direction"]):
                raise ValueError(f"inconsistent inverse anchors for {pair}")
            self.pairs.append((ids["forward"],ids["reverse"]))
        if not self.pairs:
            raise ValueError("no usable reversible pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,i):
        f,r=self.pairs[i]
        encoded={"east":0,"west":1,"north":2,"south":3}
        result=[]
        for idx in (f,r):
            row=self.samples.records[idx]
            sample=self.samples[idx]
            # Direction is a LOSS/EVAL label only, never passed to model.forward.
            sample["direction_label"]=torch.tensor(
                encoded[row["relation_direction"]],dtype=torch.long
            )
            result.append(sample)
        return tuple(result)


def paired_collate(items):
    # Strict order: forward_0, reverse_0, forward_1, reverse_1, ...
    return collate_full([x for pair in items for x in pair])


def family_split(dataset, seed=0):
    """Keep a source pair and ALL its translations in the same train/dev split."""
    families = {}
    for i, (forward, _) in enumerate(dataset.pairs):
        key = dataset.samples.records[forward]["pair_id"].split(":aug")[0]
        families.setdefault(key, []).append(i)
    keys = sorted(families)
    random.Random(seed).shuffle(keys)
    pivot = max(1, int(len(keys) * .9))
    if pivot == len(keys):
        raise ValueError("too few source pair families for dev split")
    return ([i for key in keys[:pivot] for i in families[key]],
            [i for key in keys[pivot:] for i in families[key]])


def _inputs(batch, encoder, device):
    fields=("nodes","node_types","node_valid","entity_valid","adjacency",
            "edges","prior","prior_buildings","rules","relation_fields",
            "pair_geometry","grid_xy")
    x={k:batch[k].to(device,non_blocking=True) for k in fields}
    states,mask=encoder.encode_tokens(batch["text"])
    names=[n for group in batch["node_names"] for n in group]
    node_emb=encoder.encode(names).reshape(len(batch["text"]),len(batch["node_names"][0]),-1)
    return x, states.to(device).clone(), mask.to(device), node_emb.to(device).clone()


def batch_loss(model,batch,encoder,device,*,inverse_weight=.1):
    x,tokens,mask,nodes=_inputs(batch,encoder,device)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(device.type=="cuda")):
        field=model(x,variant="geometry",tokens=tokens,token_mask=mask,node_emb=nodes)
    scores=field.belief_logits.float()
    label=batch["target_xy"].to(device)
    target_loss=cross_entropy_goal(scores,label,grid_size=int(batch["grid_xy"].shape[1]**.5))
    # Expected relative displacements of paired predictions must cancel.
    probabilities=torch.softmax(scores,-1)
    prediction=(probabilities.unsqueeze(-1)*x["grid_xy"]).sum(1)
    anchor=x["nodes"][:,0,:2]
    scale=batch["map_sizes"].to(device)
    delta=(prediction-anchor)*scale / 100.
    inverse=F.smooth_l1_loss(delta[0::2]+delta[1::2],torch.zeros_like(delta[0::2]))
    return target_loss+inverse_weight*inverse, dict(
        goal=float(target_loss.detach()), inverse=float(inverse.detach()),
    ), field


@torch.no_grad()
def evaluate(model,loader,encoder,device):
    model.eval()
    totals=dict(count=0,NE_sum_m=0.,**{f"R@{k}/20m_count":0 for k in (1,4,8)})
    inverse_sum, steps, inverse_m_sum=0.,0,0.
    correct_side, directional_samples=0,0
    for batch in loader:
        _,values,field=batch_loss(model,batch,encoder,device)
        stats=static_recall(field.belief_logits.float(),
                            batch["target_xy"].to(device),
                            batch["map_sizes"].to(device),
                            batch["grid_xy"].to(device))
        for key in totals:
            totals[key]+=stats[key]
        pairs = len(batch["text"]) // 2
        inverse_sum+=values["inverse"] * pairs
        scores=field.belief_logits.float()
        expected=(scores.softmax(-1).unsqueeze(-1)
                  *batch["grid_xy"].to(device)).sum(1)
        delta=(expected-batch["nodes"][:,0,:2].to(device))
        delta_m = delta * batch["map_sizes"].to(device)
        inverse_m_sum += float((delta_m[0::2]+delta_m[1::2]).norm(dim=-1).sum())
        direction=batch["direction_label"].to(device)
        wanted=torch.stack((delta[:,0],-delta[:,0],delta[:,1],-delta[:,1]),-1)
        selected=wanted.gather(1,direction[:,None]).squeeze(1)
        correct_side+=int((selected>0).sum())
        directional_samples+=len(selected)
        steps+=pairs
    n=max(1,totals["count"])
    return dict(
        samples=totals["count"], NE_m=totals["NE_sum_m"]/n,
        inverse_error= inverse_sum/max(1,steps),
        inverse_displacement_error_m=inverse_m_sum/max(1,steps),
        direction_sign_accuracy=correct_side/max(1,directional_samples),
        **{f"R@{k}/20m":totals[f"R@{k}/20m_count"]/n for k in (1,4,8)},
    )


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device=torch.device(args.device)
    train=PairedGeometryDataset(args.train,grid_size=args.grid_size)
    unseen=PairedGeometryDataset(args.unseen,grid_size=args.grid_size)
    train_maps={r["source_map_name"] for r in train.samples.records}
    unseen_maps={r["source_map_name"] for r in unseen.samples.records}
    if train_maps & unseen_maps:
        raise ValueError("Train and unseen synthetic examples share source maps")
    train_ids, dev_ids = family_split(train, args.seed)
    kwargs=dict(batch_size=args.batch_size,num_workers=args.workers,
                pin_memory=device.type=="cuda",collate_fn=paired_collate,
                persistent_workers=args.workers > 0)
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(Subset(train,train_ids),shuffle=True,generator=generator,**kwargs)
    dev_loader=DataLoader(Subset(train,dev_ids),**kwargs)
    unseen_loader=DataLoader(unseen,**kwargs)
    model=FullTokenGeometryLocalizer(
        hidden=args.hidden,heads=args.heads,
        max_relations=8, relative_position=True,
    ).to(device)
    encoder=FrozenBERTTokens(name=args.text_backbone,device=args.device,max_length=128)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,weight_decay=.01,
    )
    dest=Path(args.output)
    resume=getattr(args,'resume',None)
    if dest.exists() and not resume:
        raise ValueError('refusing to overwrite P1 experiment; choose new output or --resume')
    dest.mkdir(parents=True,exist_ok=True)
    fingerprint=input_fingerprint(args,('train','unseen'));origin=provenance()
    history=[]
    best=-float("inf")
    start_epoch=0
    if resume:
        saved=restore_training(resume,model,optimizer,args,fingerprint)
        start_epoch=saved['epoch'];history=saved['history'];best=saved['best']
        if not (dest/'pretrain_best.pt').is_file():
            raise ValueError('resume requires preserved best checkpoint in same output')
    for epoch in range(start_epoch,args.epochs):
        generator.manual_seed(args.seed+epoch)
        model.train()
        t0=time.perf_counter()
        losses, sample_count = [], 0
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss,parts,_=batch_loss(
                model,batch,encoder,device,inverse_weight=args.inverse_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            n = len(batch["text"])
            losses.append((float(loss.detach()), n))
            sample_count += n
        if device.type=="cuda":
            torch.cuda.synchronize()
        elapsed=time.perf_counter()-t0
        peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None
        validation_start=time.perf_counter()
        dev=evaluate(model,dev_loader,encoder,device)
        record=dict(epoch=epoch+1,train_loss=sum(v*n for v,n in losses)/sample_count,
                    train_seconds=elapsed,dev=dev,
                    train_samples=sample_count, samples_per_second=sample_count/elapsed,
                    gpu_gib=peak,validation_seconds=time.perf_counter()-validation_start)
        history.append(record)
        print(json.dumps(record),flush=True)
        if dev["R@8/20m"]>best:
            best=dev["R@8/20m"]
            atomic_save(dict(
                model=model.state_dict(), epoch=epoch+1,
                hidden=args.hidden, heads=args.heads, grid_size=args.grid_size,
                relative_position=True,training_stage="P1_bidirectional_geometry",
                train_maps=sorted(train_maps),
                args=vars(args),
                input_fingerprint=fingerprint,provenance=origin,
            ),dest/"pretrain_best.pt")
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch+1,args=vars(args),rng=rng_state(),history=history,best=best,
                        input_fingerprint=fingerprint,provenance=origin), dest/"pretrain_last.pt")
        (dest/"history.json").write_text(json.dumps(history,indent=2))
    state=torch.load(dest/"pretrain_best.pt",map_location=device,weights_only=True)
    model.load_state_dict(state["model"])
    unseen_stats=evaluate(model,unseen_loader,encoder,device)
    report=dict(
        train_pairs=len(train),dev_pairs=len(dev_loader.dataset),
        unseen_pairs=len(unseen),
        checkpoint_epoch=state["epoch"],train_maps=sorted(train_maps),
        unseen_maps=sorted(unseen_maps),
        history=history,unseen=unseen_stats,
        input_fingerprint=fingerprint,provenance=origin,
        claim="synthetic reversible geometry localization only; not CityNav SR/SPL",
    )
    (dest/"bidirectional_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"unseen":unseen_stats,"best_epoch":state["epoch"]}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",required=True)
    p.add_argument("--unseen",required=True)
    p.add_argument("--output",default="runs/p1_bidirectional")
    p.add_argument('--resume',default=None)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=32,
                   help="number of pairs, model runs 2x this many samples")
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--grid-size",type=int,default=28)
    p.add_argument("--hidden",type=int,default=256)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--inverse-weight",type=float,default=.1)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--text-backbone",default="bert-base-uncased")
    run(p.parse_args())


if __name__=="__main__":
    main()
