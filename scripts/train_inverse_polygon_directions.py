"""Train directional Polygon-conditioned Belief from synthetic inverse pairs.

This is a standalone language/geometry PROBE, NOT a replacement for frozen
CityNav Geometry/PGVB. Every quartet has two geometrically true and two
counterfactual statements with the EXACT SAME visual/map objects.
A wrong-word record is a negative on the ORIGINAL target polygon, not a
fabricated new target coordinate.

Report two metrics:
  * target R20/NE on the held-out map for original true statements;
  * paired counterfactual map-direction response: does the heatmap's
    distribution *move toward the new direction* when only the word changes?
No val-seen, val-unseen or test data is loaded by this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import Polygon

from graphnav.inverse_polygon_direction_data import DIR,INVERSE
from graphnav.inverse_polygon_direction_model import (
    DirectionalPolygonBelief, collate_direction_features,
)


def read_quartets(path):
    rows=[json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()
          if line.strip()]
    if not rows or len(rows)%4:
        raise ValueError("Expected at least one complete quartet")
    groups=defaultdict(list)
    for row in rows:
        if row["source"]["source_kind"]!="train_polygon_synthetic_not_original_instruction":
            raise ValueError("Unverified source records")
        if any(key in row["input"] for key in ("target_contour","target_xy","target_world")):
            raise ValueError("Target polygon label leaked to model input")
        groups[row["family_id"]].append(row)
    quartets=[]
    for family,group in sorted(groups.items()):
        if len(group)!=4:
            raise ValueError(f"Family {family} has {len(group)} records")
        by={(r["source"]["orientation"],r["label"]["truth"]):r for r in group}
        order=[("forward",True),("inverse",True),("forward",False),("inverse",False)]
        if set(by)!=set(order):
            raise ValueError("Incomplete paired roles and truth labels")
        q=[by[k] for k in order]
        for pos,neg in ((q[0],q[2]),(q[1],q[3])):
            if pos["input"]["anchor_contour"] != neg["input"]["anchor_contour"]:
                raise ValueError("Counterfactual must keep Polygon unchanged")
            if pos["label"]["target_contour"] != neg["label"]["target_contour"]:
                raise ValueError("Counterfactual cannot invent a new target")
            if pos["input"]["relation_phrase"] == neg["input"]["relation_phrase"]:
                raise ValueError("Counterfactual should change relation word")
        if q[1]["input"]["relation_phrase"]!=INVERSE[q[0]["input"]["relation_phrase"]]:
            raise ValueError("Target/anchor swap did not invert predicate")
        quartets.append(q)
    return quartets


def cached_examples(quartets,device):
    """Cache geometry tensors once. Only truth labels change across swap."""
    result=[]
    for q in quartets:
        t=collate_direction_features(q,"cpu")
        result.append({
            key:value
            for key,value in t.items()
        })
    return result


def to_device(q,device):
    return {k:(v.to(device) if isinstance(v,torch.Tensor) else v)
            for k,v in q.items()}


def _forward(model,batch,bert_vectors):
    return model(batch["geometry"],batch["outline"],batch["part_mask"],
                 batch["areas"],batch["direction_id"],bert_vectors[batch["direction_id"]])


def _mass(logits, label):
    """Soft expected density on fixed target polygon (not hard cell only)."""
    return (logits.log_softmax(-1).exp()*label).sum(-1).clamp_min(1e-12)


def quartet_loss(model,pack,bert_vectors,*,gap=.3):
    logits=_forward(model,pack,bert_vectors)  # 4x784
    labels=pack["soft_target"]
    # True statements have the original geometry labels, no GT in inputs.
    positive=-(labels[:2]*logits[:2].log_softmax(-1)).sum(-1).mean()
    pos=torch.log(_mass(logits[:2],labels[:2]))
    neg=torch.log(_mass(logits[2:],labels[2:]))
    margin=F.relu(gap+neg-pos).mean()
    # The correct and counterfactual statements use identical anchor Polygon.
    # Their output should differ. This is a diagnostic stability prior rather
    # than an oracle position for the counterfactual target.
    kl=(F.kl_div(logits[:2].log_softmax(-1),
                 logits[2:].softmax(-1).detach(),reduction="none").sum(-1).mean())
    separation=F.relu(0.1-kl)
    return positive+1.5*margin+0.1*separation,dict(positive=float(positive.detach()),
                                             ranking=float(margin.detach()),
                                             divergence=float(kl.detach()))


def metrics(model,groups,bert_vectors,device):
    model.eval()
    vals=[]
    with torch.no_grad():
        for group in groups:
            q=to_device(group,device)
            logits=_forward(model,q,bert_vectors)
            prob=logits.softmax(-1)
            centroids=(prob[:,:,None]*q["world_grid"]).sum(1)
            pos,inv,cf_pos,cf_inv=centroids
            # Strict paired response along *the changed word's* signed axis.
            axes=model.axes[q["direction_id"]]
            # Swapping word should move probability centroid toward new named
            # direction in common world XY; original label is never used here.
            d0=(cf_pos-pos)@axes[2]
            d1=(cf_inv-inv)@axes[3]
            target_xy=[]
            for i in range(2):
                label=q["soft_target"][i]
                k=int(prob[i].argmax())
                # Expected metric error relative to label-weighted polygon
                # centroid is diagnostic; polygon/shape exact error is below.
                true_center=(label[:,None]*q["world_grid"][i]).sum(0)
                err=torch.linalg.vector_norm(q["world_grid"][i,k]-true_center)
                target_xy.append(float(err))
            positives=[]
            for i in range(2):
                ll=q["soft_target"][i]
                k=int(prob[i].argmax())
                poly_mass=float(ll[k])
                positives.append(dict(top1_xy=q["world_grid"][i,k].cpu().tolist(),
                                      label_mass_at_top1=poly_mass,
                                      proxy_NE_m=target_xy[i]))
            result=dict(
                family_id=group["family_id"][0],
                map_name=group["family_id"][0].split(":")[0],
                forward=group["text"][0],inverse=group["text"][1],
                direction=group["direction_id"].cpu().tolist(),
                inverse_relation_verified=(group["direction_id"][0]!=group["direction_id"][1]).item(),
                centroid_xy=centroids.cpu().tolist(),
                changed_forward_toward_new_direction_m=float(d0),
                changed_inverse_toward_new_direction_m=float(d1),
                changed_forward_expected=bool(d0>0),
                changed_inverse_expected=bool(d1>0),
                positive_predictions=positives,
            )
            vals.append(result)
    if not vals:raise ValueError("No pairs")
    return dict(
        pairs=len(vals),
        paired_word_response_accuracy=float(
            np.mean([int(v["changed_forward_expected"] and v["changed_inverse_expected"])
                     for v in vals])),
        response_directions_individual=float(np.mean([
            int(v[k]) for v in vals for k in
            ("changed_forward_expected","changed_inverse_expected")])),
        average_shift_m=float(np.mean([
            v[k] for v in vals for k in
            ("changed_forward_toward_new_direction_m",
             "changed_inverse_toward_new_direction_m")])),
        positive_proxy_NE_m=float(np.mean([
            x["proxy_NE_m"] for v in vals for x in v["positive_predictions"]])),
        cases=vals,
    )


def save_figures(model,group,bert_vectors,output,device,max_pairs=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=output/"figures";folder.mkdir(parents=True,exist_ok=True)
    model.eval()
    with torch.no_grad():
        for j,g in enumerate(group[:max_pairs]):
            batch=to_device(g,device)
            p=_forward(model,batch,bert_vectors).softmax(-1).cpu().numpy()
            fig,axes=plt.subplots(2,2,figsize=(9,8))
            for i,ax in enumerate(axes.flat):
                z=p[i].reshape(28,28)
                extent=batch["bounds"][i]
                ax.imshow(z,origin="lower",extent=extent,interpolation="nearest",
                          cmap="inferno",aspect="auto")
                anchor=np.asarray(g["outline"][i,0,:,:2])
                # Outline tokens are normalized to map frame; recover physical XY.
                bounds=np.asarray(extent)
                world=anchor*(bounds[2:]-bounds[:2])+bounds[:2]
                ax.plot(np.r_[world[:,0],world[0,0]],
                        np.r_[world[:,1],world[0,1]],color="cyan",lw=1)
                t=("positive" if i<2 else "wrong-direction")
                ax.set_title(g["text"][i]+" ["+t+"]",fontsize=8)
                ax.set_xlabel("map X (m)");ax.set_ylabel("map Y (m)")
            fig.tight_layout()
            fig.savefig(folder/f"pair_{j:04d}.png",dpi=150)
            plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--max-figures",type=int,default=6)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--bert-model",default="bert-base-uncased",
                    help="Existing frozen local BERT checkpoint")
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    if args.epochs<1 or args.lr<=0:raise ValueError("Invalid training hyperparameters")
    if not (args.dataset/"inventory.json").exists():
        raise ValueError("Missing dataset inventory")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Refusing to overwrite output")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device)
    from transformers import AutoTokenizer,AutoModel
    tokenizer=AutoTokenizer.from_pretrained(args.bert_model)
    bert=AutoModel.from_pretrained(args.bert_model).eval().to(device).requires_grad_(False)
    # Exactly four relation phrases, computed once; full-length object names
    # are kept in audit rows but cannot leak identity into the direction probe.
    with torch.no_grad():
        phrases=[f"the object is {w} of the landmark" for w in DIR]
        inputs=tokenizer(phrases,padding=True,truncation=True,return_tensors="pt").to(device)
        bert_vectors=bert(**inputs).last_hidden_state[:,0].detach()
    model=DirectionalPolygonBelief(text_dim=bert_vectors.shape[-1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
    fit=read_quartets(args.dataset/"fit.jsonl")
    dev=read_quartets(args.dataset/"dev.jsonl")
    maps_fit={q[0]["map_name"] for q in fit}
    maps_dev={q[0]["map_name"] for q in dev}
    if maps_fit&maps_dev:raise ValueError("Fit and dev maps overlap")
    manifest=dict(seed=args.seed,train_pairs=len(fit),dev_pairs=len(dev),
                  bert_model=args.bert_model,split_fit=sorted(maps_fit),
                  split_dev=sorted(maps_dev),epochs=args.epochs,lr=args.lr,
                  source="synthetic TRAIN Polygon map-relative directions",
                  target_GT_used_for_input=False,
                  no_original_navigation_instruction_replacement=True,
                  output_not_closedloop_navigation=True)
    if args.resume:
        if not (args.output/"last.pt").exists():
            raise FileNotFoundError("No previous checkpoint to resume")
        prev=torch.load(args.output/"last.pt",map_location=device,weights_only=False)
        if {k:v for k,v in prev["manifest"].items() if k!="epochs"}!={
            k:v for k,v in manifest.items() if k!="epochs"}:
            raise ValueError("Checkpoint provenance differs")
        model.load_state_dict(prev["model"])
        optimizer.load_state_dict(prev["optimizer"])
        start_epoch=int(prev["epoch"]);history=list(prev["history"])
    else:
        start_epoch=0;history=[]
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Cache geometries for {len(fit)} FIT + {len(dev)} DEV pairs",flush=True)
    train=cached_examples(fit,device)
    valid=cached_examples(dev,device)
    for epoch in range(start_epoch,args.epochs):
        model.train();running=[]
        perm=np.random.default_rng(args.seed+epoch).permutation(len(train))
        for index in perm:
            group=to_device(train[index],device)
            objective,parts=quartet_loss(model,group,bert_vectors)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            running.append(float(objective.detach()))
        dev_stats=metrics(model,valid,bert_vectors,device)
        summary=dict(epoch=epoch+1,fit_loss=float(np.mean(running)),
                     dev_pairs=dev_stats["pairs"],
                     dev_word_response_accuracy=dev_stats["paired_word_response_accuracy"],
                     dev_shift_m=dev_stats["average_shift_m"],
                     dev_proxy_NE_m=dev_stats["positive_proxy_NE_m"])
        history.append(summary)
        print(json.dumps(summary),flush=True)
        torch.save(dict(epoch=epoch+1,model=model.state_dict(),
                        optimizer=optimizer.state_dict(),manifest=manifest,
                        history=history),args.output/"last.pt.tmp")
        (args.output/"last.pt.tmp").replace(args.output/"last.pt")
        (args.output/"history.json").write_text(json.dumps(history,indent=2))
    evaluation=metrics(model,valid,bert_vectors,device)
    (args.output/"dev_metrics.json").write_text(json.dumps(
        {k:v for k,v in evaluation.items() if k!="cases"},indent=2))
    with (args.output/"dev_cases.jsonl").open("w") as f:
        for row in evaluation["cases"]:f.write(json.dumps(row)+"\n")
    save_figures(model,valid,bert_vectors,args.output,device,max_pairs=args.max_figures)
    print("Output:",args.output,flush=True)


if __name__=="__main__":
    main()
