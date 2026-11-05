"""TRAIN-only complete-data fit: ONLY public names, adjacent spatial words & Polygon.

No original-instruction token embedding. No target category, color, intermediate
node, RGB, old Geometry/Relation checkpoint or processed target annotations.

Only --public-objects JSON is used for map-name-to-Public-Polygon binding
(never GT-selected reference lists from the revised data). The revised TRAIN
JSONL is used for original instruction, PUBLIC map bounds, and post-forward
target labels. 21,878 query coverage is checked; unknown inputs stay in the
denominator and produce uniform Belief.

Cambridge TRAIN-Fit and Birmingham TRAIN-Dev use disjoint maps. Neither
split is a genuine unseen-world evaluation of all prior research experiments.
No Val/Test/navigation SR is performed here.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,Subset

from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.geometry import grid_centers,normalize_xy
from graphnav.landmark_geometry_words import extract_landmark_geometry
from graphnav.landmark_geometry_only_model import LandmarkGeometryWordsOnly
from graphnav.static_localization import cross_entropy_goal
from graphnav.token_language import FrozenBERTTokens
from scripts.analyze_instruction_landmark_association import PublicNames

FORMAT="landmark-and-geometric-words-only-v1"
M=16
P=16


class LandmarkWordsDataset(Dataset):
    def __init__(self,train_jsonl,public_objects_path,*,max_anchors=M):
        self.records=[]
        self.counter=Counter()
        catalog=json.loads(Path(public_objects_path).read_text())
        public_names=PublicNames(catalog)
        source=[json.loads(s) for s in Path(train_jsonl).read_text().splitlines() if s.strip()]
        if not source:raise ValueError("Empty TRAIN JSONL")
        if {r.get("dataset_version") for r in source}!={"revised"}:
            raise ValueError("Only Revised TRAIN JSONL is allowed")
        if {r.get("match_mode") for r in source}!={"fuzzy"}:
            raise ValueError("Only full Revised/fuzzy records are allowed")
        for row in source:
            map_name=row["map_name"]
            if map_name not in catalog:raise ValueError(f"Missing public map catalog {map_name}")
            text=row["instruction"]
            bounds=row["world_bounds"]
            matches=public_names.match(map_name,text)
            selected=[]
            for mention in matches:
                for object_id in mention["public_ids"]:
                    raw=catalog[map_name][str(object_id)]
                    # Raw PUBLIC CityRefer contour is world XY; normalize using
                    # the PUBLIC per-map bounds before any model operation.
                    xy=normalize_xy(raw["contour"],bounds)
                    selected.append(dict(id=str(object_id),name=raw["name"],
                                         aliases=[mention["text"]],contour=xy.tolist()))
            try:
                parsed=extract_landmark_geometry(text,selected,max_anchors=max_anchors)
                status="paired" if parsed["paired_geometry"] else (
                    "name_only" if parsed["matched_names"] else "no_public_name")
            except ValueError as exc:
                # An over-capacity/invalid input is never silently dropped
                # from full 21,878-task denominators.
                parsed=dict(anchors=[],matched_names=0,paired_geometry=0,
                            invalid_public_contours=0,target_GT_used=False)
                status="rejected_"+str(exc).split(";")[0][:80]
            self.counter[status]+=1
            self.records.append(dict(
                map_name=map_name,anchors=parsed["anchors"],status=status,
                target_xy=np.asarray(row["target_xy"],np.float32),
                map_sizes=np.asarray(row["physical_map_sizes"],np.float32)))
        if len(self.records)!=len(source):
            raise AssertionError("TRAIN query coverage changed")

    def __len__(self):return len(self.records)

    def __getitem__(self,idx):
        row=self.records[idx]
        names=[""]*M;words=[""]*M
        contours=np.zeros((M,P,2),np.float32)
        valid=np.zeros(M,bool);word_valid=np.zeros(M,bool)
        for i,anchor in enumerate(row["anchors"]):
            names[i]=anchor["name"];words[i]=anchor["geometry_words"]
            contours[i]=np.asarray(anchor["contour"],np.float32)
            valid[i]=True;word_valid[i]=bool(anchor["geometry_words"])
        # Labels are explicitly NOT part of the input dictionary.
        return dict(
            model_inputs=dict(name_text=names,geometry_words=words,
                              contours=torch.from_numpy(contours),
                              anchor_mask=torch.from_numpy(valid),
                              word_mask=torch.from_numpy(word_valid)),
            label_xy=torch.from_numpy(row["target_xy"]),
            map_sizes=torch.from_numpy(row["map_sizes"]),
            status=row["status"])


def collate(items):
    inp=[row["model_inputs"] for row in items]
    return dict(
        model_inputs=dict(
            name_text=[s["name_text"] for s in inp],
            geometry_words=[s["geometry_words"] for s in inp],
            contours=torch.stack([s["contours"] for s in inp]),
            anchor_mask=torch.stack([s["anchor_mask"] for s in inp]),
            word_mask=torch.stack([s["word_mask"] for s in inp]),
        ),
        label_xy=torch.stack([row["label_xy"] for row in items]),
        map_sizes=torch.stack([row["map_sizes"] for row in items]),
        status=[row["status"] for row in items],
    )


def encode_model_inputs(batch,bert,device,*,control="real"):
    """Whitelist ONLY name, adjacent words and public Polygon. Never text/GT."""
    inputs=batch["model_inputs"]
    mask=inputs["anchor_mask"].to(device).bool()
    word_mask=inputs["word_mask"].to(device).bool()
    b,m=mask.shape
    name=torch.zeros((b,m,768),device=device)
    words=torch.zeros_like(name)
    name_entries=[(i,j,s) for i,group in enumerate(inputs["name_text"])
                  for j,s in enumerate(group) if mask[i,j]]
    phrase_entries=[(i,j,s) for i,group in enumerate(inputs["geometry_words"])
                    for j,s in enumerate(group) if word_mask[i,j]]
    if name_entries:
        encoded=bert.encode([s for _,_,s in name_entries]).detach().clone().float()
        for k,(i,j,_) in enumerate(name_entries):name[i,j]=encoded[k]
    if phrase_entries and control!="name_only":
        encoded=bert.encode([s for _,_,s in phrase_entries]).detach().clone().float()
        if control=="scrambled_words" and len(phrase_entries)>1:
            encoded=encoded.roll(1,0)  # preserves number of word-bearing nodes
        for k,(i,j,_) in enumerate(phrase_entries):words[i,j]=encoded[k]
    if control=="name_only":
        word_mask=torch.zeros_like(word_mask)
    elif control not in ("real","scrambled_words"):
        raise ValueError("Invalid language control")
    return dict(
        name_embedding=name,words_embedding=words,
        landmark_contours=inputs["contours"].to(device).float(),
        landmark_valid=mask,words_valid=word_mask,
        candidate_xy=grid_centers(28,device=device)[None].expand(b,-1,-1),
    )


def _empty_stat():
    return dict(n=0,hit=0,ne=0.,fix=0,damage=0,changed=0)


def add_metric(stats,before,after,truth,map_size):
    # before: a paired "name-only" same-model ablation.
    grid=grid_centers(28,device=after.device)
    start=before.argmax(-1);end=after.argmax(-1)
    e0=((grid[start]-truth)*map_size).norm(dim=-1)
    e1=((grid[end]-truth)*map_size).norm(dim=-1)
    good0=e0<=20.;good1=e1<=20.
    stats["n"]+=len(e1)
    stats["hit"]+=int(good1.sum())
    stats["ne"]+=float(e1.sum())
    stats["fix"]+=int((~good0&good1).sum())
    stats["damage"]+=int((good0&~good1).sum())
    stats["changed"]+=int((start!=end).sum())


def report_stat(stats):
    n=stats["n"]
    return dict(
        N=n,R20=stats["hit"]/max(1,n),NE_m=stats["ne"]/max(1,n),
        corrected_vs_name_only=stats["fix"],
        damaged_vs_name_only=stats["damage"],
        changed_top1=stats["changed"])


@torch.no_grad()
def evaluate(model,bert,dataset,ids,device,batch_size,max_batches=0):
    model.eval()
    loader=DataLoader(Subset(dataset,ids),batch_size=batch_size,
                      collate_fn=collate,shuffle=False,num_workers=0)
    stats={cat:{arm:_empty_stat() for arm in ("name_only","real","scrambled_words")}
           for cat in ("all","paired","name_only_input","no_public_name")}
    for num,batch in enumerate(loader):
        if max_batches and num>=max_batches:break
        outputs={}
        for arm in ("name_only","real","scrambled_words"):
            x=encode_model_inputs(batch,bert,device,control=arm)
            outputs[arm]=model(**x)
        base=outputs["name_only"]
        target=batch["label_xy"].to(device)
        scale=batch["map_sizes"].to(device)
        for arm,pred in outputs.items():
            for label in stats:
                if label=="all":
                    keep=torch.ones(len(target),dtype=torch.bool,device=device)
                elif label=="paired":
                    keep=torch.tensor([s=="paired" for s in batch["status"]],device=device)
                elif label=="name_only_input":
                    keep=torch.tensor([s=="name_only" for s in batch["status"]],device=device)
                else:
                    keep=torch.tensor([s!="paired" and s!="name_only" for s in batch["status"]],device=device)
                if keep.any():
                    add_metric(stats[label][arm],base[keep],pred[keep],
                               target[keep],scale[keep])
    return {cat:{arm:report_stat(data) for arm,data in group.items()}
            for cat,group in stats.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl",type=Path,required=True)
    p.add_argument("--public-objects",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--expected-train-records",type=int,default=21878,
                   help="Refuse partial TRAIN by default")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--max-train-batches",type=int,default=0)
    p.add_argument("--max-dev-batches",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.seed!=42 or a.epochs<1 or a.batch_size<1:
        raise ValueError("Protocol requires seed=42, positive epochs/batch size")
    if a.max_train_batches<0 or a.max_dev_batches<0:
        raise ValueError("Negative smoke limit")
    if not torch.cuda.is_available():
        raise RuntimeError("TRAIN requires local CUDA")
    if a.output.exists() and not a.resume:
        raise FileExistsError("Refuse overwrite; create new output directory")
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda")
    dataset=LandmarkWordsDataset(a.train_jsonl,a.public_objects)
    if len(dataset)!=a.expected_train_records:
        raise ValueError(
            f"Full TRAIN requires {a.expected_train_records} tasks, got {len(dataset)}"
        )
    fit=[i for i,r in enumerate(dataset.records) if r["map_name"].lower().startswith("cambridge_")]
    dev=[i for i,r in enumerate(dataset.records) if r["map_name"].lower().startswith("birmingham_")]
    if len(fit)+len(dev)!=len(dataset) or not fit or not dev:
        raise ValueError("All TRAIN records must be accounted by disjoint Cambridge/Birmingham")
    model=LandmarkGeometryWordsOnly().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    bert=FrozenBERTTokens(device=device)
    cached=a.output/"bert_cache.pt"
    if a.resume and cached.exists():bert.load_cache(cached)
    manifest=dict(protocol=FORMAT,
        input_fields=["public_landmark_name","adjacent_geometric_words","public_polygon"],
        forbidden=["original_instruction_embedding","target_appearance","target_class",
                   "intermediate_objects","RGB","GT_as_input","old_pretrained_Geometry"],
        data_sha256=file_sha256(a.train_jsonl),
        public_catalog_sha256=file_sha256(a.public_objects),
        model_sha256=file_sha256("graphnav/landmark_geometry_only_model.py"),
        extractor_sha256=file_sha256("graphnav/landmark_geometry_words.py"),
        training_sha256=file_sha256(__file__),
        seed=42,epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,
        n_train_all=len(dataset),expected_train_records=a.expected_train_records,
        fit_queries=len(fit),dev_queries=len(dev),
        fit_maps=sorted({dataset.records[i]["map_name"] for i in fit}),
        dev_maps=sorted({dataset.records[i]["map_name"] for i in dev}),
        extraction_distribution=dict(dataset.counter),
        smoke=bool(a.max_train_batches or a.max_dev_batches),
        benchmark_claims="TRAIN-Dev goal localization only; no Val/Test/SR")
    if a.resume:
        if not (a.output/"manifest.json").exists() or json.loads(
                (a.output/"manifest.json").read_text())!=manifest:
            raise ValueError("Resume manifest mismatch; code/data/params changed")
        state=torch.load(a.output/"last.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start=int(state["epoch"]);history=state["history"]
    else:
        atomic_json(manifest,a.output/"manifest.json")
        # Public auditing of what reaches the network: ONLY pairs of mapped
        # landmark names, adjacency words and Polygon geometry validity.
        with (a.output/"extraction_audit.jsonl").open("w") as stream:
            for row in dataset.records:
                stream.write(json.dumps(dict(
                    map_name=row["map_name"],status=row["status"],
                    pairs=[dict(name=x["name"],geometric_words=x["geometry_words"],
                                public_id=x["public_id"])
                           for x in row["anchors"]]
                ),ensure_ascii=False)+"\n")
        start=0;history=[]
    print(json.dumps(dict(n_total=len(dataset),fit=len(fit),dev=len(dev),
                          extraction=dict(dataset.counter))),flush=True)
    for epoch in range(start,a.epochs):
        order=np.random.default_rng(a.seed+epoch).permutation(fit).tolist()
        loader=DataLoader(Subset(dataset,order),batch_size=a.batch_size,
                          shuffle=False,collate_fn=collate,num_workers=0)
        model.train();seen=0;loss_total=0.;steps=0
        for step,batch in enumerate(loader):
            if a.max_train_batches and step>=a.max_train_batches:break
            x=encode_model_inputs(batch,bert,device)
            out=model(**x)
            label=batch["label_xy"].to(device)
            loss=cross_entropy_goal(out,label)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            seen+=len(label);loss_total+=float(loss.detach())*len(label);steps+=1
            if step%200==0:
                print(json.dumps(dict(epoch=epoch+1,step=step+1,seen=seen,
                                      loss=float(loss.detach()),grad_norm=float(grad))),flush=True)
        if not a.max_train_batches and seen!=len(fit):
            raise RuntimeError(f"TRAIN-Fit incomplete {seen}/{len(fit)}")
        history.append(dict(epoch=epoch+1,fit_seen=seen,steps=steps,
                            train_loss=loss_total/max(1,seen)))
        atomic_save(dict(format=FORMAT,epoch=epoch+1,model=model.state_dict(),
                         optimizer=optimizer.state_dict(),manifest=manifest,
                         history=history),a.output/"last.pt")
        atomic_json(history,a.output/"history.json")
        bert.save_cache(cached)
        print(json.dumps(history[-1]),flush=True)
    report=dict(manifest=manifest,completed_epochs=a.epochs,
                dev=evaluate(model,bert,dataset,dev,device,a.batch_size,
                             max_batches=a.max_dev_batches))
    atomic_json(report,a.output/"report.json")
    bert.save_cache(cached)
    print(json.dumps(report["dev"],indent=2),flush=True)


if __name__=="__main__":main()
