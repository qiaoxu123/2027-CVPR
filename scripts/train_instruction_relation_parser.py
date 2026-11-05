"""Train a BERT-span navigation relation graph parser on reviewed JSONL labels.

Each line:
 {"instruction":"A blue car in the parking lot behind One Stop.",
  "map_name":"cambridge_block_4",
  "reviewed":true,
  "spans":[{"start":7,"end":10,"role":"target"},
           {"start":18,"end":29,"role":"intermediate"},
           {"start":37,"end":45,"role":"public_landmark"}],
  "relations":[{"source":0,"target":1,"predicate":"inside","frame":"topological"},
               {"source":1,"target":2,"predicate":"behind","frame":"reference_frame_unknown"}]}

Source maps must be disjoint across fit and dev. No target positions, RGB,
teacher trajectories or val/test instructions. The external frozen BERT is
used only for token features. Unreviewed weak labels are NOT accepted.
This is a supervised parser head, not a verified raw-string-to-graph system.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from graphnav.instruction_relation_graph import ROLES, PREDICATES, FRAMES
from graphnav.instruction_graph_parser_head import (
    InstructionSpanGraphParser, supervised_parser_loss
)


def read_records(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError("Empty reviewed instruction-graph annotations")
    for i,row in enumerate(rows):
        if row.get("reviewed") is not True:
            raise ValueError(f"Unreviewed parser proposal at row {i} cannot be used as gold")
        if any(x in row for x in ("target_xy","target_world","teacher_poses","target_position")):
            raise ValueError("Private GT geometry must not enter parser labels")
        spans=row.get("spans",[])
        text=row.get("instruction")
        if not isinstance(text,str) or not spans:
            raise ValueError("Every example needs a sentence and reviewed spans")
        if len(spans)>32:
            raise ValueError("More than 32 spans; do not silently truncate")
        for item in spans:
            a,b=int(item["start"]),int(item["end"])
            if not(0<=a<b<=len(text)) or item["role"] not in ROLES:
                raise ValueError("Invalid reviewed mention")
        for edge in row.get("relations",[]):
            if not(0<=int(edge["source"])<len(spans) and
                   0<=int(edge["target"])<len(spans) and
                   edge["predicate"] in PREDICATES and edge["frame"] in FRAMES):
                raise ValueError("Invalid reviewed relation or frame")
    return rows


def make_batch(rows,tokenizer,device,max_length=96):
    texts=[x["instruction"] for x in rows]
    encoded=tokenizer(texts,truncation=True,max_length=max_length,
                      padding=True,return_tensors="pt",return_offsets_mapping=True)
    offsets=encoded.pop("offset_mapping")
    batch=len(rows);token_count=offsets.shape[1];n=max(len(x["spans"]) for x in rows)
    mask=torch.zeros((batch,n,token_count),dtype=torch.bool)
    valid=torch.zeros((batch,n),dtype=torch.bool)
    roles=torch.full((batch,n),-100,dtype=torch.long)
    edge_labels=torch.full((batch,n,n),-100,dtype=torch.long)
    frames=torch.full((batch,n,n),-100,dtype=torch.long)
    for i,row in enumerate(rows):
        m=len(row["spans"]);valid[i,:m]=True
        for j,s in enumerate(row["spans"]):
            a,b=int(s["start"]),int(s["end"])
            # Require positive overlap; truncation must not silently drop a span.
            token_ok=((offsets[i,:,0]<b)&(offsets[i,:,1]>a)
                      &(offsets[i,:,1]>offsets[i,:,0]))
            if not token_ok.any():
                raise ValueError(f"Span truncated/unaligned at task {row.get('map_name')} {j}")
            mask[i,j]=token_ok
            roles[i,j]=ROLES.index(s["role"])
        for s in range(m):
            for d in range(m):
                if s!=d:edge_labels[i,s,d]=0
        seen=set()
        for edge in row.get("relations",[]):
            src,dst=int(edge["source"]),int(edge["target"])
            if src==dst or (src,dst) in seen:
                raise ValueError("Duplicate/self-loop reviewed relation")
            seen.add((src,dst))
            edge_labels[i,src,dst]=PREDICATES.index(edge["predicate"])+1
            frames[i,src,dst]=FRAMES.index(edge["frame"])
    inp={k:v.to(device) for k,v in encoded.items()}
    return inp,mask.to(device),valid.to(device),roles.to(device),edge_labels.to(device),frames.to(device)


@torch.no_grad()
def evaluate(rows,tokenizer,bert,head,device,batch_size):
    head.eval();stats=dict(node_good=0,node_total=0,edge_tp=0,edge_fp=0,edge_fn=0,
                           predicate_correct=0,predicate_gold=0,
                           frame_correct=0,frame_gold=0)
    for start in range(0,len(rows),batch_size):
        inp,spans,valid,roles,edges,frames=make_batch(
            rows[start:start+batch_size],tokenizer,device
        )
        feat=bert(**inp).last_hidden_state.float()
        out=head(feat,spans,valid)
        node_mask=valid&(roles>=0)
        stats["node_good"]+=int(((out["role_logits"].argmax(-1)==roles)&node_mask).sum())
        stats["node_total"]+=int(node_mask.sum())
        eligible=out["valid_pairs"]&(edges>=0)
        pred=out["edge_logits"].argmax(-1)
        a=(pred>0)&eligible;b=(edges>0)&eligible
        stats["edge_tp"]+=int((a&b&(pred==edges)).sum())
        stats["edge_fp"]+=int((a&~b).sum()+(a&b&(pred!=edges)).sum())
        stats["edge_fn"]+=int((b&~a).sum()+(a&b&(pred!=edges)).sum())
        stats["predicate_correct"]+=int(((pred==edges)&b).sum())
        stats["predicate_gold"]+=int(b.sum())
        fmask=b&(frames>=0)
        stats["frame_correct"]+=int(((out["frame_logits"].argmax(-1)==frames)&fmask).sum())
        stats["frame_gold"]+=int(fmask.sum())
    tp,fp,fn=stats["edge_tp"],stats["edge_fp"],stats["edge_fn"]
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return dict(
        node_role_accuracy=stats["node_good"]/max(stats["node_total"],1),
        edge_triple_precision=precision,edge_triple_recall=recall,
        edge_triple_F1=2*precision*recall/max(precision+recall,1e-8),
        relation_label_accuracy_on_gold=stats["predicate_correct"]/max(stats["predicate_gold"],1),
        reference_frame_accuracy_on_gold=stats["frame_correct"]/max(stats["frame_gold"],1),
        raw_counts=stats,
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-annotated",type=Path,required=True)
    p.add_argument("--dev-annotated",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bert",default="bert-base-uncased")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--max-length",type=int,default=96)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Refusing to overwrite pretrained parser output")
    if a.epochs<1 or a.batch_size<1 or a.max_length<8:
        raise ValueError("Invalid training config")
    fit=read_records(a.fit_annotated);dev=read_records(a.dev_annotated)
    maps_fit={r["map_name"] for r in fit};maps_dev={r["map_name"] for r in dev}
    if maps_fit&maps_dev:
        raise ValueError("Fit/Dev contain shared maps; split by environment first")
    if len({r["instruction"] for r in fit}&{r["instruction"] for r in dev}):
        raise ValueError("Fit/Dev contain shared instructions")
    import transformers
    from transformers import AutoTokenizer,AutoModel
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(a.bert,use_fast=True)
    bert=AutoModel.from_pretrained(a.bert).eval().to(device).requires_grad_(False)
    head=InstructionSpanGraphParser(text_dim=bert.config.hidden_size).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=.01)
    a.output.mkdir(parents=True)
    (a.output/"manifest.json").write_text(json.dumps(dict(
        seed=a.seed,fit=len(fit),dev=len(dev),fit_maps=sorted(maps_fit),
        dev_maps=sorted(maps_dev),epochs=a.epochs,batch_size=a.batch_size,
        base_bert=a.bert,bert_frozen=True,labels_reviewed=True,
        target_GT_forward=False,model="span_graph_parser",no_navigation_claim=True,
    ),indent=2))
    history=[]
    for epoch in range(a.epochs):
        random.Random(a.seed+epoch).shuffle(fit)
        head.train();losses=[]
        for start in range(0,len(fit),a.batch_size):
            inp,spans,valid,roles,edges,frames=make_batch(
                fit[start:start+a.batch_size],tok,device,a.max_length
            )
            with torch.no_grad():
                token_feat=bert(**inp).last_hidden_state.float()
            out=head(token_feat,spans,valid)
            losses_parts=supervised_parser_loss(out,roles,edges,frames)
            optimizer.zero_grad(set_to_none=True)
            losses_parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step()
            losses.append(float(losses_parts["total"].detach()))
        report=evaluate(dev,tok,bert,head,device,a.batch_size)
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),dev=report)
        history.append(record)
        print(json.dumps(record),flush=True)
        tmp=a.output/"last.pt.tmp"
        torch.save(dict(epoch=epoch+1,model=head.state_dict(),
                        optimizer=optimizer.state_dict(),manifest=json.loads((a.output/"manifest.json").read_text()),
                        history=history),tmp)
        os.replace(tmp,a.output/"last.pt")
        (a.output/"history.json").write_text(json.dumps(history,indent=2))
    print(f"Saved reviewed-label parser checkpoint: {a.output/'last.pt'}")


if __name__=="__main__":
    main()
