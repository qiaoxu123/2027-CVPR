"""TRAIN-Dev only: paired word-swap diagnostic of the frozen Polygon-language head.

Re-encode one cardinal phrase, keep the SAME frozen base and all Polygon
geometry fixed; compare actual language residuals and 28x28 belief centroids.
Do not claim that 'north' means positive map Y unless verified externally.
This reports language sensitivity, NOT automatic predicate correctness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from graphnav.candidate_geometry import collate_full
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.full_relation_training import load_full_relation_model
from graphnav.polygon_language_attention import PolygonGuidedLanguageAttention
from graphnav.polygon_language_counterfactual import swap_single_cardinal
from graphnav.token_language import FrozenBERTTokens
from scripts.train_polygon_language_attention import make_dataset,frozen_features,FORMAT


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl",type=Path,required=True)
    p.add_argument("--relation-checkpoint",type=Path,required=True)
    p.add_argument("--adapter-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-pairs",type=int,default=100)
    p.add_argument("--verify-world-north",action="store_true",
                   help="Only after independent map CRS/north verification")
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("Do not overwrite a previous diagnostic")
    if a.max_pairs<1:
        raise ValueError("max-pairs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run on local CityNav GPU")
    device=torch.device("cuda")
    state=torch.load(a.adapter_checkpoint,map_location="cpu",weights_only=True)
    if state.get("format")!=FORMAT:
        raise ValueError("Expected standalone Polygon-language adapter")
    if (state["manifest"]["source_sha"]!=file_sha256(a.relation_checkpoint) or
            state["manifest"]["train_sha"]!=file_sha256(a.train_jsonl)):
        raise ValueError("Adapter trained on different frozen checkpoint or dataset")
    source=torch.load(a.relation_checkpoint,map_location="cpu",weights_only=True)
    model=load_full_relation_model(source,device).eval().requires_grad_(False)
    adapter=PolygonGuidedLanguageAttention().to(device).eval()
    adapter.load_state_dict(state["adapter"])
    cfg=source["config"]["geometry_args"]
    ds,_,dev=make_dataset(a.train_jsonl,cfg)
    bert=FrozenBERTTokens(cfg["text_backbone"],device=device,
                          max_length=int(cfg["max_text_length"]))
    cached=a.adapter_checkpoint.parent/"bert_cache.pt"
    if cached.exists():bert.load_cache(cached)
    loader=DataLoader(Subset(ds,dev),batch_size=8,shuffle=False,
                      collate_fn=collate_full,num_workers=0)
    rows=[];eligible=0;total=0
    for batch in loader:
        x=frozen_features(model,batch,bert,device)
        original=adapter(**x)["logits"].softmax(-1)
        for i,text in enumerate(batch["text"]):
            total+=1
            proposal=swap_single_cardinal(text,batch["node_names"][i])
            if proposal is None:continue
            eligible+=1
            if len(rows)>=a.max_pairs:continue
            tokens,mask=bert.encode_tokens([proposal["mutated"]])
            row={key:value[i:i+1] for key,value in x.items()}
            row.update(tokens=tokens,token_mask=mask)
            modified=adapter(**row)["logits"].softmax(-1)[0]
            initial=original[i]
            grid=batch["grid_xy"][i].to(device).float()
            map_size=batch["map_sizes"][i].to(device).float()
            world=grid*map_size
            centroid0=(initial[:,None]*world).sum(0)
            centroid1=(modified[:,None]*world).sum(0)
            delta=(centroid0-centroid1).cpu().tolist()
            variation=float(.5*torch.abs(initial-modified).sum())
            direction={"north":[0,1],"south":[0,-1],
                       "east":[1,0],"west":[-1,0]}[proposal["old"]]
            directional_shift=float(delta[0]*direction[0]+delta[1]*direction[1])
            rows.append(dict(
                index=int(dev[total-1]),
                original=proposal["original"],mutated=proposal["mutated"],
                swapped_relation=f'{proposal["old"]}->{proposal["new"]}',
                total_variation=variation,centroid_delta_m=delta,
                direction_signed_shift_m=(
                    directional_shift if a.verify_world_north else None),
                direction_frame_verified=bool(a.verify_world_north),
                scope_status="unique_lexical_phrase; not guaranteed correct relation scope",
            ))
        if len(rows)>=a.max_pairs:break
    summary=dict(
        output_type="fixed_geometry_cardinal_swap_language_head_only",
        evaluated_pairs=len(rows),qualified_pairs_seen=eligible,
        total_dev_queries_scanned=total,world_north_verified=a.verify_world_north,
        mean_total_variation=float(np.mean([r["total_variation"] for r in rows])) if rows else None,
        mean_signed_direction_shift_m=(
            float(np.mean([r["direction_signed_shift_m"] for r in rows]))
            if rows and a.verify_world_north else None),
        source_sha=file_sha256(a.relation_checkpoint),
        adapter_sha=file_sha256(a.adapter_checkpoint),
        warnings=["TRAIN-Dev only; no Val/Test",
                  "Same frozen base in original/counterfactual: isolates new attention",
                  "Shallow lexical scope only: inspect examples, do not infer truth",
                  "Unverified geographic direction cannot be scored"]),
    a.output.mkdir(parents=True)
    atomic_json(summary,a.output/"summary.json")
    (a.output/"pairs.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
