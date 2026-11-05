"""Fixed-Polygon, four-word Belief sweep from trained synthetic direction probe.

NO target labels are read or used in inference. The *same* public anchor
Polygon, map bounds, and vocabulary remain fixed; only direction changes.
This is a synthetic map-XY directional probe, not a natural-language
CityNav benchmark and not GPS/geographic north verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from graphnav.inverse_polygon_direction_data import DIR, instruction
from graphnav.inverse_polygon_direction_model import (
    DirectionalPolygonBelief, example_features,
)


def run_sweep(model, text_emb, input_row, *, device="cpu"):
    """Return Belief for four direction variants, without accessing target label."""
    source=input_row["input"]
    bounds=input_row["world_bounds"]
    anchor=source["anchor_contour"]
    # Only inputs, never the original target contour/coordinates.
    mock_input={
        "input":{"anchor_contour":anchor},
        "world_bounds":bounds,
    }
    from graphnav.inverse_polygon_direction_model import (
        map_grid,candidate_polygon_vectors,outline_tokens,
    )
    grid=map_grid(bounds)
    poly=outline_tokens([anchor],bounds)
    geo=candidate_polygon_vectors(grid,[anchor],bounds)
    b=torch.tensor(np.tile(geo,(len(DIR),1,1)),device=device,dtype=torch.float32)
    outline=torch.tensor(np.tile(poly["outline"],(len(DIR),1,1,1)),device=device)
    part_mask=torch.tensor(np.tile(poly["part_mask"],(len(DIR),1)),device=device)
    areas=torch.tensor(np.tile(poly["area_ratio"],(len(DIR),1)),device=device)
    ids=torch.arange(len(DIR),device=device)
    with torch.no_grad():
        scores=model(b,outline,part_mask,areas,ids,text_emb.to(device))
        probabilities=scores.softmax(-1).cpu().numpy()
    return {
        word:dict(
            probability=p.reshape(28,28),
            max_xy=grid[int(p.argmax())].tolist(),
            centroid_xy=(p[:,None]*grid).sum(0).tolist(),
            peak_probability=float(p.max()),
        )
        for word,p in zip(DIR,probabilities)
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--dev-jsonl",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--family-id",default=None)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    checkpoint=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if "bert_direction_embeddings" not in checkpoint:
        raise ValueError("Checkpoint predates stored frozen relation embeddings; retrain")
    device=torch.device(a.device)
    emb=checkpoint["bert_direction_embeddings"].to(device)
    model=DirectionalPolygonBelief(text_dim=emb.shape[1]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    candidates=[]
    for ln in a.dev_jsonl.read_text().splitlines():
        if not ln.strip():continue
        row=json.loads(ln)
        if not row["label"]["truth"] or row["source"]["orientation"]!="forward":
            continue
        if a.family_id and row["family_id"]!=a.family_id:continue
        candidates.append(row)
    if not candidates:raise ValueError("No matching dev public-Polygon probe")
    row=candidates[0]
    result=run_sweep(model,emb,row,device=device)
    a.output.mkdir(parents=True,exist_ok=True)
    bounds=row["world_bounds"]
    anchor=np.asarray(row["input"]["anchor_contour"])
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    vmax=max(float(x["probability"].max()) for x in result.values())
    for ax,(word,r) in zip(axes.flat,result.items()):
        image=ax.imshow(r["probability"],origin="lower",extent=bounds,
                        cmap="inferno",vmin=0,vmax=vmax,aspect="auto")
        ring=np.vstack((anchor,anchor[0]))
        ax.plot(ring[:,0],ring[:,1],color="cyan",linewidth=1.5,label="input Polygon")
        ax.scatter(*r["centroid_xy"],c="white",s=20,marker="+",label="Belief centroid")
        ax.set_title(f"{word}: top-1={np.round(r['max_xy'],1).tolist()}")
        ax.set_xlabel("map X (meters)");ax.set_ylabel("map Y (meters)")
    fig.colorbar(image,ax=list(axes.flat),shrink=.8,label="Belief mass per grid cell")
    fig.suptitle("Fixed same input Polygon; only relation phrase changes")
    fig.savefig(a.output/"four_direction_beliefs.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    metrics=dict(
        experiment="synthetic_fixed_polygon_direction_sweep",
        family_id=row["family_id"],source_map=row["map_name"],
        public_anchor_id=row["input"]["anchor_id"],
        geographic_orientation_verified=False,
        trained_epoch=checkpoint["epoch"],
        input_target_label_used=False,
        results={key:{k:v for k,v in x.items() if k!="probability"}
                 for key,x in result.items()}
    )
    (a.output/"four_direction_beliefs.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()
