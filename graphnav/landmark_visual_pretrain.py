"""P2: same-instance cross-view retrieval + object-type text alignment.

Supervision comes from SOURCE map polygon projection to genuinely observed
CityFlight RGB. Polygon ROI is a *geometric pseudo-label*, not a verified
semantic segmentation. It must be audited before interpreting performance.
Object IDs are used ONLY to form positive pairs and evaluation labels.
No final target IDs, ground-truth destination, future RGB, or trajectory
waypoint enters the model forward pass.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import inspect
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class FrozenSigLIP2Patches:
    """Frozen SigLIP2 vision tower patch tokens, NOT pooled get_image_features."""

    def __init__(self, name="google/siglip2-large-patch16-384",
                 device="cuda", cache_limit=2048, microbatch=8):
        from transformers import AutoModel, AutoProcessor
        self.device=torch.device(device)
        self.processor=AutoProcessor.from_pretrained(name,use_fast=False)
        self.model=AutoModel.from_pretrained(name).to(self.device).eval()
        self.model.requires_grad_(False)
        if not hasattr(self.model,"vision_model"):
            raise RuntimeError("Requested backbone has no exposed vision_model; cannot extract patches")
        self.cache=OrderedDict()
        self.cache_limit=int(cache_limit)
        self.microbatch=int(microbatch)
        if self.microbatch<1:
            raise ValueError("invalid vision encoder microbatch")
        self.hidden_size=int(self.model.config.vision_config.hidden_size)

    @torch.inference_mode()
    def encode(self,paths):
        missing=list(dict.fromkeys(p for p in paths if p not in self.cache))
        allowed=inspect.signature(self.model.vision_model.forward).parameters
        for start in range(0,len(missing),self.microbatch):
            chunk=missing[start:start+self.microbatch]
            rgb=[]
            for path in chunk:
                bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(path)
                rgb.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            batch=self.processor(images=rgb,return_tensors="pt")
            kwargs={key:value.to(self.device) for key,value in batch.items()
                    if key in allowed}
            if "pixel_values" not in kwargs:
                raise ValueError("SigLIP2 processor must supply pixel_values")
            output=self.model.vision_model(**kwargs)
            tokens=output.last_hidden_state
            if tokens.ndim!=3:
                raise RuntimeError("vision_model did not return [B,P,D] patch tokens")
            mask=batch.get("pixel_attention_mask")
            for i,path in enumerate(chunk):
                item=(
                    tokens[i].detach().to("cpu",dtype=torch.float16).clone(),
                    (mask[i].detach().cpu().bool().clone()
                     if mask is not None and mask.ndim==2 and mask.shape[1]==tokens.shape[1]
                     else torch.ones(tokens.shape[1],dtype=torch.bool))
                )
                self.cache[path]=item
                self.cache.move_to_end(path)
            while len(self.cache)>self.cache_limit:
                self.cache.popitem(last=False)
        output=[self.cache[p] for p in paths]
        length=max(x.shape[0] for x,_ in output)
        patches=torch.zeros(len(output),length,self.hidden_size,dtype=torch.float32)
        valid=torch.zeros(len(output),length,dtype=torch.bool)
        for i,(tensor,mask) in enumerate(output):
            patches[i,:len(tensor)]=tensor.float()
            valid[i,:len(mask)]=mask
            self.cache.move_to_end(paths[i])
        return patches.to(self.device),valid.to(self.device)


class PairedROIs(Dataset):
    """Each sample contains two different observed views of the SAME map ID."""

    def __init__(self,jsonl_path,*,min_visible=.80,seed=0):
        groups=defaultdict(list)
        for line in Path(jsonl_path).open(encoding="utf-8"):
            if not line.strip():
                continue
            row=json.loads(line)
            if (row.get("roi_path")
                and row.get("visible_fraction",0)>=min_visible
                and row.get("annotation_kind")=="geometry_projected_roi_not_verified_segmentation"):
                groups[(row["map_name"],str(row["object_id"]))].append(row)
        self.groups=[]
        for key,rows in sorted(groups.items()):
            unique={r["observed_frame"]:r for r in rows}
            if len(unique)>=2:
                self.groups.append((key,list(unique.values())))
        if not self.groups:
            raise ValueError("Need >=2 DIFFERENT observed frames per landmark ID; use along-trajectory data")
        self.seed=seed
        self.types=sorted({rows[0]["object_type"] for _,rows in self.groups})
        self.labels={v:i for i,v in enumerate(self.types)}
        self.maps={key[0] for key,_ in self.groups}

    def __len__(self):
        return len(self.groups)

    def __getitem__(self,index):
        key,rows=self.groups[index]
        # Index-stable but varied two-view pairs (no reliance on GT goal).
        # Train data loader shuffling changes pair order; frozen source frames
        # cannot be invented from reversed directions.
        chosen=random.sample(rows,2)
        if chosen[0]["observed_frame"]==chosen[1]["observed_frame"]:
            raise AssertionError("positive visual pair must use different views")
        return (
            chosen[0]["roi_path"], chosen[1]["roi_path"],
            self.labels[rows[0]["object_type"]],
            key,
        )


def roi_collate(items):
    return {
        "first":[x[0] for x in items],
        "second":[x[1] for x in items],
        "type":torch.tensor([x[2] for x in items],dtype=torch.long),
        "source_ids":[x[3] for x in items],
    }


class VisualLandmarkAligner(nn.Module):
    """Image patch attention + text-type alignment and cross-view InfoNCE."""

    def __init__(self,vision_dim=1024,text_dim=768,hidden=256,heads=8):
        super().__init__()
        self.patch=nn.Sequential(nn.Linear(vision_dim,hidden),nn.LayerNorm(hidden))
        self.query=nn.Parameter(torch.randn(1,1,hidden)*.01)
        self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.image_proj=nn.Sequential(nn.LayerNorm(hidden),
                                      nn.Linear(hidden,hidden))
        self.type_proj=nn.Sequential(nn.Linear(text_dim,hidden),
                                     nn.LayerNorm(hidden))
        self.logit_scale=nn.Parameter(torch.tensor(2.3))

    def forward(self,patches,valid):
        x=self.patch(patches)
        q=self.query.expand(x.shape[0],-1,-1)
        pooled,_=self.attn(q,x,x,key_padding_mask=~valid.bool(),need_weights=False)
        return F.normalize(self.image_proj(pooled[:,0]),dim=-1)

    def type_logits(self,images,all_type_features):
        types=F.normalize(self.type_proj(all_type_features),dim=-1)
        return self.logit_scale.exp().clamp(max=100.)*(images @ types.T)


def match_loss(model,first,second,types,all_type_features,*,type_weight=.25):
    """Same-ID positives across time, other IDs in batch are negatives."""
    b=len(first)
    if b<2:
        raise ValueError("contrastive train batch needs at least two distinct IDs")
    if torch.unique(types).numel()<1:
        raise ValueError("empty class labels")
    logits=model.logit_scale.exp().clamp(max=100.)*(first @ second.T)
    correct=torch.arange(b,device=first.device)
    contrastive=(F.cross_entropy(logits,correct)+
                 F.cross_entropy(logits.T,correct))*.5
    type_loss=(F.cross_entropy(model.type_logits(first,all_type_features),types)
              +F.cross_entropy(model.type_logits(second,all_type_features),types))*.5
    return contrastive+type_weight*type_loss,dict(
        instance=float(contrastive.detach()),type=float(type_loss.detach()),
    )


@torch.no_grad()
def retrieval_metrics(model,loader,encoder,device,max_groups=1500):
    """One view/gallery pair per instance; gallery across ALL chosen unseen IDs."""
    model.eval()
    queries,gallery,keys=[],[],[]
    for batch in loader:
        p,m=encoder.encode(batch["first"])
        q,valid=encoder.encode(batch["second"])
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=(device.type=="cuda")):
            queries.append(model(p,m).float().cpu())
            gallery.append(model(q,valid).float().cpu())
        keys.extend(batch["source_ids"])
        if len(keys)>=max_groups:
            break
    if not queries:
        raise ValueError("no evaluable two-view ROIs")
    x=torch.cat(queries)[:max_groups]
    y=torch.cat(gallery)[:max_groups]
    n=len(x)
    ranks=(x@y.T).argsort(dim=-1,descending=True)
    true=torch.arange(n)
    return {"gallery_objects":n,
            "R@1":float((ranks[:,0]==true).float().mean()),
            "R@5":float((ranks[:,:min(5,n)]==true[:,None]).any(-1).float().mean()),
            "protocol":"different-frame instance-ID retrieval, no GT target images"}
