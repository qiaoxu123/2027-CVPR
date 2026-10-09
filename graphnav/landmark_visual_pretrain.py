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
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def observed_patch_mask(images, image_shape, patch_size):
    """Mask wholly black source holes/letterboxing, never polygon segmentation."""
    height,width = image_shape
    values = np.stack([cv2.resize((x.max(-1)>0).astype(np.float32),(width,height),
                                  interpolation=cv2.INTER_NEAREST) for x in images])
    return F.avg_pool2d(torch.from_numpy(values)[:,None],patch_size,patch_size).flatten(1) > 0


class FrozenSigLIP2Patches:
    """Frozen SigLIP2 vision tower patch tokens, NOT pooled get_image_features."""

    def __init__(self, name="google/siglip2-large-patch16-384",
                 device="cuda", cache_limit=2048, microbatch=8, cache_directory=None):
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
        self.cache_directory = Path(cache_directory) if cache_directory else None
        if self.cache_directory:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.protocol = (name, getattr(self.model.config, "_commit_hash", None),
                         "bf16" if self.device.type == "cuda" else "fp32",
                         "observed-nonblack-patches-v1")
        self.encoded_images = 0
        self.disk_hits = 0

    def disk_path(self, path):
        source = Path(path)
        key = repr((self.protocol, str(source.resolve()), source.stat().st_size,
                    source.stat().st_mtime_ns)).encode()
        return self.cache_directory / (hashlib.sha256(key).hexdigest() + ".pt")

    @torch.inference_mode()
    def encode(self,paths):
        missing=list(dict.fromkeys(p for p in paths if p not in self.cache))
        for path in list(missing):
            if self.cache_directory and self.disk_path(path).exists():
                state = torch.load(self.disk_path(path), map_location="cpu", weights_only=True)
                self.cache[path] = (state["tokens"], state["mask"])
                self.disk_hits += 1
                missing.remove(path)
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
            with torch.autocast(
                "cuda",dtype=torch.bfloat16,enabled=(self.device.type=="cuda")
            ):
                output=self.model.vision_model(**kwargs)
            tokens=output.last_hidden_state
            if tokens.ndim!=3:
                raise RuntimeError("vision_model did not return [B,P,D] patch tokens")
            mask=batch.get("pixel_attention_mask")
            if mask is None:
                patch_size = getattr(getattr(getattr(self.model,"config",None),"vision_config",None),"patch_size",16)
                support = observed_patch_mask(rgb,kwargs["pixel_values"].shape[-2:],patch_size)
                if support.shape[1] == tokens.shape[1]:
                    mask = support
            for i,path in enumerate(chunk):
                item=(
                    tokens[i].detach().to("cpu",dtype=torch.float16).clone(),
                    (mask[i].detach().cpu().bool().clone()
                     if mask is not None and mask.ndim==2 and mask.shape[1]==tokens.shape[1]
                     else torch.ones(tokens.shape[1],dtype=torch.bool))
                )
                self.cache[path]=item
                self.encoded_images += 1
                if self.cache_directory:
                    torch.save({"tokens": item[0], "mask": item[1]}, self.disk_path(path))
                self.cache.move_to_end(path)
        output=[self.cache[p] for p in paths]
        length=max(x.shape[0] for x,_ in output)
        patches=torch.zeros(len(output),length,self.hidden_size,dtype=torch.float32)
        valid=torch.zeros(len(output),length,dtype=torch.bool)
        for i,(tensor,mask) in enumerate(output):
            patches[i,:len(tensor)]=tensor.float()
            valid[i,:len(mask)]=mask
            self.cache.move_to_end(paths[i])
        # Evict only AFTER assembling outputs: a microbatch can exceed the LRU
        # size, and evicting during extraction would remove requested inputs.
        while len(self.cache)>self.cache_limit:
            self.cache.popitem(last=False)
        return patches.to(self.device),valid.to(self.device)


class PairedROIs(Dataset):
    """Each sample contains two different observed views of the SAME map ID."""

    def __init__(self,jsonl_path,*,min_visible=.80,seed=0,deterministic=False):
        groups=defaultdict(list)
        for line in Path(jsonl_path).open(encoding="utf-8"):
            if not line.strip():
                continue
            row=json.loads(line)
            if ("observation_cutoff_step" in row and
                    row["observed_step"] > row["observation_cutoff_step"]):
                raise ValueError("future ROI observation rejected")
            if (row.get("roi_path")
                and row.get("visible_fraction",0)>=min_visible
                and row.get("annotation_kind")=="geometry_projected_roi_not_verified_segmentation"):
                groups[(row["map_name"],str(row["object_id"]))].append(row)
        self.groups=[]
        for key,rows in sorted(groups.items()):
            # Two episode paths can replay the identical camera footprint.
            # They are not distinct visual views, even if filenames differ.
            unique={
                tuple(np.round(np.asarray(r["footprint_world"]).reshape(-1), 3))
                if "footprint_world" in r else r["observed_frame"]: r for r in rows
            }
            if len(unique)>=2:
                self.groups.append((key,list(unique.values())))
        if not self.groups:
            raise ValueError("Need >=2 DIFFERENT observed frames per landmark ID; use along-trajectory data")
        self.seed=seed
        # Shared epoch survives persistent DataLoader workers. Pair selection is
        # independent of worker scheduling, backbone size and model RNG usage.
        self._epoch=torch.zeros((),dtype=torch.long).share_memory_()
        self.deterministic=deterministic
        self.types=sorted({rows[0]["object_type"] for _,rows in self.groups})
        self.labels={v:i for i,v in enumerate(self.types)}
        self.maps={key[0] for key,_ in self.groups}

    def __len__(self):
        return len(self.groups)

    def set_epoch(self,epoch):
        self._epoch.fill_(int(epoch))

    def __getitem__(self,index):
        key,rows=self.groups[index]
        # Index-stable but varied two-view pairs (no reliance on GT goal).
        # Train data loader shuffling changes pair order; frozen source frames
        # cannot be invented from reversed directions.
        chosen=([rows[0], rows[-1]] if self.deterministic
                else random.Random(f'{self.seed}:{int(self._epoch)}:{index}').sample(rows,2))
        if chosen[0]["observed_frame"]==chosen[1]["observed_frame"]:
            raise AssertionError("positive visual pair must use different views")
        return (
            chosen[0]["roi_path"], chosen[1]["roi_path"],
            self.labels[rows[0]["object_type"]],
            key,
        )


class SameMapTypeBatchSampler:
    """Prefer hard negatives: different source IDs, same map AND object type.

    Works with a Subset whose indices point into PairedROIs.groups. No use
    of goal labels, future data or visual similarity when building batches.
    Falls back to same-type across maps if no sufficiently large map bucket.
    """

    def __init__(self, dataset, indices, batch_size=16, seed=0):
        self.dataset, self.indices = dataset, list(indices)
        self.batch_size, self.seed = int(batch_size), seed
        if self.batch_size < 2:
            raise ValueError("batch_size must contain >=2 unique source IDs")
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        buckets = defaultdict(list)
        for position, original_id in enumerate(self.indices):
            key, rows = self.dataset.groups[original_id]
            # Hard negatives from the same CITY and category, different IDs.
            buckets[(key[0].split("_block_")[0], rows[0]["object_type"])].append(position)
        batches = []
        covered = set()
        for entries in buckets.values():
            rng.shuffle(entries)
            for start in range(0, len(entries), self.batch_size):
                group = entries[start:start+self.batch_size]
                if len(group) >= 2:
                    batches.append(group)
                    covered.update(group)
        # A singleton city/type bucket has no valid hard negative. Drop it and
        # report coverage; do not silently mix categories/cities as negatives.
        rng.shuffle(batches)
        yield from batches

    def __len__(self):
        # Conservative estimate; actual count depends on rare map/type buckets.
        return max(1, len(self.indices) // self.batch_size)


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
def retrieval_metrics(model,loader,encoder,device,max_groups=1500, prediction_path=None):
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
    # DataLoader pin_memory recursively converts tuple metadata into lists.
    keys = [tuple(key) for key in keys[:n]]
    dataset = loader.dataset.dataset if hasattr(loader.dataset, "dataset") else loader.dataset
    key_types = {key: rows[0]["object_type"] for key, rows in dataset.groups}
    categories = [key_types[key] for key in keys]
    hits1 = ranks[:,0] == true
    hits5 = (ranks[:,:min(5,n)] == true[:,None]).any(-1)
    by_type = {}
    for typ in sorted(set(categories)):
        indices = torch.tensor([i for i, t in enumerate(categories) if t == typ])
        by_type[typ] = {"queries": len(indices), "R@1": float(hits1[indices].float().mean()),
                        "R@5": float(hits5[indices].float().mean())}
    hard_hits1, hard_hits5, hard_rows = [], [], []
    for i, key in enumerate(keys):
        indices = torch.tensor([j for j, other in enumerate(keys)
            if other[0].split("_block_")[0] == key[0].split("_block_")[0]
            and categories[j] == categories[i]])
        if len(indices) < 2:
            continue
        order = indices[(x[i] @ y[indices].T).argsort(descending=True)]
        hard_hits1.append(int(order[0]) == i)
        hard_hits5.append(bool((order[:5] == i).any()))
        hard_rows.append(len(indices))
    if prediction_path:
        Path(prediction_path).write_text("".join(json.dumps({
            "map_name": key[0], "object_id": key[1], "object_type": categories[i],
            "R1_hit": bool(hits1[i]), "R5_hit": bool(hits5[i]),
            "top5_ids": [keys[j] for j in ranks[i,:5].tolist()]
        }) + "\n" for i, key in enumerate(keys)))
    return {"gallery_objects":n,
            "R@1":float((ranks[:,0]==true).float().mean()),
            "R@5":float((ranks[:,:min(5,n)]==true[:,None]).any(-1).float().mean()),
            "by_type": by_type,
            "same_city_type_hard": {"queries": len(hard_rows),
                "R@1": float(np.mean(hard_hits1)) if hard_rows else None,
                "R@5": float(np.mean(hard_hits5)) if hard_rows else None,
                "mean_gallery": float(np.mean(hard_rows)) if hard_rows else 0},
            "protocol":"different-frame instance-ID retrieval, no GT target images"}
