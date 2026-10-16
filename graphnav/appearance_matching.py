"""Frozen native SigLIP2 space and a small observed-only multi-scale adapter.

No target coordinates/IDs, source raster, future view or route is accepted.
Invalid scales are unknown. Initial scores preserve native shared-space matching.
"""
from collections import OrderedDict
import hashlib

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleAppearanceMatcher(nn.Module):
    def __init__(self, dimension=1024, hidden=64, quality_dim=3):
        super().__init__()
        self.dimension = dimension
        self.image_residual = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                            nn.Linear(hidden, dimension))
        self.text_residual = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                           nn.Linear(hidden, dimension))
        self.scale_quality = nn.Linear(quality_dim, 1)
        for layer in (self.image_residual[-1], self.text_residual[-1], self.scale_quality):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, text, images, valid, quality):
        if (text.ndim != 2 or images.ndim != 4 or images.shape[0] != text.shape[0]
                or text.shape[-1] != self.dimension or images.shape[-1] != self.dimension):
            raise ValueError('expected text[B,D] and images[B,C,S,D] in native shared space')
        if valid.shape != images.shape[:3] or valid.dtype != torch.bool:
            raise ValueError('actual observed-scale boolean validity required')
        if quality.shape != (*valid.shape, self.scale_quality.in_features):
            raise ValueError('scale quality must align with each actual ROI')
        if not all(torch.isfinite(t).all() for t in (text, images, quality)):
            raise ValueError('nonfinite matching input')
        language = F.normalize(text.float() + .1*self.text_residual(text.float()), dim=-1)
        # Process supported scales only: most global grid cells are unknown.
        # Invalid vectors are never substituted for a negative observation.
        if images.shape[1]>8192:
            per_scale=torch.zeros_like(valid,dtype=torch.float32)
            if valid.any():
                present=images[valid].float()
                visual=F.normalize(present+.1*self.image_residual(present),dim=-1)
                batch_ids=valid.nonzero()[:,0]
                per_scale[valid]=(visual*language[batch_ids]).sum(-1)
        else:
            visual = torch.zeros_like(images, dtype=torch.float32)
            if valid.any():
                present = images[valid].float()
                visual[valid] = F.normalize(present + .1*self.image_residual(present), dim=-1)
            per_scale = (visual * language[:, None, None]).sum(-1)
        weights = self.scale_quality(quality.float()).squeeze(-1).masked_fill(~valid, -1e4)
        weights = weights.softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        scores = (per_scale * weights).sum(-1) * self.log_scale.exp().clamp(max=100)
        known = valid.any(-1)
        # Zero residual + explicit known flag; absent ROI is not a negative.
        return torch.where(known, scores, torch.zeros_like(scores)), known, weights


class ObservedSigLIPMatchingEncoder:
    """Encode supplied current/history RGB, with content-keyed local LRU.

    Caller owns observation provenance. Full-map raster access is absent here.
    Native text and image towers use the same HF checkpoint, not BERT projections.
    """
    def __init__(self, name='google/siglip2-large-patch16-384', device='cuda',
                 microbatch=16, cache_limit=4096):
        from transformers import AutoModel, AutoProcessor
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(name, use_fast=False)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.model.requires_grad_(False)
        self.name, self.microbatch, self.cache_limit = name, int(microbatch), int(cache_limit)
        if min(self.microbatch, self.cache_limit) < 1:
            raise ValueError('positive microbatch and cache limit required')
        self.cache, self.text_cache = OrderedDict(), {}
        self.encoded_images = self.cache_hits = 0
        self.dimension = int(self.model.config.text_config.hidden_size)

    @staticmethod
    def image_key(bgr):
        bgr = np.asarray(bgr)
        if bgr.ndim != 3 or bgr.shape[-1] != 3 or bgr.dtype != np.uint8:
            raise ValueError('actual uint8 BGR image required')
        h = hashlib.sha256(str(bgr.shape).encode())
        h.update(np.ascontiguousarray(bgr).tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def images(self, views):
        if not views:
            return torch.empty(0, self.dimension, device=self.device)
        keys = [self.image_key(v) for v in views]
        missing = {k: v for k, v in zip(keys, views) if k not in self.cache}
        self.cache_hits += len(keys) - len(missing)
        pairs = list(missing.items())
        for offset in range(0, len(pairs), self.microbatch):
            chunk = pairs[offset:offset+self.microbatch]
            batch = self.processor(images=[cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for _,v in chunk],
                                   return_tensors='pt').to(self.device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type=='cuda'):
                result = self.model.get_image_features(**batch)
            if not torch.is_tensor(result):
                result = result.pooler_output
            result = F.normalize(result.float(), dim=-1)
            for (key,_), vector in zip(chunk,result):
                self.cache[key] = vector.detach().cpu().clone()
                self.encoded_images += 1
        result = torch.stack([self.cache[k] for k in keys]).to(self.device)
        for k in keys:
            self.cache.move_to_end(k)
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return result

    @torch.inference_mode()
    def texts(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.text_cache))
        if missing:
            if any(not isinstance(t,str) or not t.strip() for t in missing):
                raise ValueError('nonempty actual language required')
            batch = self.processor(text=missing, padding='max_length', truncation=True,
                                   max_length=self.model.config.text_config.max_position_embeddings,
                                   return_tensors='pt').to(self.device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type=='cuda'):
                result = self.model.get_text_features(**batch)
            if not torch.is_tensor(result):
                result = result.pooler_output
            for t,v in zip(missing,F.normalize(result.float(),dim=-1)):
                self.text_cache[t] = v.detach().clone()
        return torch.stack([self.text_cache[t] for t in texts])
