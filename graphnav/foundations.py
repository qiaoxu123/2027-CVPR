"""Optional frozen pretrained encoders, cached at the episode/patch level.

Transformers is deliberately optional for geometry-only CPU tests. Production
navigation should use trained HF weights, not the test-only synthetic tensors.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from hashlib import blake2b

import numpy as np
import torch
import torch.nn.functional as F


class FrozenBERT:
    def __init__(self, name="bert-base-uncased", device="cuda", max_length=96):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.model.requires_grad_(False)
        self.cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def encode(self, sentences: Iterable[str]) -> torch.Tensor:
        texts = list(sentences)
        unknown = list(dict.fromkeys(s for s in texts if s not in self.cache))
        if unknown:
            batch = self.tokenizer(
                unknown, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**batch)
            # CLS is a view into [batch,tokens,hidden]. Copy it before caching
            # so each entry does not retain all token states on the GPU.
            vectors = outputs.last_hidden_state[:, 0].float().clone()
            for text, feature in zip(unknown, vectors):
                self.cache[text] = feature.detach()
        return torch.stack([self.cache[s] for s in texts], dim=0)


class FrozenSigLIP2:
    """Vision-only frozen SigLIP2. The caller MUST supply observed-only crops."""

    def __init__(self, name="google/siglip2-base-patch16-224", device="cuda"):
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, crops: list[np.ndarray]) -> torch.Tensor:
        from PIL import Image

        if not crops:
            return torch.empty((0, self.model.config.projection_dim), device=self.device)
        pil = [Image.fromarray(arr, mode="RGB") for arr in crops]
        batch = self.processor(images=pil, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**batch)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return F.normalize(features.float(), dim=-1)


class CandidateVisionCache:
    """LRU cache keyed by exact observed RGB and validity mask, never map coordinates alone."""

    def __init__(self, limit=4096):
        self.limit = limit
        self._data: OrderedDict[bytes, torch.Tensor] = OrderedDict()

    @staticmethod
    def key(image, valid):
        h = blake2b(digest_size=16)
        h.update(np.ascontiguousarray(image).tobytes())
        h.update(np.ascontiguousarray(valid).tobytes())
        return h.digest()

    def get(self, image, valid):
        key = self.key(image, valid)
        result = self._data.get(key)
        if result is not None:
            self._data.move_to_end(key)
        return result

    def put(self, image, valid, features):
        key = self.key(image, valid)
        self._data[key] = features.detach().cpu()
        self._data.move_to_end(key)
        while len(self._data) > self.limit:
            self._data.popitem(last=False)
