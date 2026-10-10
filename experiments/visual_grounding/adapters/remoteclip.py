"""Official RemoteCLIP/OpenCLIP encoder for frozen observed CityNav ROIs.

Run with the isolated RemoteCLIP environment; this module never installs packages
or alters existing GraphNav caches. Images are BGR uint8 numpy arrays.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
import open_clip

HF_REVISION = "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
OFFICIAL_COMMIT = "a6a4787507e441f444c20404c90dd18520a8960d"
CHECKPOINT_SHA256 = {
    "ViT-B-32": "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4",
    "ViT-L-14": "fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb",
}


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RemoteCLIPEncoder:
    def __init__(self, model_name="ViT-L-14", checkpoint=None, device="cuda",
                 batch_size=16, precision="bf16"):
        if model_name not in {"ViT-L-14", "ViT-B-32"}:
            raise ValueError(f"Unsupported main-comparison model: {model_name}")
        if checkpoint is None:
            raise ValueError("An explicit pinned official RemoteCLIP checkpoint is required")
        self.model_name = model_name
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = _sha256(self.checkpoint)
        if self.checkpoint_sha256 != CHECKPOINT_SHA256[model_name]:
            raise ValueError("Checkpoint does not match the pinned official RemoteCLIP weights")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.precision = precision
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None)
        status = self.model.load_state_dict(torch.load(
            self.checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.load_status = str(status)
        self.model.to(self.device).eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.timings = {"images_seconds": 0., "texts_seconds": 0.,
                        "images_count": 0, "texts_count": 0}
        self.dimension = int(self.model.text_projection.shape[-1])

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == "cuda" and self.precision == "bf16")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def images(self, bgr_images):
        if not bgr_images:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._sync()
        start = time.perf_counter()
        output = []
        for offset in range(0, len(bgr_images), self.batch_size):
            batch = []
            for array in bgr_images[offset:offset + self.batch_size]:
                if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                    raise ValueError("ROI input must be a HxWx3 BGR uint8 array")
                batch.append(self.preprocess(Image.fromarray(array[:, :, ::-1].copy())))
            x = torch.stack(batch).to(self.device)
            with self._autocast():
                feat = self.model.encode_image(x)
            feat = torch.nn.functional.normalize(feat.float(), dim=-1)
            if not torch.isfinite(feat).all():
                raise FloatingPointError("Nonfinite RemoteCLIP image embeddings")
            output.append(feat.cpu().numpy())
        self._sync()
        self.timings["images_seconds"] += time.perf_counter() - start
        self.timings["images_count"] += len(bgr_images)
        return np.concatenate(output)

    @torch.inference_mode()
    def texts(self, strings):
        if not strings:
            return np.empty((0, self.dimension), dtype=np.float32)
        self._sync()
        start = time.perf_counter()
        output = []
        for offset in range(0, len(strings), self.batch_size):
            tokens = self.tokenizer(strings[offset:offset + self.batch_size]).to(self.device)
            with self._autocast():
                feat = self.model.encode_text(tokens)
            feat = torch.nn.functional.normalize(feat.float(), dim=-1)
            if not torch.isfinite(feat).all():
                raise FloatingPointError("Nonfinite RemoteCLIP text embeddings")
            output.append(feat.cpu().numpy())
        self._sync()
        self.timings["texts_seconds"] += time.perf_counter() - start
        self.timings["texts_count"] += len(strings)
        return np.concatenate(output)

    def provenance(self):
        return {"model": f"RemoteCLIP {self.model_name}",
                "official_repository": "https://github.com/ChenDelong1999/RemoteCLIP",
                "official_commit": OFFICIAL_COMMIT,
                "weights_repository": "chendelong/RemoteCLIP", "hf_revision": HF_REVISION,
                "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
                "code_license": "Apache-2.0", "weights_license": "not separately specified on HF model card",
                "torch": torch.__version__, "open_clip_torch": importlib.metadata.version("open_clip_torch"),
                "device": str(self.device), "precision": self.precision,
                "image_size": list(self.model.visual.image_size), "dimension": self.dimension,
                "text_context_length": int(self.model.context_length),
                "text_truncation": "official OpenCLIP tokenizer truncates at model context length",
                "trainable_parameters": 0, "parameters": sum(p.numel() for p in self.model.parameters()),
                "load_status": self.load_status, "timings": dict(self.timings)}


Encoder = RemoteCLIPEncoder
