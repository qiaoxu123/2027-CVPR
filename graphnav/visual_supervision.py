"""Frozen SigLIP2 visual evidence for training, assembled causally.

One record exposes only the observed RGB frames up to the CURRENT time.
Training uses the network's own candidate IDs; no GT-guided crop locations.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .foundations import CandidateVisionCache
from .memory import ObservedRGBMemory


def visual_candidate_batch(
    outputs,
    observation_sequences: list[list[dict]],
    world_bounds: list,
    vision_encoder,
    *,
    vision_dim: int,
    min_coverage: float = 0.30,
    crop_extent_m=100.0,
    memory_size=512,
    cache: CandidateVisionCache | None = None,
    encoder_batch_size: int = 32,
):
    """Compute [B,K,V] visual embeddings and [B,K] validity coverage.

    Observation data each: {path: filename, footprint_world: [[x,y] x 4]}.
    No full map raster, future frames, GT targets or teacher trajectories enter.
    """
    if vision_encoder is None:
        raise ValueError("frozen visual encoder required")
    b, k, _ = outputs.candidate_xy.shape
    if len(observation_sequences) != b or len(world_bounds) != b:
        raise ValueError("image sequences and bounds must align with batch")
    cache = cache if cache is not None else CandidateVisionCache()
    device = outputs.candidate_xy.device
    embeddings = torch.zeros((b, k, vision_dim), device=device)
    confidence = torch.zeros((b, k), device=device)
    uncached, assignments = [], []

    for i in range(b):
        sequence = observation_sequences[i]
        if not sequence:
            # Dataset validation requires an explicit unavailable-observation
            # reason. Preserve this query and apply exactly zero RGB residual.
            continue
        bounds = np.asarray(world_bounds[i], dtype=np.float32)
        memory = ObservedRGBMemory(bounds, size=memory_size)
        for observation in sequence:
            bgr = cv2.imread(str(observation["path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(observation["path"])
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            memory.observe(
                image, np.asarray(observation["footprint_world"], dtype=np.float32)
            )
        norm_xy = outputs.candidate_xy[i].detach().cpu().numpy()
        # Exact empty-center screening: the central crop covers at most
        # extent/8 on either side. Outside the observed pixel bounding box
        # (plus sampling tolerance), center coverage is necessarily zero.
        rows, cols = np.nonzero(memory.seen)
        if not len(rows):
            continue
        observed_lo = np.array([cols.min(), rows.min()])
        observed_hi = np.array([cols.max(), rows.max()])
        center_margin = crop_extent_m * (memory.size - 1) / (bounds[2:] - bounds[:2]) / 8 + 1
        world_candidates = bounds[:2] + norm_xy * (bounds[2:] - bounds[:2])
        pixels = memory.world_to_image(world_candidates)
        for j, xy in enumerate(norm_xy):
            if np.any(pixels[j] < observed_lo - center_margin) or np.any(pixels[j] > observed_hi + center_margin):
                continue
            world_xy = world_candidates[j]
            crop = memory.crop(world_xy, extent_m=crop_extent_m)
            coverage = min(crop.coverage, crop.center_coverage)
            if coverage < min_coverage:
                continue
            confidence[i, j] = coverage
            hit = cache.get(crop.image, crop.valid)
            if hit is not None:
                embeddings[i, j] = hit.to(device)
            else:
                uncached.append(crop.image)
                assignments.append((i, j, crop))
    if uncached:
        if encoder_batch_size < 1:
            raise ValueError("encoder_batch_size must be positive")
        # Bound frozen ViT inference memory even with 16xTop20 valid crops.
        for start in range(0, len(uncached), encoder_batch_size):
            images = uncached[start:start + encoder_batch_size]
            matching = assignments[start:start + encoder_batch_size]
            vectors = vision_encoder.encode(images)
            if vectors.shape != (len(images), vision_dim):
                raise ValueError("SigLIP2 embedding width does not match config")
            for (i, j, crop), vector in zip(matching, vectors):
                embeddings[i, j] = vector.to(device)
                cache.put(crop.image, crop.valid, vector)
    return embeddings, confidence
