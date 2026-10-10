"""Frozen SigLIP2 current-frame patches with native attention pooling per ROI.

The transformer sees one actually observed 384px frame. Region masks select
bounding-box patch centers, not claimed semantic segmentation. No destination,
language, target identity, original orthophoto or future frame enters selection.
"""
import hashlib

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.landmark_projection import project_landmark
from graphnav.observed_patch_scales import masked_native_pool


def select_regions(observation, landmarks, max_rois=32):
    rgb = observation['bgr']
    if rgb.shape != (384, 384, 3):
        raise ValueError('native observed 384px pixels required')
    footprint = np.asarray(observation['footprint_world'])
    low, high = footprint.min(0), footprint.max(0)
    candidates = landmarks.candidates(low, high)
    support = rgb.max(-1) > 0
    patch_support = support.reshape(24, 16, 24, 16).mean((1, 3)).ravel() >= .9
    positions = (np.arange(24) + .5) * 16 - .5
    xx, yy = np.meshgrid(positions, positions)
    selected = []
    for landmark in candidates:
        roi = project_landmark(landmark, footprint, rgb.shape, min_visible=.9, min_pixels=256)
        if roi is None:
            continue
        x0, y0, x1, y1 = roi.bbox_xyxy
        side = min(x1-x0, y1-y0)
        if side < 8:
            continue
        fraction = float(support[roi.mask.astype(bool)].mean())
        if fraction < .9:
            continue
        mask = ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).ravel() & patch_support
        if not mask.any():
            continue
        contour = np.asarray(landmark['contour'], np.float32)
        selected.append(dict(object_id=roi.object_id, object_type=roi.object_type,
            name=roi.name, center_world=contour.mean(0).tolist(), bbox=list(roi.bbox_xyxy),
            pixel_size=[x1-x0, y1-y0], native_min_side=side,
            quality=[side/384., fraction, roi.visible_fraction],
            mask=mask, visible_pixels=roi.visible_pixels))
    selected.sort(key=lambda row: (-row['visible_pixels']*row['quality'][1], row['object_id']))
    return selected[:max_rois]


class FrozenROIEncoder:
    def __init__(self, device='cuda', max_rois=32):
        self.native = ObservedSigLIPMatchingEncoder(device=device, microbatch=16)
        self.device = self.native.device
        self.max_rois = max_rois
        self.image_precision = 'fp16_backbone_fp32_head'
        self.revision = getattr(self.native.model.config, '_commit_hash', None)
        if self.revision != '1b426889ea62b5a72bf9839009a1b184bfc9c178':
            raise ValueError('unexpected SigLIP2 revision')

    @torch.inference_mode()
    def encode(self, observations, landmarks):
        rows = [select_regions(o, lm, self.max_rois) for o, lm in zip(observations, landmarks)]
        if len(observations) != len(landmarks):
            raise ValueError('observation/public map alignment')
        batch = self.native.processor(images=[cv2.cvtColor(o['bgr'], cv2.COLOR_BGR2RGB)
                                              for o in observations], return_tensors='pt').to(self.device)
        with torch.autocast('cuda', dtype=torch.float16):
            patches = self.native.model.vision_model(**batch).last_hidden_state
        # Small masked regions can amplify BF16 rounding. The fixed FP16 image
        # transformer + FP32 official attention head passes the FP32 ROI check.
        with torch.autocast('cuda', enabled=False):
            vectors = []
            for patch, regions in zip(patches, rows):
                if regions:
                    masks = torch.tensor(np.stack([r.pop('mask') for r in regions]), device=self.device)
                    pooled = masked_native_pool(self.native.model.vision_model.head, patch.float(), masks)
                    vector = F.normalize(pooled.float(), dim=-1).cpu().half()
                else:
                    vector = torch.zeros(0, 1024, dtype=torch.float16)
                if not torch.isfinite(vector).all():
                    raise ValueError('nonfinite real ROI features')
                vectors.append(vector)
        return list(zip(rows, vectors))

    def texts(self, strings):
        return self.native.texts(strings).detach().cpu().half()


def observation_key(map_name, observation):
    digest = hashlib.sha256(map_name.encode())
    digest.update(np.ascontiguousarray(observation['bgr']).tobytes())
    digest.update(np.asarray(observation['footprint_world'], np.float64).tobytes())
    return digest.hexdigest()
