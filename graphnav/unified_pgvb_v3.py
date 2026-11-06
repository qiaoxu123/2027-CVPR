"""UnifiedPGVB-V3: decoupled GLOBAL geometry + RGB-D visual beliefs.

The same WORLD map bounds index both 28x28 geometry and 384x384 vision.
One frozen SigLIP2 ViT produces image patches and native text embeddings;
the learned depth adapter / cross-attention are lightweight, NOT a second
vision backbone. No GT target image, polygon, coordinate or trajectory enters
forward(). The high-resolution field is a global *distribution*, not a second
camera image or 147456 full geometry/GAT tokens.

Training GT functions are separated below the model and receive labels ONLY
after forward(). The observed RGB-D source is the causal frame renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .losses import goal_distribution
from .semantic_geometry import valid_surface_polygon


SIZE = 384
PATCHES = 24
COARSE = 28


def world_grid(size, *, device, dtype):
    v = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size
    yy, xx = torch.meshgrid(v, v, indexing="ij")
    return torch.stack((xx, yy), -1)


def project_global_to_frame(bounds, footprints, *, size=SIZE):
    """[B,F,384,384,2] normalized frame UV and valid footprint mask.

    The CityNav nadir renderer's quadrilateral is affine for fixed yaw.
    TL,TR,BR,BL ordering is checked; no GT target position is involved.
    """
    if (bounds.ndim != 2 or bounds.shape[1] != 4 or footprints.ndim != 4
            or footprints.shape[-2:] != (4, 2)
            or footprints.shape[0] != bounds.shape[0]):
        raise ValueError("invalid map bounds or RGB-D world footprints")
    if not torch.isfinite(bounds).all() or not torch.isfinite(footprints).all():
        raise ValueError("nonfinite world registration inputs")
    span = bounds[:, 2:] - bounds[:, :2]
    if (span <= 0).any():
        raise ValueError("world_bounds must have positive physical extent")
    b, f = footprints.shape[:2]
    xy = bounds[:, None, None, None, :2] + (
        world_grid(size, device=bounds.device, dtype=bounds.dtype)[None, None]
        * span[:, None, None, None]
    )
    tl = footprints[:, :, None, None, 0, :]
    right = (footprints[:, :, 1, :] - footprints[:, :, 0, :])[:, :, None, None]
    down = (footprints[:, :, 3, :] - footprints[:, :, 0, :])[:, :, None, None]
    determinant = right[..., 0] * down[..., 1] - right[..., 1] * down[..., 0]
    if (determinant.abs() < 1e-5).any():
        raise ValueError("degenerate camera footprint")
    delta = xy - tl
    u = (delta[..., 0] * down[..., 1] - delta[..., 1] * down[..., 0]) / determinant
    v = (right[..., 0] * delta[..., 1] - right[..., 1] * delta[..., 0]) / determinant
    uv = torch.stack((u, v), -1)
    inside = (uv >= 0).all(-1) & (uv <= 1).all(-1)
    return uv * 2.0 - 1.0, inside


@dataclass
class V3Output:
    geometry_logits: torch.Tensor     # [B,784] global
    geometry_global: torch.Tensor     # [B,384,384] same world bounds
    visual_global: torch.Tensor       # [B,384,384] zero for unobserved
    observed_mask: torch.Tensor       # [B,384,384] bool
    fused_logits: torch.Tensor        # [B,384,384] primary goal distribution
    patch_scores: torch.Tensor        # [B,F,576] whole-text conditioned
    patch_features: torch.Tensor      # [B,F,576,H] for GT teacher loss
    patch_valid: torch.Tensor         # [B,F,576]
    patch_world_xy: torch.Tensor      # [B,F,576,2] in world meters
    geometry_context: Optional[torch.Tensor] = None  # [B,784,H], controller export only


class UnifiedPGVBV3(nn.Module):
    """Single ViT feature stream, independent geometry/visual prediction heads.

    geom is V2's FullTokenGeometryLocalizer with its pretrained weights.
    The visual backbone is external and frozen for cached-feature training,
    and remains exactly ONE SigLIP2 model for both image/text embedding.
    """
    def __init__(self, geometry: nn.Module, *, vision_dim=1024, hidden=128,
                 size=SIZE, patch_grid=PATCHES):
        super().__init__()
        if size != SIZE or patch_grid != PATCHES:
            raise ValueError("V3 pilot uses native384 / SigLIP2 patch16 only")
        self.geometry = geometry
        self.vision_proj = nn.Sequential(nn.Linear(vision_dim, hidden),
                                         nn.LayerNorm(hidden))
        self.depth_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(),
                                        nn.Linear(hidden, hidden))
        self.pose_proj = nn.Linear(5, hidden)
        self.bert_to_visual = nn.Linear(768, hidden)
        self.cross_attention = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True)
        self.visual_text = nn.Sequential(nn.Linear(vision_dim, hidden),
                                         nn.LayerNorm(hidden))
        self.visual_head = nn.Sequential(nn.LayerNorm(hidden),
                                         nn.Linear(hidden, hidden), nn.GELU())
        self.teacher_head = nn.Linear(vision_dim, hidden)
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.visual_gain = nn.Parameter(torch.tensor(0.7))

    def forward(
        self, *, static_inputs, bert_tokens, bert_mask, landmark_embeddings,
        siglip_text, rgb_patches, depth_patches, patch_valid,
        footprints_world, frame_poses, world_bounds,
    ) -> V3Output:
        """GT-FREE. Only public landmark geometry and actual sampled RGB-D.

        Shapes: rgb_patches [B,F,576,1024]; depth [B,F,576,2]
        (depth in 100m units, valid fraction); masks [B,F,576]; footprints
        [B,F,4,2]; poses [B,F,4]=world(x,y,z,yaw).
        """
        b, frames, n, d = rgb_patches.shape
        if (n != PATCHES**2 or d != self.vision_proj[0].in_features or
                depth_patches.shape != (b, frames, n, 2) or
                patch_valid.shape != (b, frames, n) or
                frame_poses.shape != (b, frames, 4) or
                siglip_text.shape != (b, d) or
                bert_tokens.shape[:2] != bert_mask.shape or
                bert_tokens.shape[0] != b or
                world_bounds.shape != (b, 4) or
                footprints_world.shape != (b, frames, 4, 2) or
                patch_valid.dtype != torch.bool or bert_mask.dtype != torch.bool):
            raise ValueError("V3 observed feature dimensions mismatch")
        if not bert_mask.any(-1).all():
            raise ValueError("each original instruction needs valid text tokens")
        for value in (rgb_patches, depth_patches, frame_poses,
                      world_bounds, footprints_world, bert_tokens, siglip_text):
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite inputs in V3 forward")
        # No rule-parser fields are allowed to drive V3's geometry branch.
        # Everything is original BERT tokens + public whole Polygon nodes.
        static = dict(static_inputs)
        static["rules"] = torch.zeros_like(static_inputs["rules"])
        static["relation_fields"] = torch.zeros_like(
            static_inputs["relation_fields"])
        field = self.geometry(
            static, variant="geometry", tokens=bert_tokens,
            token_mask=bert_mask, node_emb=landmark_embeddings,
        )
        geometry_logits = field.belief_logits.float()
        if geometry_logits.shape != (b, COARSE**2):
            raise ValueError("V3 requires V2 28x28 geometry")
        geom = F.interpolate(
            geometry_logits.reshape(b, 1, COARSE, COARSE),
            size=(SIZE, SIZE), mode="bilinear", align_corners=False,
        )[:, 0]
        geom = geom - geom.mean(dim=(-2, -1), keepdim=True)

        # Pose enters only through legally sampled frames and their known
        # registration. Fixed yaw is enforced in the data/observation builder.
        span = world_bounds[:, 2:] - world_bounds[:, :2]
        if (span <= 0).any():
            raise ValueError("invalid world bounds")
        normalized_pose = (frame_poses[..., :2] - world_bounds[:, None, :2]) / span[:, None]
        yaw = frame_poses[..., 3]
        pose = torch.stack((
            normalized_pose[..., 0], normalized_pose[..., 1],
            frame_poses[..., 2] / 100.0, yaw.sin(), yaw.cos(),
        ), -1)
        vision = (self.vision_proj(rgb_patches.float())
                  + self.depth_proj(depth_patches.float())
                  + self.pose_proj(pose)[:, :, None, :])
        query = vision.reshape(b, frames * n, -1)
        keys = self.bert_to_visual(bert_tokens.float())
        attn, _ = self.cross_attention(
            query, keys, keys, key_padding_mask=~bert_mask, need_weights=False)
        features = F.normalize(
            self.visual_head(query + attn).reshape(b, frames, n, -1), dim=-1)
        lang = F.normalize(self.visual_text(siglip_text.float()), dim=-1)
        scores = self.scale.exp().clamp(max=100.) * (
            features * lang[:, None, None]).sum(-1)
        scores = scores.masked_fill(~patch_valid, 0.)

        global_grid, inside = project_global_to_frame(world_bounds, footprints_world)
        sampled = F.grid_sample(
            scores.reshape(b * frames, 1, PATCHES, PATCHES),
            global_grid.reshape(b * frames, SIZE, SIZE, 2),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        ).reshape(b, frames, SIZE, SIZE)
        valid_grid = F.grid_sample(
            patch_valid.float().reshape(b * frames, 1, PATCHES, PATCHES),
            global_grid.reshape(b * frames, SIZE, SIZE, 2),
            mode="nearest", padding_mode="zeros", align_corners=False,
        ).reshape(b, frames, SIZE, SIZE) > .5
        support = inside & valid_grid
        count = support.float().sum(1)
        visual = (sampled * support.float()).sum(1) / count.clamp_min(1.)
        visible = count > 0
        visual = visual * visible
        fused = geom + torch.tanh(self.visual_gain) * torch.tanh(visual / 5.) * visible

        # World patch locations for loss labels ONLY (not target centers).
        s = (torch.arange(PATCHES, device=vision.device, dtype=vision.dtype) + .5) / PATCHES
        py, px = torch.meshgrid(s, s, indexing="ij")
        uv = torch.stack((px.reshape(-1), py.reshape(-1)), dim=-1)
        tl = footprints_world[:, :, 0, :]
        right = footprints_world[:, :, 1, :] - tl
        down = footprints_world[:, :, 3, :] - tl
        patch_world = (tl[:, :, None] + uv[None, None, :, :1] * right[:, :, None]
                       + uv[None, None, :, 1:] * down[:, :, None])
        return V3Output(
            geometry_logits=geometry_logits, geometry_global=geom,
            visual_global=visual, observed_mask=visible,
            fused_logits=fused, patch_scores=scores,
            patch_features=features, patch_valid=patch_valid,
            patch_world_xy=patch_world,
            geometry_context=getattr(field, "dense_context", None),
        )


def raster_target_polygon(vertices_world, bounds, size):
    """TRAIN-only polygon coverage in global +X/+Y map coordinates.

    Invalid CityRefer contours do not become fake GT, and no spatial GT
    selection may happen in the observation builder.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    if vertices_world is None:
        return mask, False
    contour = np.asarray(vertices_world, np.float32)
    bounds = np.asarray(bounds, np.float32)
    if (not valid_surface_polygon(contour) or bounds.shape != (4,)
            or not np.isfinite(bounds).all() or (bounds[2:] <= bounds[:2]).any()):
        return mask, False
    normalized = (contour - bounds[:2]) / (bounds[2:] - bounds[:2])
    if (normalized.max(0) < 0).any() or (normalized.min(0) > 1).any():
        return mask, False
    aa = 4 if size <= COARSE else 2
    hi = np.zeros((size * aa, size * aa), np.uint8)
    vertices = np.rint(normalized * (size * aa) - .5).astype(np.int32)
    cv2.fillPoly(hi, [vertices], 1)
    mask = cv2.resize(hi.astype(np.float32), (size, size),
                      interpolation=cv2.INTER_AREA)
    return mask, bool(mask.sum() > 0)


def target_mask_batch(polygons, bounds, *, size, device):
    if len(polygons) != len(bounds):
        raise ValueError("Polygon and bounds batches differ")
    a = [raster_target_polygon(p, bb, size) for p, bb in zip(polygons, bounds)]
    return (torch.from_numpy(np.stack([x[0] for x in a])).to(device),
            torch.tensor([x[1] for x in a], device=device, dtype=torch.bool))


def region_mass_loss(logits, masks, valid):
    """Weak GT Polygon: put probability mass inside region; NOT mask BCE/Dice."""
    if logits.ndim != 3 or logits.shape != masks.shape or valid.shape != (len(logits),):
        raise ValueError("region mass loss shape mismatch")
    if not valid.any():
        return logits.sum() * 0.
    support = F.max_pool2d(masks.float()[:, None], 3, padding=1, stride=1)[:, 0]
    mass = (logits.float().flatten(1).softmax(-1) * support.flatten(1)).sum(-1)
    return -mass[valid].clamp_min(1e-9).log().mean()


def dense_goal_ce(logits, target_xy, *, sigma_pixels=1.5):
    if (logits.ndim != 3 or logits.shape[1] != logits.shape[2] or
            target_xy.shape != (len(logits), 2) or sigma_pixels <= 0):
        raise ValueError("invalid dense goal loss")
    # No materialized [B,H*W] target from an unrealistically tiny sigma.
    grid = world_grid(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    dist2 = ((grid[None] - target_xy[:, None, None])**2).sum(-1)
    sigma = sigma_pixels / logits.shape[-1]
    target = torch.softmax((-dist2 / (2*sigma*sigma)).flatten(1), -1)
    return -(target * logits.float().flatten(1).log_softmax(-1)).sum(-1).mean()


def visual_patch_supervision(
    output: V3Output, polygons, bounds, teacher, teacher_head,
    *, teacher_weight=0.10,
):
    """Target-visible-only native patch contrast and GT-image Teacher InfoNCE.

    GT teacher is read here, AFTER model.forward. Never train on patches of
    offscreen target. Polygon overlap is a weak visibility proxy, not proof
    that a car's RGB appearance is visible.
    """
    b, frames, n = output.patch_scores.shape
    if teacher.shape[0] != b or len(polygons) != b or len(bounds) != b:
        raise ValueError("private labels must match batch")
    positives = torch.zeros(b, frames, n, device=output.patch_scores.device,
                            dtype=torch.bool)
    for i, (poly, bound) in enumerate(zip(polygons, bounds)):
        if poly is None or not valid_surface_polygon(np.asarray(poly, np.float32)):
            continue
        # Patch-area weak occupancy, not just patch-center containment.
        # Small cars can cover only a fraction of one native 16x16 ViT patch.
        # This radius comes from OBSERVATION pixel footprints, not GT crops;
        # it deliberately marks uncertain neighbors as possible positives.
        from .semantic_geometry import geometry_distance
        world = output.patch_world_xy[i].detach().cpu().numpy().reshape(
            frames, PATCHES, PATCHES, 2)
        dx = np.linalg.norm(world[:, 0, 1] - world[:, 0, 0], axis=-1)
        dy = np.linalg.norm(world[:, 1, 0] - world[:, 0, 0], axis=-1)
        patch_radius = .5 * np.sqrt(dx**2 + dy**2)
        dist = geometry_distance(world.reshape(frames, n, 2),
                                 np.asarray(poly, np.float32))
        good = dist <= patch_radius[:, None]
        positives[i] = torch.as_tensor(
            good, dtype=torch.bool, device=positives.device)
    positive = positives & output.patch_valid
    negative = output.patch_valid & ~positive
    eligible = positive.any(dim=(1, 2)) & negative.any(dim=(1, 2))
    if not eligible.any():
        zero = output.patch_scores.sum() * 0.
        return dict(text_contrast=zero, gt_image_contrast=zero,
                    eligible=eligible, positive_patches=positive.sum().item())
    logits = output.patch_scores.flatten(1)
    positive_flat = positive.flatten(1)
    valid_flat = output.patch_valid.flatten(1)
    text_loss = (
        torch.logsumexp(logits.masked_fill(~valid_flat, -1e4), -1)
        - torch.logsumexp(logits.masked_fill(~positive_flat, -1e4), -1)
    )[eligible].mean()
    projected = F.normalize(teacher_head(teacher.detach().float()), dim=-1)
    similarity = (output.patch_features * projected[:, None, None]).sum(-1) / .1
    sim = similarity.flatten(1)
    teacher_eligible = eligible & torch.isfinite(teacher).all(-1) & (
        teacher.norm(dim=-1) > 1e-5)
    if teacher_eligible.any():
        teacher_loss = (
            torch.logsumexp(sim.masked_fill(~valid_flat, -1e4), -1)
            - torch.logsumexp(sim.masked_fill(~positive_flat, -1e4), -1)
        )[teacher_eligible].mean()
    else:
        teacher_loss = output.patch_scores.sum() * 0.
    return dict(text_contrast=text_loss,
                gt_image_contrast=teacher_weight * teacher_loss,
                eligible=eligible, positive_patches=int(positive.sum()))


def v3_losses(output, target_xy, target_polygons, world_bounds, teacher_features,
              teacher_head, *, geo_poly_weight=.2, fused_weight=1.,
              fused_poly_weight=.25, visual_weight=.2):
    """Separate supervised branches + fused global goal, no Stop/navigation."""
    device = output.geometry_logits.device
    target_xy = target_xy.to(device).float()
    bb = world_bounds.to(device).float()
    masks28, ok28 = target_mask_batch(target_polygons, bb.detach().cpu().numpy(),
                                    size=COARSE, device=device)
    masks384, ok384 = target_mask_batch(target_polygons, bb.detach().cpu().numpy(),
                                      size=SIZE, device=device)
    coarse = output.geometry_logits.reshape(-1, COARSE, COARSE)
    geo_ce = -(goal_distribution(target_xy, COARSE) *
               output.geometry_logits.float().log_softmax(-1)).sum(-1).mean()
    geo_poly = region_mass_loss(coarse, masks28, ok28)
    fused_ce = dense_goal_ce(output.fused_logits, target_xy)
    fused_poly = region_mass_loss(output.fused_logits, masks384, ok384)
    visual = visual_patch_supervision(
        output, target_polygons, bb.detach().cpu().numpy(), teacher_features,
        teacher_head)
    total = (geo_ce + geo_poly_weight * geo_poly
             + fused_weight * fused_ce + fused_poly_weight * fused_poly
             + visual_weight * (visual["text_contrast"] +
                                visual["gt_image_contrast"]))
    return dict(total=total, geo_ce=geo_ce, geo_polygon=geo_poly,
                fused_ce=fused_ce, fused_polygon=fused_poly,
                text_visual=visual["text_contrast"],
                gt_image_visual=visual["gt_image_contrast"],
                visual_eligible=visual["eligible"].sum(),
                target_polygon_valid=ok384.sum())


@torch.no_grad()
def offline_topk(output, target_xy, world_bounds, *, branch="fused",
                 k_values=(1,5,20), radius_m=20., target_polygons=None):
    """OFFLINE localization only. Never call it closed-loop SR/OSR."""
    if branch not in ("fused", "geometry"):
        raise ValueError("unknown offline Belief branch")
    source = output.fused_logits if branch == "fused" else output.geometry_global
    b, h, w = source.shape
    if h != w or target_xy.shape != (b, 2) or world_bounds.shape != (b, 4):
        raise ValueError("offline metric shape mismatch")
    logits = source.reshape(b, -1)
    kmax = max(k_values)
    ids = logits.topk(kmax, -1).indices
    xy = torch.stack(((ids % w + .5) / w, (ids // w + .5) / h), -1).float()
    distances = ((xy - target_xy[:, None])
                 * (world_bounds[:, None, 2:] - world_bounds[:, None, :2])).norm(dim=-1)
    summary = {f"R@{k}/20m_count": int(
        (distances[:, :k].min(-1).values <= radius_m).sum()) for k in k_values}
    summary["n"] = b
    summary["hit_flags"] = {str(k):
        (distances[:, :k].min(-1).values <= radius_m).cpu().tolist()
        for k in k_values}
    summary["Top1_NE_m"] = distances[:, 0].cpu().tolist()
    summary["Top1_NE_m_sum"] = float(distances[:, 0].sum())
    summary["Top1_xy"] = xy[:, 0].cpu().tolist()
    if target_polygons is not None:
        masks, valid = target_mask_batch(
            target_polygons, world_bounds.detach().cpu().numpy(),
            size=h, device=logits.device)
        at_top = masks.flatten(1).gather(1, ids) > 0
        summary["Polygon_valid"] = valid.cpu().tolist()
        summary["Polygon_hit_flags"] = {
            str(k): (at_top[:, :k].any(-1) & valid).cpu().tolist()
            for k in k_values
        }
    return summary
