"""Pure preprocessing for one observed RGB frame and a PUBLIC named Polygon.

Shared by legacy PGVB diagnostic and UnifiedPGVB-v2. This is NOT an
independent neural network or an extra RGB forward. No GT targets are used.
"""
import cv2
import numpy as np
import torch

from .local_landmark_anchor import local_landmark_anchor, local_anchor_tokens
from .native_dense_regions import native_dense_regions, sparse_view
from .observed_patch_scales import masked_native_pool


@torch.no_grad()
def encode_landmark_observation(
    observation, landmark, record, native, device, *, return_metadata=False,
):
    # Existing legacy clients continue to receive two values. UnifiedPGVB
    # optionally receives public projection metadata for monitoring/tracking.
    def pack(inputs, reason, metadata=None):
        return (inputs, reason, metadata) if return_metadata else (inputs, reason)

    if landmark is None:
        return pack(None, "unsupported_or_unresolved_name")
    anchor = local_landmark_anchor(landmark["contours"], observation)
    metadata = None
    if return_metadata:
        metadata = dict(
            anchor_labels=torch.as_tensor(anchor["labels"], device=device,
                                          dtype=torch.int8)[None],
            positive_patches=int(anchor["positive_patches"]),
            negative_patches=int(anchor["negative_patches"]),
            polygon_view_fraction=float(anchor["view_fraction"]),
            polygon_whole_fraction=float(anchor["whole_fraction"]),
            source_support=float(anchor["local_source_support"]),
            visible_pixels=int(anchor["stable_valid_pixels"]),
            registration_only=True,
            actual_visibility=anchor["actual_visibility"],
    )
    if not anchor["pixel_usable"] or anchor["positive_patches"] < 2:
        return pack(None, "no_supported_local_anchor", metadata)
    dense = native_dense_regions(observation)
    sparse = sparse_view(dense)
    valid = (
        np.asarray(sparse["valid"], bool)
        & (np.asarray(sparse["quality"])[:, 0] >= 0.9)
    )
    world = np.asarray(sparse["world"], dtype=np.float32)
    bounds = np.asarray(record["world_bounds"], dtype=np.float32)
    span = bounds[2:] - bounds[:2]
    if not np.isfinite(span).all() or (span <= 0).any():
        raise ValueError("Public world bounds must have positive finite span")
    local_uv = (world - bounds[:2]) / span
    valid &= (
        np.isfinite(local_uv).all(-1)
        & (local_uv >= 0).all(-1)
        & (local_uv <= 1).all(-1)
    )
    if not valid.any():
        return pack(None, "no_valid_visual_patches", metadata)
    bgr = np.asarray(observation["bgr"])
    batch = native.processor(
        images=[cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)],
        return_tensors="pt",
    ).to(device)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        # Exactly ONE RGB forward. All sparse scales pool the SAME 24x24
        # native SigLIP2 patch map.
        patches = native.model.vision_model(**batch).last_hidden_state[0]
        masks = np.asarray(sparse["masks"], bool)
        flat = masks.reshape(-1, masks.shape[-1])
        nonempty = flat.any(-1)
        pooled = torch.zeros(
            (len(flat), native.dimension), device=device,
        )
        if nonempty.any():
            pooled[torch.from_numpy(nonempty).to(device)] = (
                torch.nn.functional.normalize(
                    masked_native_pool(
                        native.model.vision_model.head, patches,
                        torch.from_numpy(flat[nonempty]).to(device),
                    ).float(), dim=-1,
                )
            )
        features = pooled.reshape(len(valid), 3, native.dimension)
        name_text = native.texts([landmark["name"]]).detach()
        type_text = native.texts([landmark["typ"]]).detach()
        polygon = local_anchor_tokens(
            landmark["contours"], observation["footprint_world"],
        )
    return pack(
        dict(
            local_uv=torch.from_numpy(local_uv)[None].to(device),
            image_features=features[None],
            name_text=name_text,
            type_text=type_text,
            polygon_tokens=torch.from_numpy(polygon)[None].to(device),
            valid=torch.from_numpy(valid)[None].to(device),
        ),
        "available", metadata,
    )
