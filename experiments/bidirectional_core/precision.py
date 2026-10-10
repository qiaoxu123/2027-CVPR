"""Actual current-frame native ROI pooling: BF16 cache versus FP32 encoding."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .cache import CachedCoreDataset
from .data import DEFAULT_DATA
from .runtime import public_landmarks
from .vision import FrozenROIEncoder, select_regions
from graphnav.observed_patch_scales import masked_native_pool
from graphnav.renderer import CityNavObservedViewRenderer


@torch.inference_mode()
def run(args):
    cv2.setNumThreads(1); torch.set_num_threads(1)
    dataset = CachedCoreDataset(args.cache, 'E1')
    rows = [dataset[i] for i in range(len(dataset)) if len(dataset[i]['public']['roi_grid_ids'])][:16]
    if len(rows) != 16:
        raise ValueError('16 real observed frame states required for precision smoke')
    raw = json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text())
    landmarks = {m:public_landmarks(raw[m]) for m in {r['metadata']['map_name'] for r in rows}}
    encoder = FrozenROIEncoder(); native = encoder.native
    renderer = CityNavObservedViewRenderer(Path(args.data_root)/'rgbd', Path(args.output).parent/'precision_private', image_size=384)
    image_cosines = []; text_cosines = []; errors = []; counts = []
    alternatives = {'bf16_backbone_fp32_head':[], 'bf16_single_frame_head':[], 'fp16_backbone_fp32_head':[]}
    alternative_errors = []
    for row in rows:
        metadata = row['metadata']; public = row['public']
        observation = renderer.current_rgbd(metadata['map_name'], metadata['actual_camera_pose'], float(public['ground_level']))
        regions = select_regions(observation, landmarks[metadata['map_name']])
        if [r['object_id'] for r in regions] != [r['object_id'] for r in metadata['roi_objects']]:
            raise ValueError('current source/public ROI content differs from frozen cache')
        pixels = native.processor(images=[cv2.cvtColor(observation['bgr'], cv2.COLOR_BGR2RGB)], return_tensors='pt').to(native.device)
        # Autocast is disabled: all original backbone parameters remain FP32.
        with torch.autocast('cuda', enabled=False):
            patches = native.model.vision_model(**pixels).last_hidden_state[0]
            masks = torch.tensor(np.stack([r['mask'] for r in regions]), device=native.device)
            reference = F.normalize(masked_native_pool(native.model.vision_model.head, patches, masks).float(), dim=-1)
            tokens = native.processor(text=[metadata['instruction']], padding='max_length', truncation=True,
                max_length=native.model.config.text_config.max_position_embeddings, return_tensors='pt').to(native.device)
            text = native.model.get_text_features(**tokens)
            if not torch.is_tensor(text): text = text.pooler_output
            text = F.normalize(text.float(), dim=-1)[0]
        cached = public['roi_features'].float().to(native.device)
        cached_text = public['native_text'].float().to(native.device)
        if args.diagnose:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                mixed_patches = native.model.vision_model(**pixels).last_hidden_state[0]
                mixed = F.normalize(masked_native_pool(native.model.vision_model.head, mixed_patches, masks).float(), dim=-1)
            with torch.autocast('cuda', enabled=False):
                stable = F.normalize(masked_native_pool(native.model.vision_model.head, mixed_patches.float(), masks).float(), dim=-1)
            alternatives['bf16_backbone_fp32_head'].extend(F.cosine_similarity(stable,reference,dim=-1).cpu().tolist())
            alternatives['bf16_single_frame_head'].extend(F.cosine_similarity(mixed,reference,dim=-1).cpu().tolist())
            with torch.autocast('cuda', dtype=torch.float16):
                half_patches = native.model.vision_model(**pixels).last_hidden_state[0]
            with torch.autocast('cuda', enabled=False):
                precise = F.normalize(masked_native_pool(native.model.vision_model.head, half_patches.float(), masks).float(), dim=-1)
            if not torch.isfinite(precise).all(): raise ValueError('FP16 candidate nonfinite')
            alternatives['fp16_backbone_fp32_head'].extend(F.cosine_similarity(precise,reference,dim=-1).cpu().tolist())
            alternative_errors.extend((precise@cached_text-reference@text).abs().cpu().tolist())
        image_cosines.extend(F.cosine_similarity(cached, reference, dim=-1).cpu().tolist())
        text_cosines.append(float(F.cosine_similarity(cached_text[None], text[None])))
        errors.extend((cached@cached_text-reference@text).abs().cpu().tolist())
        counts.append(len(regions))
    manifest = json.loads((Path(args.cache)/'manifest.json').read_text())
    description = manifest.get('image_precision', manifest.get('protocol',{}).get('image_precision','BF16 inference and FP16 storage'))
    result = dict(actual_frames=16, actual_rois=sum(counts), cached_precision=description,
        reference_precision='FP32 encoder and native ROI attention head',
        minimum_image_cosine=min(image_cosines), minimum_text_cosine=min(text_cosines), maximum_score_error=max(errors),
        thresholds_predeclared=dict(minimum_cosine=.995, maximum_cosine_score_error=.01))
    result['pass'] = result['minimum_image_cosine'] >= .995 and result['minimum_text_cosine'] >= .995 and result['maximum_score_error'] <= .01
    if args.diagnose:
        result['diagnostic_alternatives'] = {name:dict(minimum_cosine=min(values), mean_cosine=float(np.mean(values)), below_995=sum(v<.995 for v in values)) for name,values in alternatives.items()}
        result['fp16_backbone_fp32_head_maximum_score_error'] = max(alternative_errors)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if not result['pass'] and not args.diagnose: raise ValueError('actual observed ROI precision smoke failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default=DEFAULT_DATA)
    parser.add_argument('--diagnose', action='store_true', help='record fixed numerical alternatives without accepting a failed cache')
    run(parser.parse_args())


if __name__=='__main__': main()
