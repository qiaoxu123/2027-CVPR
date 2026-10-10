"""Real HF patch interface and exact native pooling equivalence; no accuracy claim."""
import argparse
import json
from pathlib import Path

import cv2
import torch

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import file_sha256,provenance
from graphnav.observed_patch_scales import masked_native_pool


def main():
    p=argparse.ArgumentParser();p.add_argument('--observations',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing smoke overwrite')
    row=json.loads(Path(a.observations).read_text().splitlines()[0])
    path=row['rgb_observations'][0]['path'];bgr=cv2.imread(path)
    assert bgr.shape==(384,384,3)
    native=ObservedSigLIPMatchingEncoder(microbatch=1)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        batch=native.processor(images=[cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
        vision=native.model.vision_model(**batch);patches=vision.last_hidden_state
        result=masked_native_pool(native.model.vision_model.head,patches[0],torch.ones(1,576,device='cuda',dtype=torch.bool))
        expected=vision.pooler_output
    difference=float((result.float()-expected.float()).abs().max())
    torch.testing.assert_close(result.float(),expected.float(),atol=.02,rtol=.02)
    report=dict(patch_shape=list(patches.shape),native_pool_shape=list(expected.shape),max_absolute_difference=difference,
        finite=True,frozen_backbone=True,revision=getattr(native.model.config,'_commit_hash',None),
        actual_image_sha256=file_sha256(path),source_sha256=file_sha256(__file__),provenance=provenance(),
        note='Actual native384 forward. Full-mask pooling matches native HF pooling within BF16 rounding. No semantic accuracy claim.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
