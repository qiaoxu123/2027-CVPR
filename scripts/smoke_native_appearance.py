"""Real HF image/text interface and lightweight backward; not an accuracy test."""
import argparse
import json
from pathlib import Path
import time
import cv2
import torch
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder, MultiScaleAppearanceMatcher
from graphnav.checkpointing import file_sha256, provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backbone',default='google/siglip2-large-patch16-384')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing smoke overwrite')
    rows=[json.loads(s) for s in Path(a.observations).read_text().splitlines()[:2]]
    paths=[r['rgb_observations'][0]['path'] for r in rows]
    views=[cv2.imread(p) for p in paths]
    if any(v is None or v.shape!=(384,384,3) for v in views):raise ValueError('real native384 images required')
    started=time.perf_counter();encoder=ObservedSigLIPMatchingEncoder(a.backbone, microbatch=2)
    image=encoder.images(views);text=encoder.texts(['a house with a brown roof','a house with a gray roof'])
    again=encoder.images(views)
    assert torch.equal(image,again) and encoder.encoded_images==2
    # Clone outside inference mode: real frozen features can train the head.
    model=MultiScaleAppearanceMatcher(dimension=image.shape[-1]).cuda().train()
    image=image.detach().clone()[None,:,None];text=text[:1].detach().clone()
    scores,known,_=model(text,image,torch.ones(1,2,1,dtype=torch.bool,device='cuda'),
                         torch.zeros(1,2,1,3,device='cuda'))
    # Explicit interface-only probe label; discarded head, no semantic claim.
    loss=torch.nn.functional.cross_entropy(scores,torch.tensor([0],device='cuda'))
    loss.backward()
    assert torch.isfinite(loss) and model.image_residual[-1].weight.grad.abs().sum()>0
    assert not any(p.grad is not None for p in encoder.model.parameters())
    report=dict(provenance=provenance(),backbone=a.backbone,
                source_script_sha256=file_sha256(__file__),
                revision=getattr(encoder.model.config,'_commit_hash',None),
                native_text_max_length=encoder.model.config.text_config.max_position_embeddings,
                observations_sha256=file_sha256(a.observations),images=[dict(path=p,sha256=file_sha256(p)) for p in paths],
                image_embedding_shape=list(image.shape),native_text_shape=list(text.shape),
                actual_native_cosine_scores=(text@image[0,:,0].T).tolist(),
                finite_backward=True,frozen_backbone_no_grad=True,
                image_head_gradient_norm=float(model.image_residual[-1].weight.grad.norm()),
                seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                encoded_images=encoder.encoded_images,cache_hits=encoder.cache_hits,
                warning='Interface probe only. Arbitrary loss label0; no trained checkpoint saved, no semantic accuracy or navigation result. Actual pixels read only from already observed384 frames.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
