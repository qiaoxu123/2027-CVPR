"""Public-feature variability audit on TRAIN fit only; no labels are read."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import file_sha256,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=json.loads((a.cache/'manifest.json').read_text());values=[];torch.set_num_threads(4)
    for chunk in m['chunks']:
        path=a.cache/chunk['file']
        if file_sha256(path)!=chunk['sha256']:raise ValueError('feature checksum changed')
        for r in torch.load(path,map_location='cpu',weights_only=True)['records']:
            if r['subset']!='fit':continue
            v=F.normalize(r['image_features'][r['local_valid'],0].float(),dim=-1);n=len(v)
            if n<2:continue
            pair=(v.sum(0).square().sum()-n)/(n*(n-1))
            text=F.normalize(r['target_text'].float(),dim=-1)
            values.append((float(pair),float((v@text).std())))
    atomic_json(dict(fit_frames=len(values),median_within_frame_pair_cosine=float(np.median([v[0] for v in values])),
        median_target_cosine_spatial_std=float(np.median([v[1] for v in values])),
        feature_manifest_sha256=file_sha256(a.cache/'manifest.json'),
        warning='Variability does not prove target recognition. No target labels or instance GT used.'),a.output)


if __name__=='__main__':main()
