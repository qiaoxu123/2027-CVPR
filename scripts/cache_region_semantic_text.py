"""Fixed, predeclared SigLIP2 category text for a zero-shot diagnostic."""
import argparse
from pathlib import Path
import torch
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior text cache')
    torch.set_num_threads(4)
    captions = ['a building', 'a parking area', 'a ground area']
    encoder = ObservedSigLIPMatchingEncoder(args.backbone)
    atomic_save(encoder.texts(captions).cpu().half(), args.output)
    atomic_json(dict(captions=captions, backbone=args.backbone,
                     vectors_sha256=file_sha256(args.output), source_sha256=file_sha256(__file__)),
                args.output.with_suffix('.provenance.json'))


if __name__ == '__main__': main()
