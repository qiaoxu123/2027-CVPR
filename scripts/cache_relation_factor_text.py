"""Encode distinct public reference phrases with original frozen SigLIP2."""
import argparse
import json
from pathlib import Path
import time
import torch

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing vectors')
    torch.set_num_threads(4)
    roles = json.loads(args.roles.read_text())
    phrases = list(dict.fromkeys(ref['phrase'] for role in roles.values() for ref in role['references']))
    started = time.perf_counter()
    native = ObservedSigLIPMatchingEncoder(args.backbone)
    vectors = {}
    for start in range(0, len(phrases), 32):
        names = phrases[start:start+32]
        encoded = native.texts(names).cpu().half()
        for text, vector in zip(names, encoded):
            vectors[text] = vector.clone()
    task_vectors = {key: torch.stack([vectors[r['phrase']] for r in role['references']])
                    if role['references'] else torch.empty(0, native.dimension, dtype=torch.float16)
                    for key, role in roles.items()}
    atomic_save(task_vectors, args.output)
    atomic_json(dict(roles_sha256=file_sha256(args.roles), vectors_sha256=file_sha256(args.output),
        backbone=args.backbone, seconds=time.perf_counter()-started, unique_phrases=len(phrases),
        tasks=len(roles), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        private_labels_read=False), args.output.with_suffix('.provenance.json'))
    print(args.output.with_suffix('.provenance.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
