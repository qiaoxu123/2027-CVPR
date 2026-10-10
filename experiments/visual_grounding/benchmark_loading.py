"""Measure real Seen PNG read/decode throughput separately from GPU kernels."""
import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from experiments.visual_grounding.freeze_dataset import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs/visual_grounding_benchmark')
    args = parser.parse_args()
    root = Path(args.root)
    rows = read(root / 'data_audit/frozen_seen/index.jsonl')
    measurements = {}
    for field in ['roi_path', 'roi_source']:
        paths = list(dict.fromkeys(row[field] for row in rows))[:1000]
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            for path in paths:
                if cv2.imread(path) is None:
                    raise FileNotFoundError(path)
            durations.append(time.perf_counter() - start)
        measurements[field] = dict(files=len(paths), seconds=durations,
            median_files_per_second=len(paths)/statistics.median(durations),
            mean_file_bytes=statistics.mean(Path(p).stat().st_size for p in paths),
            note='serial real PNG read+OpenCV decode; warm OS page cache; excludes model and resizing')
    (root / 'data_audit/io_benchmark.json').write_text(json.dumps(measurements, indent=2))
    print(json.dumps(measurements))


if __name__ == '__main__':
    main()
