"""Package only new large assets for remote backup; never uploads or overwrites.

Usage: python -m scripts.package_paper_experiment_assets --output /tmp/unique-dir
Restore at Workspace with tar --zstd -xf the archive, then verify each file SHA.
The original full/residual weights are already in graphnav-results-20261016.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.build_paper_experiments import HERE, WORKSPACE, OUT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = []
    for control in ('real', 'no-belief', 'zero', 'shuffle'):
        folder = HERE / f'visual_evaluations/calibrated_residual_full_unseen_{control}'
        for name in ('predictions.jsonl', 'metrics.json', 'episodes.json', 'eval_started.json'):
            path = folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
            assets.append(path)
    assets.append(HERE / 'paper_baseline_assets/mgp_mturk.pth')
    manifest = dict(restore_root='Workspace', scope='Four complete calibrated residual Unseen controls and newly recovered public MGP checkpoint',
                    source_commit=subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip(),
                    prior_assets_release='https://github.com/qiaoxu123/2027-CVPR/releases/tag/graphnav-results-20261016',
                    excluded='Shared DATA, existing checkpoints and old evaluation assets are not recopied.',
                    files=[dict(path=str(p.relative_to(WORKSPACE)), bytes=p.stat().st_size, sha256=sha(p)) for p in assets])
    manifest['file_count'] = len(assets)
    manifest['total_bytes'] = sum(p.stat().st_size for p in assets)
    mp = args.output / 'POLYBELIEF_ASSET_MANIFEST.json'
    mp.write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / mp.name).write_bytes(mp.read_bytes())
    filelist = args.output / 'filelist.txt'
    filelist.write_text(''.join(str(p.relative_to(WORKSPACE)) + '\n' for p in assets))
    archive = args.output / 'polybelief-paper-experiments-20261017.tar.zst'
    if archive.exists():
        raise FileExistsError(archive)
    subprocess.run(['tar', '--zstd', '-cf', str(archive), '-C', str(WORKSPACE), '-T', str(filelist)], check=True)
    checksums = ''.join(f'{sha(p)}  {p.name}\n' for p in (archive, mp))
    (args.output / 'POLYBELIEF_SHA256SUMS').write_text(checksums)
    (OUT / 'POLYBELIEF_SHA256SUMS').write_text(checksums)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), files=len(assets))))


if __name__ == '__main__':
    main()
