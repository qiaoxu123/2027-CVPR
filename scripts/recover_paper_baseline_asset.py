"""Recover the public CityNav MGP checkpoint into a new isolated asset directory.

No inference, model selection, or historical checkpoint replacement. URL comes
from the frozen official CityNav README in reports/paper_experiments/sources.
"""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

from scripts.build_paper_experiments import HERE, OUT, sha

URL = 'https://www.dropbox.com/scl/fi/xoz7mzv4fddwktn961chg/mgp_mturk.pth?rlkey=r80r51gtu38mfx022md3gmq7m&dl=1'


def main():
    asset = HERE / 'paper_baseline_assets/mgp_mturk.pth'
    manifest = OUT / 'mgp_recovered_asset.json'
    if asset.exists():
        if not manifest.exists() or sha(asset) != json.loads(manifest.read_text())['sha256']:
            raise ValueError('Existing asset has no matching provenance; do not overwrite')
        print('Already verified; no download repeated')
        return
    asset.parent.mkdir(exist_ok=True)
    partial = asset.with_suffix('.pth.part')
    started = time.time()
    digest = hashlib.sha256()
    with urllib.request.urlopen(URL, timeout=60) as response, partial.open('xb') as dest:
        expected = int(response.headers['Content-Length'])
        assert 'text/html' not in response.headers.get('Content-Type', '')
        size = 0
        while True:
            block = response.read(4 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
            dest.write(block)
        assert size == expected == 228385010, (size, expected)
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        entries = archive.namelist()
        assert any(x.endswith('data.pkl') for x in entries)
    partial.rename(asset)
    record = dict(method='MGP', asset=str(asset), bytes=size, sha256=digest.hexdigest(),
                  url=URL, source_readme=str(OUT/'sources/citynav_readme.md'),
                  source_readme_sha256=sha(OUT/'sources/citynav_readme.md'),
                  zip_integrity_passed=True, started=started, finished=time.time(),
                  command='python -m scripts.recover_paper_baseline_asset',
                  status='asset_recovered_not_evaluated',
                  note='Public checkpoint recovered in isolated new directory; not assumed identical to missing historical checkpoint. No old output or weight overwritten.')
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
