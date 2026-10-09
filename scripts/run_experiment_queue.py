"""Run a frozen local command manifest sequentially after required stage artifacts exist."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);args=parser.parse_args()
    path=Path(args.manifest);raw=path.read_bytes();manifest=json.loads(raw)
    fingerprint=hashlib.sha256(raw).hexdigest()
    status_path=path.with_suffix('.status.json')
    status=dict(manifest_sha256=fingerprint,status='waiting',jobs=[])
    def save():status_path.write_text(json.dumps(status,indent=2))
    save()
    while not all(Path(p).is_file() for p in manifest.get('requires',[])):
        time.sleep(15)
        if path.read_bytes()!=raw:raise ValueError('queued manifest changed')
    env=dict(os.environ,**manifest.get('environment',{}))
    status['status']='running';save()
    for job in manifest['jobs']:
        if Path(job['complete_artifact']).exists():
            raise ValueError('refusing to overwrite existing queued experiment: '+job['name'])
        log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
        record=dict(name=job['name'],command=job['command'],started=time.time(),status='running')
        status['jobs'].append(record);save()
        with log.open('w') as stream:
            code=subprocess.call(job['command'],env=env,stdout=stream,stderr=subprocess.STDOUT)
        record.update(seconds=time.time()-record['started'],returncode=code,status='complete' if code==0 else 'failed')
        if code or not Path(job['complete_artifact']).is_file():
            status['status']='failed';save();raise SystemExit(code or 1)
        save()
    status['status']='complete';save()


if __name__=='__main__':main()
