"""Run a frozen local command manifest sequentially after required stage artifacts exist."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import shutil


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
        # Imports must remain unchanged even while development continues elsewhere.
        snapshot=Path('runs/source_snapshots')/(fingerprint[:12]+'_'+job['name'])
        if snapshot.exists():raise ValueError('refusing existing source snapshot')
        snapshot.mkdir(parents=True);source_hashes={}
        for folder in ('graphnav','scripts'):
            for source in Path(folder).rglob('*.py'):
                if '__pycache__' in source.parts:continue
                target=snapshot/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                source_hashes[str(source)]=hashlib.sha256(target.read_bytes()).hexdigest()
        source_info=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            diff=subprocess.check_output(['git','diff','HEAD','--binary'],text=True),files=source_hashes)
        (snapshot/'source_manifest.json').write_text(json.dumps(source_info,indent=2))
        command=list(job['command']);job_env=dict(env,PYTHONPATH=str(snapshot.resolve()))
        if len(command)>2 and command[1]=='-m':
            module=command[2]
            command=[command[0],'-c',"import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name='__main__')",str(snapshot.resolve()),module,*command[3:]]
        elif len(command)>1 and command[1].startswith('scripts/'):
            command[1]=str((snapshot/command[1]).resolve())
        record=dict(name=job['name'],command=job['command'],executed_command=command,
            source_snapshot=str(snapshot.resolve()),started=time.time(),status='running')
        status['jobs'].append(record);save()
        with log.open('w') as stream:
            code=subprocess.call(command,env=job_env,stdout=stream,stderr=subprocess.STDOUT)
        record.update(seconds=time.time()-record['started'],returncode=code,status='complete' if code==0 else 'failed')
        if code or not Path(job['complete_artifact']).is_file():
            status['status']='failed';save();raise SystemExit(code or 1)
        save()
    status['status']='complete';save()


if __name__=='__main__':main()
