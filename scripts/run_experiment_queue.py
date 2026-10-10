"""Run frozen jobs; independent evaluations may overlap, training is serialized."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import shutil
import fcntl


def validate_required_checks(manifest):
    for check in manifest.get('required_checks',[]):
        state=json.loads(Path(check['path']).read_text())
        for key,expected in check['equals'].items():
            if state.get(key)!=expected:
                raise ValueError('required stage check failed: '+check['path']+' / '+key)


def is_evaluation(job):
    command=job['command']
    return job.get('kind')=='evaluation' or (len(command)>1 and command[1] in
        ('scripts/evaluate_closedloop.py','scripts/evaluate_frozen_static.py'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--evaluation-workers',type=int,choices=(1,2))
    parser.add_argument('--resume-waiting',action='store_true',help='Only restart an empty waiting queue; no running experiment resume')
    args=parser.parse_args()
    path=Path(args.manifest);raw=path.read_bytes();manifest=json.loads(raw)
    fingerprint=hashlib.sha256(raw).hexdigest()
    status_path=path.with_suffix('.status.json')
    lock=path.with_suffix('.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=None
    if status_path.exists():
        prior=json.loads(status_path.read_text())
        if not args.resume_waiting or prior['status']!='waiting' or prior['jobs'] or prior['manifest_sha256']!=fingerprint:
            raise ValueError('refusing existing nonempty/nonwaiting queue; preserve completed/failed jobs')
    workers=args.evaluation_workers or manifest.get('evaluation_workers',1)
    if workers not in (1,2):raise ValueError('at most two independent evaluations')
    names=[job['name'] for job in manifest['jobs']]
    outputs=[str(Path(job['complete_artifact']).resolve()) for job in manifest['jobs']]
    logs=[str(Path(job['log']).resolve()) for job in manifest['jobs']]
    if len(set(names))!=len(names) or len(set(outputs))!=len(outputs) or len(set(logs))!=len(logs):
        raise ValueError('parallel jobs require unique names, outputs and logs')
    status=dict(manifest_sha256=fingerprint,status='waiting',jobs=[],evaluation_workers=workers,
        scheduler_pid=os.getpid(),resumed_empty_waiting_status=prior,
        timing_note='Evaluation latency under overlapping GPU processes is not isolated benchmarking; actual overlap is given by job windows. Training never overlaps another job in this queue.')
    def save():
        temporary=status_path.with_suffix('.json.tmp');temporary.write_text(json.dumps(status,indent=2));os.replace(temporary,status_path)
    save()
    while not all(Path(p).is_file() for p in manifest.get('requires',[])):
        time.sleep(15)
        if path.read_bytes()!=raw:raise ValueError('queued manifest changed')
    env=dict(os.environ,**manifest.get('environment',{}))
    try:validate_required_checks(manifest)
    except Exception as error:
        status.update(status='failed_prerequisite_check',reason=str(error));save();raise
    status['status']='running';save()
    def launch(job):
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
        if len(command)>1 and command[1]=='scripts/evaluate_closedloop.py' and '--evaluation-concurrency' not in command:
            command+=['--evaluation-concurrency',str(workers)]
        if len(command)>2 and command[1]=='-m':
            module=command[2]
            command=[command[0],'-c',"import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name='__main__')",str(snapshot.resolve()),module,*command[3:]]
        elif len(command)>1 and command[1].startswith('scripts/'):
            command[1]=str((snapshot/command[1]).resolve())
        record=dict(name=job['name'],command=job['command'],executed_command=command,
            source_snapshot=str(snapshot.resolve()),started=time.time(),status='running',
            kind='evaluation' if is_evaluation(job) else 'serialized',concurrency_limit=workers if is_evaluation(job) else 1)
        with log.open('w') as stream:
            process=subprocess.Popen(command,env=job_env,stdout=stream,stderr=subprocess.STDOUT)
        record['pid']=process.pid;status['jobs'].append(record);save()
        return process,record,job
    pending=list(manifest['jobs']);active=[];failed=False
    while pending or active:
        for entry in active[:]:
            process,record,job=entry;code=process.poll()
            if code is None:continue
            valid=code==0 and Path(job['complete_artifact']).is_file()
            record.update(seconds=time.time()-record['started'],returncode=code,status='complete' if valid else 'failed')
            failed=failed or not valid;active.remove(entry);save()
        if failed:
            pending=[] # Drain already-started independent jobs; don't erase their results.
        while pending and not failed:
            evaluation=is_evaluation(pending[0])
            if active and (not evaluation or any(not is_evaluation(e[2]) for e in active) or len(active)>=workers):break
            job=pending.pop(0)
            try:active.append(launch(job))
            except Exception as error:
                failed=True;pending=[];status['reason']=str(error);save();break
            if not evaluation:break
        if active:time.sleep(.2)
    status['status']='failed' if failed else 'complete';save()
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
