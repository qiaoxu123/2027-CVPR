"""Bind the real guest PID and completed status to its systemd invocation.

Run inside the legacy path adapter. The original scientific module executes in
this same Python process, retaining all of its own locks and device checks.
"""
import argparse,hashlib,json,os,re,runpy,subprocess,time
from pathlib import Path

UNITS={
 'graphnav-a2-relation-v3-pipeline-20261028.service':('relation_v3_pipeline','pipeline_status.json'),
 'graphnav-a2-v3-confirmation-20261028.service':('relation_v3_confirmation_pipeline','post_pipeline_status.json'),
 'graphnav-a2-v3-navigation-20261028.service':('relation_v3_navigation_pipeline','navigation_pipeline_status.json')}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def current_service(unit):
    invocation=os.environ.get('INVOCATION_ID','')
    if re.fullmatch('[0-9a-f]{32}',invocation) is None:raise ValueError('A real systemd invocation is required')
    env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}')
    text=subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','InvocationID','-p','ExecStart'],env=env,text=True)
    properties=dict(line.split('=',1) for line in text.strip().splitlines())
    host_root=os.environ.get('A2_EXECUTION_HOST_ROOT')
    if (properties['InvocationID']!=invocation or int(properties['MainPID'])<=0 or not host_root or host_root not in properties['ExecStart']):raise ValueError('Guest does not match the active approved host invocation')
    return dict(unit=unit,invocation_id=invocation,host_main_pid=int(properties['MainPID']),guest_pid=os.getpid(),execution_host_root=host_root)


def require_identity(service,status,identity,expected):
    if (not identity or identity.get('terminal') is not True or status.get('phase')!='finished'
        or identity.get('guest_pid')!=status.get('pid') or identity.get('host_main_pid')!=int(service['ExecMainPID'])
        or int(service['ExecMainPID'])<=0 or identity.get('invocation_id')!=service.get('InvocationID')
        or re.fullmatch('[0-9a-f]{32}',identity.get('invocation_id','')) is None
        or any(identity.get(key)!=value for key,value in expected.items())):
        raise ValueError('Completed guest/status does not match the current reviewed service invocation')
    return identity


def run(unit,module):
    if unit not in UNITS or module!='experiments.a2_waldo_landmark_belief.'+UNITS[unit][0]:raise ValueError('Unsupported scientific service/module pair')
    project=Path(__file__).resolve().parents[1];root=project/'runs/a2_waldo_landmark_20261027/revision_v3'
    target=project/'experiments/a2_waldo_landmark_belief'/(UNITS[unit][0]+'.py')
    record=dict(current_service(unit),source_root=str(project),module=module,wrapper_sha256=digest(__file__),module_sha256=digest(target),terminal=False)
    path=root/'service_identity'/(unit+'.json')
    if path.exists():
        previous=path.parent/'archive'/(digest(path)+'.json');previous.parent.mkdir(exist_ok=True);previous.write_bytes(path.read_bytes())
    put(path,record)
    try:runpy.run_module(module,run_name='__main__')
    except SystemExit as error:
        if error.code not in (None,0):raise
    status_path=root/UNITS[unit][1];status=json.loads(status_path.read_text())
    if status.get('pid')!=os.getpid() or status.get('phase')!='finished':raise ValueError('Scientific module exited without its own finished status')
    if digest(target)!=record['module_sha256'] or digest(__file__)!=record['wrapper_sha256']:raise ValueError('Scientific service source changed during execution')
    put(path,dict(record,terminal=True,status_sha256=digest(status_path),finished_at=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--unit',required=True);parser.add_argument('--module',required=True);args=parser.parse_args()
    # The original modules receive their original zero-argument command line.
    import sys
    sys.argv=[args.module];run(args.unit,args.module)


if __name__=='__main__':main()
