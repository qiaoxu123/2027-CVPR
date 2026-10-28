"""Explicit GPU occupancy policy; default exclusive, never stop other jobs.

Only an operator-installed, source-fingerprinted policy may exempt existing
user-approved background services. UID, full NUL-separated command digest and
executable argument must all match, with a strict instance cap. Another A2 or
an unreadable/changed process remains blocking. This changes resource admission,
not model/data/controller/training configuration.
"""
import hashlib,json,os,re,subprocess
from pathlib import Path

POLICY=Path(__file__).with_suffix('.json')


def validate_policy(policy):
    mode=policy.get('mode');services=policy.get('services')
    if mode not in ('exclusive','approved_background_services') or not isinstance(services,list):raise ValueError('Invalid GPU execution policy')
    if mode=='exclusive':
        if services:raise ValueError('Exclusive GPU policy cannot authorize services')
        return policy
    if not isinstance(policy.get('approval_reference'),str) or not policy['approval_reference'].strip() or not 1<=len(services)<=8:
        raise ValueError('Explicit user-approved service policy required')
    identities=set()
    for service in services:
        identity=tuple(service.get(k) for k in ('uid','process_name','cmdline_sha256'))
        if (type(identity[0]) is not int or identity[0]<0 or not isinstance(identity[1],str) or not identity[1].startswith('/')
            or not isinstance(identity[2],str) or re.fullmatch('[0-9a-f]{64}',identity[2]) is None
            or type(service.get('maximum_instances')) is not int or not 1<=service['maximum_instances']<=2
            or identity in identities):raise ValueError('Invalid or duplicate approved service identity')
        identities.add(identity)
    return policy


def process_identity(pid):
    try:
        proc=Path('/proc')/str(pid);command=(proc/'cmdline').read_bytes()
        if not command:return None
        return dict(uid=proc.stat().st_uid,process_name=os.fsdecode(command.split(b'\0',1)[0]),cmdline_sha256=hashlib.sha256(command).hexdigest())
    except (OSError,ValueError):return None


def classify(pids,policy,reader=process_identity,*,self_pid,allow_self=False):
    validate_policy(policy);services=policy['services'];counts=[0]*len(services);blocked=[]
    for pid in pids:
        if allow_self and pid==self_pid:continue
        if policy['mode']=='exclusive':blocked.append(pid);continue
        identity=reader(pid);matched=False
        if identity is not None:
            for i,service in enumerate(services):
                if all(identity.get(k)==service[k] for k in ('uid','process_name','cmdline_sha256')) and counts[i]<service['maximum_instances']:
                    counts[i]+=1;matched=True;break
        if not matched:blocked.append(pid)
    return blocked


def blocking_gpu_pids(*,allow_self=False):
    policy=validate_policy(json.loads(POLICY.read_text()))
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    pids=[int(line.strip()) for line in text.splitlines() if line.strip()]
    return classify(pids,policy,self_pid=os.getpid(),allow_self=allow_self)
