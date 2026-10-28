"""Host-side, resumable transfer verification and original A2 unit handoff.

Run with the destination's system Python. All scientific code runs inside the
legacy-path adapter with the original isolated Python environment. This script
cannot shorten the fixed20+5 recipe or select any benchmark checkpoint.
"""
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess,time
from contextlib import contextmanager
from pathlib import Path,PurePosixPath

LEGACY=PurePosixPath('/home/20261024_1')
PROJECT=LEGACY/'Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBelief'
BASE='runs/a2_waldo_landmark_20261027/revision_v3'
MAIN='graphnav-a2-relation-v3-pipeline-20261028.service'
POST='graphnav-a2-v3-confirmation-20261028.service'
NAV='graphnav-a2-v3-navigation-20261028.service'
POLICY='experiments/a2_waldo_landmark_belief/gpu_execution_policy.json'
INPUT_MANIFESTS=('current_transfer_plan.json','reuse_manifest.json','exact_compute_files.json','approved_gpu_policy.json')


@contextmanager
def host_lock(root):
    # Lock preparation as well as handoff: the GPU lock alone is too late.
    with (Path(root)/'bootstrap.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def source_inventory(project):
    names=subprocess.check_output(['git','ls-files','-z'],cwd=project,text=True).split('\0')
    return {name:digest(Path(project)/name) for name in names if name and name!=POLICY
            and name.split('/')[0] not in ('reports','runs','docs')
            and Path(name).suffix in ('.py','.json','.toml','.yaml','.yml','.sh')}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**22),b''):h.update(block)
    return h.hexdigest()


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def mapped_path(name,root):
    path=PurePosixPath(name)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe legacy path')
    try:relative=path.relative_to(LEGACY)
    except ValueError:raise ValueError('Path is outside the declared legacy home')
    return Path(root)/'legacy_home'/str(relative)


def archive_ready(path,entry):
    path=Path(path)
    if not path.is_file() or path.stat().st_size<entry['bytes']:return False
    if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('Complete transfer has incorrect bytes: '+str(path))
    return True


def archive_untrained_proofs(root):
    root=Path(root);archive=root/'remote_migration/pre_remote_runtime_recertification';marker=archive/'complete.json'
    training=root/'training'
    if training.exists() and any(p.is_file() for p in training.rglob('*')):
        raise ValueError('Refuse to reset proofs of an existing formal training run')
    if marker.exists():
        plan=json.loads((archive/'plan.json').read_text())
        if any(digest(archive/name)!=expected for name,expected in plan.items()):raise ValueError('Preserved original proof changed')
        return archive
    archive.mkdir(parents=True,exist_ok=True);plan_path=archive/'plan.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())
    else:
        paths=[]
        for name in ('cpu_gates.json','posttraining_cpu_gates.json','navigation_cpu_gates.json','cuda_smoke.json','runtime_preflight.json'):
            if (root/name).is_file():paths.append(root/name)
        for name in ('pretraining','preflight'):
            if (root/name).exists():paths.extend(p for p in (root/name).rglob('*') if p.is_file())
        plan={str(p.relative_to(root)):digest(p) for p in paths};put(plan_path,plan)
    for name,expected in plan.items():
        source=root/name;target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=expected:raise ValueError('Archived proof was changed')
            if source.exists():raise ValueError('Both old and new proof exist before archival commit')
        else:
            if not source.is_file() or digest(source)!=expected:raise ValueError('Old proof changed during archival')
            source.rename(target)
    for name in ('pretraining','preflight'):
        path=root/name
        if path.exists():
            for directory in sorted((p for p in path.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):directory.rmdir()
            path.rmdir()
    put(marker,dict(status='zero_epoch_original_proofs_preserved',files=len(plan)))
    return archive


class Bootstrap:
    def __init__(self,root,launch_sha256=None):
        self.launch_sha256=launch_sha256
        self.root=Path(root).resolve();self.project=mapped_path(str(PROJECT),self.root);self.science=self.project/BASE
        self.logs=self.root/'bootstrap_logs';self.logs.mkdir(exist_ok=True)
        self.env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            A2_EXECUTION_HOST_ROOT=str(self.root),
            HF_HOME=str(LEGACY/'.cache/huggingface'),HF_HUB_CACHE=str(LEGACY/'.cache/huggingface/hub'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg',MPLCONFIGDIR=str(self.root/'matplotlib'))
        libraries=str(self.root/'tools/root/usr/lib/x86_64-linux-gnu')
        self.env['LD_LIBRARY_PATH']=libraries+(':'+self.env['LD_LIBRARY_PATH'] if self.env.get('LD_LIBRARY_PATH') else '')
        self.prefix=[str(self.root/'tools/root/usr/bin/proot'),'-b',str(self.root/'legacy_home')+':'+str(LEGACY),'-w',str(PROJECT),
            str(LEGACY/'miniconda3/envs/graphnav/bin/python'),'-u']

    def status(self,state,**extra):
        record=dict(state=state,pid=os.getpid(),time=time.time(),**extra);put(self.root/'bootstrap_status.json',record)
        try:print(json.dumps(record),flush=True)
        except BrokenPipeError:pass

    def verify_launch(self):
        manifest=self.root/'launch_manifest.json'
        if not self.launch_sha256 or digest(manifest)!=self.launch_sha256:raise ValueError('Launch manifest differs from the reviewed source anchor')
        bound=json.loads(manifest.read_text())
        if set(bound['input_sha256'])!=set(INPUT_MANIFESTS):raise ValueError('Launch lacks the complete input/policy manifest inventory')
        for command,key in ((['git','rev-parse','HEAD'],'code_commit'),(['git','branch','--show-current'],'branch')):
            if subprocess.check_output(command,cwd=self.project,text=True).strip()!=bound[key]:raise ValueError('Destination Git revision differs from reviewed launch')
        if source_inventory(self.project)!=bound['source_sha256']:raise ValueError('Executable/test inventory differs from reviewed launch')
        script=Path(__file__).resolve()
        try:name=str(script.relative_to(self.project))
        except ValueError:raise ValueError('Bootstrap must run from the reviewed checkout')
        if bound['source_sha256'].get(name)!=digest(script):raise ValueError('Running bootstrap was not bound before launch')
        for name,expected in bound['input_sha256'].items():
            if Path(name).name!=name or digest(self.root/name)!=expected:raise ValueError('Input/policy manifest differs from reviewed launch')
        if digest(self.project/POLICY)!=bound['input_sha256']['approved_gpu_policy.json']:raise ValueError('Installed resource policy differs from reviewed launch')

    def guest(self,args,label,*,cpu=True):
        self.status('running_check',phase=label);env=dict(self.env)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        with (self.logs/(label+'.log')).open('a') as stream:
            result=subprocess.run(self.prefix+list(args),cwd=self.project,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{label} failed, exit={result.returncode}; inspect its preserved log')

    def wait_extract(self):
        plan=json.loads((self.root/'current_transfer_plan.json').read_text())
        if Path(plan['remote_root']).resolve()!=self.root or plan['legacy_home']!=str(LEGACY):raise ValueError('Transfer plan belongs to another location')
        for name,entry in plan['archives'].items():
            if Path(entry['filename']).name!=entry['filename']:raise ValueError('Unsafe archive filename')
            marker=self.root/('bootstrap_extracted_'+name+'.json')
            prior={'environment_base':'environment_base_complete.json','private_bootstrap_v2':'private_bootstrap_complete.json'}.get(name)
            if prior and (self.root/prior).exists():
                saved=json.loads((self.root/prior).read_text())
                if saved.get('sha256',saved.get('archive_sha256'))==entry['sha256']:continue
            if marker.exists():
                if json.loads(marker.read_text())['sha256']!=entry['sha256']:raise ValueError('Extraction commit changed')
                continue
            path=self.root/'incoming'/entry['filename']
            while not archive_ready(path,entry):self.status('waiting_for_transfer',archive=name,received=path.stat().st_size if path.exists() else 0,total=entry['bytes']);time.sleep(60)
            command=['tar','--unlink-first','--no-same-owner','-xf',str(path),'-C',str(self.root/'legacy_home'),'-I',
                'zstd --long=29' if entry.get('long_window_log')==29 else 'zstd']
            self.status('extracting_verified_archive',archive=name);subprocess.run(command,check=True,env=self.env)
            put(marker,dict(sha256=entry['sha256'],status='verified_extracted'))

    def verify_inputs(self):
        manifest=json.loads((self.root/'reuse_manifest.json').read_text())
        records=manifest['datasets']+manifest['encoders'];self.status('checking_exact_data_and_models')
        for row in records:
            path=mapped_path(row['path'],self.root)
            if 'symlink' in row:
                if not path.is_symlink() or str(path.readlink())!=row['symlink']:raise ValueError('Required dataset link changed')
            elif not path.is_file() or digest(path)!=row['sha256']:raise ValueError('Required source data/model changed: '+str(path))
        frozen=manifest['frozen_dependencies']
        bound=dict(frozen['source_sha256']);bound[frozen['checkpoint']]=frozen['checkpoint_sha256'];bound[frozen['waldo']]=frozen['waldo_sha256']
        for name,expected in bound.items():
            if digest(mapped_path(name,self.root))!=expected:raise ValueError('Original frozen dependency changed')
        compute=json.loads((self.root/'exact_compute_files.json').read_text())['files'];site=mapped_path(str(LEGACY/'miniconda3/envs/graphnav/lib/python3.10/site-packages'),self.root)
        for row in compute:
            relative=PurePosixPath(row['component'])/row['filename']
            if row['component'] not in ('torch','triton','nvidia') or relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe compute manifest')
            path=site/str(relative)
            if not path.is_file() or digest(path)!=row['sha256']:raise ValueError('Exact compute reconstruction incomplete: '+str(relative))
        approved=self.root/'approved_gpu_policy.json';active=self.project/'experiments/a2_waldo_landmark_belief/gpu_execution_policy.json'
        if digest(approved)!=digest(active):raise ValueError('Reviewed actual host resource policy was not installed')

    def recertify(self):
        ready=self.root/'bootstrap_ready.json'
        if ready.exists():
            if json.loads(ready.read_text()).get('launch_sha256')!=self.launch_sha256:raise ValueError('Existing CPU certification belongs to another launch')
            self.guest(['-c',"from experiments.a2_waldo_landmark_belief.relation_v3_navigation_sources import require_cpu_gates;require_cpu_gates();from experiments.a2_waldo_landmark_belief.relation_v3_navigation_pipeline import require_cpu_gates as handoff;handoff(__import__('pathlib').Path('runs/a2_waldo_landmark_20261027/revision_v3'))"],'retain_current_CPU_proofs');return
        archive=archive_untrained_proofs(self.science)
        previous=json.loads((archive/'cpu_gates.json').read_text());expected=previous['initial_weights_sha256']
        old_audit=archive/'original_data_audit.json'
        if not old_audit.exists():shutil.copy2(self.science/'data_audit.json',old_audit)
        audit_marker=self.root/'bootstrap_audit_complete.json'
        if audit_marker.exists():
            if json.loads(audit_marker.read_text())['sha256']!=digest(self.science/'data_audit.json'):raise ValueError('New remote audit changed')
        else:
            self.guest(['-m','scripts.a2_relation_v3_data_audit'],'full_TRAIN_dev_audit')
            before=json.loads(old_audit.read_text());after=json.loads((self.science/'data_audit.json').read_text())
            for key in ('counts','roles_sha256','label_code_sha256'):
                if before[key]!=after[key]:raise ValueError('Remote data/label audit differs from preserved local audit')
            put(audit_marker,dict(sha256=digest(self.science/'data_audit.json')))
        self.guest(['-m','experiments.a2_waldo_landmark_belief.relation_v3_preparation','--cpu'],'original_full_CPU_gate')
        current=json.loads((self.science/'cpu_gates.json').read_text())
        if current['initial_weights_sha256']!=expected:raise ValueError('Seed0 initial weights differ from preserved original weights')
        self.guest(['-m','experiments.a2_waldo_landmark_belief.relation_v3_confirmation_pipeline','--cpu-gates'],'posttraining_CPU_gate')
        self.guest(['-m','experiments.a2_waldo_landmark_belief.relation_v3_navigation_protocol'],'fixed_pilot_metadata')
        self.guest(['-c',NAVIGATION_CERTIFICATION],'navigation_CPU_gate')
        self.guest(['-m','experiments.a2_waldo_landmark_belief.relation_v3_navigation_pipeline','--cpu-gates'],'complete_handoff_CPU_gate')
        put(ready,dict(status='remote_CPU_inputs_and_original_initial_weights_verified',initial_weights_sha256=expected,
            launch_sha256=self.launch_sha256,source_sha256=digest(__file__),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.project,text=True).strip(),time=time.time()))

    def service(self,unit):
        result=subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus','-p','Result','-p','ExecStart'],env=self.env,text=True,capture_output=True)
        service=dict(line.split('=',1) for line in result.stdout.strip().splitlines())
        if result.returncode and service.get('LoadState')!='not-found':raise RuntimeError('Cannot inspect user service: '+result.stderr)
        return service

    def start(self,unit,module):
        service=self.service(unit)
        if service['LoadState']=='loaded':
            if str(self.root) not in service['ExecStart'] or module not in service['ExecStart']:raise ValueError('Unit belongs to another execution')
            if service['ActiveState']=='failed' or service['Result']!='success' or int(service['ExecMainStatus'])!=0:raise ValueError('Existing failed unit requires diagnosis; not restarting blindly')
            if int(service['MainPID'])==0:
                required=self.science/{MAIN:'training/curve_manifest.json',POST:'post_pipeline_status.json',NAV:'navigation_pipeline_status.json'}[unit]
                if not required.exists():raise ValueError('Existing terminal unit lacks completion evidence; inspect before resuming')
            return
        command=['systemd-run','--user','--unit='+unit,'--property=RemainAfterExit=yes','--property=WorkingDirectory='+str(self.project)]
        command += ['--setenv='+k+'='+self.env[k] for k in ('LD_LIBRARY_PATH','XDG_RUNTIME_DIR','A2_EXECUTION_HOST_ROOT','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','MPLBACKEND','MPLCONFIGDIR')]
        subprocess.run(command+self.prefix+['-m','scripts.a2_guest_service','--unit',unit,'--module','experiments.a2_waldo_landmark_belief.'+module],check=True,env=self.env)

    def run(self):
        with host_lock(self.root):
            try:self._run()
            except Exception as error:self.status('failed',error=str(error));raise

    def _run(self):
        self.verify_launch();self.wait_extract();self.verify_launch();self.verify_inputs();self.verify_launch();self.recertify();self.verify_launch()
        target=self.root/'execution_target.json'
        if not target.exists() or json.loads(target.read_text()).get('target')!='remote':self.status('prepared_no_remote_execution_selected');return
        self.start(MAIN,'relation_v3_pipeline');self.start(POST,'relation_v3_confirmation_pipeline');self.start(NAV,'relation_v3_navigation_pipeline')
        while True:
            services={name:self.service(name) for name in (MAIN,POST,NAV)}
            for name,service in services.items():
                if service['ActiveState']=='failed' or service['Result']!='success' or int(service['ExecMainStatus'])!=0:raise RuntimeError('Scientific execution unit failed: '+name)
            if all(int(s['MainPID'])==0 and s['SubState'] in ('exited','dead') for s in services.values()):break
            self.status('scientific_pipeline_live',services={k:{n:v[n] for n in ('MainPID','ActiveState','SubState')} for k,v in services.items()});time.sleep(60)
        post=json.loads((self.science/'post_pipeline_status.json').read_text())
        self.guest(['-m','experiments.a2_waldo_landmark_belief.relation_v3_navigation_pipeline','--verify-complete'],'verify_terminal_navigation')
        self.verify_launch()
        navigation=json.loads((self.science/'navigation_pipeline_status.json').read_text())
        self.status('scientific_units_terminal',posttraining_state=post['state'],navigation_state=navigation['state'],full_benchmark_completed=navigation['full_benchmark_completed'])


NAVIGATION_CERTIFICATION=r'''
import re,subprocess,sys,os
from experiments.a2_waldo_landmark_belief.paths import ROOT,sha,save
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.relation_v3_navigation_sources import code_hashes,test_hashes,require_cpu_gates
root=BASE/'revision_v3';before=(code_hashes(),test_hashes());log=root/'navigation_cpu_tests.log'
with log.open('w') as stream:
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests','tests_confirmation_v3','tests_navigation_v3'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
match=re.search(r'\d+ passed[^\n]*',log.read_text())
if result.returncode or match is None or before!=(code_hashes(),test_hashes()):raise ValueError('Full navigation CPU gate failed or source changed')
proof=dict(status='v3_real_navigation_CPU_gates_passed',implementation=before[0],test_source_sha256=before[1],test_log_sha256=sha(log),unit_test_summary=match.group(),
    pilot_population_sha256=sha(root/'navigation_pilot_population.json'),protocol_sha256=sha(root/'navigation_protocol.json'),no_CUDA_used=True,real_pilot_completed=False,full_benchmark_completed=False)
save(proof,root/'navigation_cpu_gates.json');require_cpu_gates(root)
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--launch-sha256',required=True)
    args=parser.parse_args();Bootstrap(args.root,args.launch_sha256).run()


if __name__=='__main__':main()
