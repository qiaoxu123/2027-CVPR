"""CPU gates and isolated real-TRAIN CUDA backward, before formal V3 training."""
import argparse,fcntl,json,os,re,subprocess,sys,time
import torch
from .paths import ROOT,A0_SHA,sha,save
from .collected_snapshot import BASE
from .relation_v3_train import code_hashes,validate_audit,initial_model,fingerprint,require_idle_gpu,optimizer_for
from .relation_v3_protocol import CONFIG
from .relation_v3_data import RelationV3Views
from .relation_v3_loss import relation_v3_loss
from .batching import collate_views,to_device
from .train import atomic_checkpoint


def test_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tests').rglob('*.py'))}


def cpu_gates():
    torch.set_num_threads(2);hashes=code_hashes();tests=test_hashes();audit=validate_audit()
    directory=BASE/'revision_v3/pretraining';directory.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');log=directory/'cpu_tests.log'
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode or not re.search(r'\d+ passed',log.read_text()):raise ValueError('Current V3 unit tests must pass')
    if hashes!=code_hashes() or tests!=test_hashes():raise ValueError('Code/tests changed during CPU gate')
    model=initial_model();weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    state=dict(scope='TRAIN_functional_only',epoch=0,checkpoint_eligible=False,config=CONFIG,code_sha256=hashes,
        roles_sha256=sha(BASE/'revision_v3/roles.json'),audit_sha256=audit,a0_sha256=A0_SHA,
        model_config=model.config,model=weights,weights_sha256=fingerprint(weights))
    path=directory/'initial.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=False)
        if any(old[k]!=v for k,v in state.items() if k!='model') or fingerprint(old['model'])!=state['weights_sha256']:
            raise ValueError('Existing pretraining snapshot changed; preserve it before a versioned rebuild')
    else:atomic_checkpoint(state,path)
    report=dict(status='v3_cpu_tests_and_initial_state_verified',code_sha256=hashes,roles_sha256=state['roles_sha256'],
        audit_sha256=audit,initial_checkpoint_sha256=sha(path),initial_weights_sha256=state['weights_sha256'],
        test_source_sha256=tests,test_log_sha256=sha(log),test_summary=re.search(r'\d+ passed[^\n]*',log.read_text()).group(),
        no_CUDA_used=True,formal_training=False)
    save(report,BASE/'revision_v3/cpu_gates.json');print(json.dumps(report),flush=True)


def cuda_smoke():
    require_idle_gpu();hashes=code_hashes();roles=sha(BASE/'revision_v3/roles.json')
    cpu=json.loads((BASE/'revision_v3/cpu_gates.json').read_text())
    if cpu['code_sha256']!=hashes or cpu['roles_sha256']!=roles or cpu['test_source_sha256']!=test_hashes():raise ValueError('Current CPU proof required')
    torch.set_num_threads(2);dataset=RelationV3Views('train');rows=[];counts={};sources=[]
    for i,entry in enumerate(dataset.entries):
        family=entry['source_kind']
        if counts.get(family,0)>=2:continue
        row=dataset[i];y=row['labels'];positive=y['localization_utility']&y['utility_valid'];negative=~positive&y['utility_valid']
        if positive.any() and negative.any():
            rows.append(row);counts[family]=counts.get(family,0)+1
            sources.append({k:entry[k] for k in ('source_kind','filename','sha256','episode','step')})
        if len(rows)==6:break
    if len(rows)!=6 or len(counts)!=3:raise ValueError('All three TRAIN families required for CUDA smoke')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=initial_model();require_idle_gpu();model=model.cuda().train();batch=to_device(collate_views(rows),'cuda');optimizer=optimizer_for(model)
    initial=fingerprint(model.state_dict());history=[];grads={}
    for step in range(6):
        torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        loss,stats=relation_v3_loss(model,model(**batch['network']),batch,refine=step>=3)
        if not torch.isfinite(loss):raise ValueError('Nonfinite real TRAIN CUDA loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),5.,error_if_nonfinite=True)
        if not norm>0:raise ValueError('No CUDA gradient')
        grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        if step>=3:
            for prefix in ('base.language.','base.scorer.rank_head.','base.match.','base.relation.','category.','risk.'):
                if not any(v>0 for n,v in grads.items() if n.startswith(prefix)):raise ValueError('Missing CUDA gradient '+prefix)
        optimizer.step();torch.cuda.synchronize()
        history.append(dict(step=step,refine=step>=3,loss=float(loss),seconds=time.perf_counter()-start,stats=stats))
    if fingerprint(model.state_dict())==initial or hashes!=code_hashes() or roles!=sha(BASE/'revision_v3/roles.json'):
        raise ValueError('CUDA smoke made no update or code/roles changed')
    report=dict(status='v3_real_TRAIN_CUDA_backward_passed',code_sha256=hashes,roles_sha256=roles,
        cpu_gates_sha256=sha(BASE/'revision_v3/cpu_gates.json'),initial_weights_sha256=initial,sources=sources,history=history,
        gradients=grads,device=torch.cuda.get_device_name(0),max_cuda_memory_gb=torch.cuda.max_memory_allocated()/2**30,
        smoke_steps=6,batch_views=6,formal_training=False,no_benchmark_used=True)
    save(report,BASE/'revision_v3/cuda_smoke.json');print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu',action='store_true');parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    if args.cpu==args.cuda:raise ValueError('Choose exactly one CPU or CUDA gate')
    if args.cpu:cpu_gates();return
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cuda_smoke()


if __name__=='__main__':main()
