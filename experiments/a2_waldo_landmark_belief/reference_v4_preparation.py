"""Actual CPU gates and isolated six-step TRAIN CUDA backward for V4."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time

import torch

from .paths import ROOT,A0_SHA,sha,save,verify_frozen_assets
from .reference_overlay_context import BASE
from .reference_overlay_data import ReferenceOverlayViews
from .reference_v4_train import code_hashes,validate_complete_feature_cache
from .reference_v4_protocol import CONFIG
from .relation_v3_train import initial_model,fingerprint,require_idle_gpu,optimizer_for
from .relation_v3_loss import relation_v3_loss
from .batching import collate_views,to_device
from .train import atomic_checkpoint


DIRECTORY = BASE/'revision_v4'


def test_hashes():
    return {str(path.relative_to(ROOT)):sha(path) for path in sorted((ROOT/'tests').rglob('*.py'))}


def runtime_assets():
    from huggingface_hub.constants import HF_HUB_CACHE
    from pathlib import Path
    assets = verify_frozen_assets()
    revision = '1b426889ea62b5a72bf9839009a1b184bfc9c178'
    snapshot = Path(HF_HUB_CACHE)/'models--google--siglip2-large-patch16-384/snapshots'/revision
    if not (snapshot/'model.safetensors').is_file():
        raise ValueError('Exact frozen SigLIP2 snapshot required')
    assets['siglip2_revision'] = revision
    assets['siglip2_snapshot_sha256'] = {path.name:sha(path) for path in sorted(snapshot.iterdir()) if path.is_file()}
    return assets


def current_binding(cpu=None):
    feature = validate_complete_feature_cache()
    hashes = code_hashes()
    if cpu is None:
        cpu = json.loads((DIRECTORY/'cpu_gates.json').read_text())
    initial = DIRECTORY/'pretraining/initial.pt'
    binding = dict(feature,code_sha256=hashes,initial_checkpoint_sha256=sha(initial))
    if any(cpu.get(key) != value for key,value in binding.items()) or cpu.get('test_source_sha256') != test_hashes():
        raise ValueError('Current source/feature-bound V4 CPU proof required')
    if cpu.get('test_log_sha256') != sha(DIRECTORY/'pretraining/cpu_tests.log'):
        raise ValueError('Actual V4 CPU test log changed')
    assets_path = DIRECTORY/'pretraining/frozen_assets.json'
    if cpu.get('frozen_assets_sha256') != sha(assets_path) or json.loads(assets_path.read_text()) != runtime_assets():
        raise ValueError('Frozen V4 baseline/encoder assets changed')
    return binding


def cpu_gates():
    if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
        raise ValueError('Run V4 CPU gate with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(2)
    feature = validate_complete_feature_cache();hashes=code_hashes();tests=test_hashes()
    directory=DIRECTORY/'pretraining';directory.mkdir(parents=True,exist_ok=True)
    assets=runtime_assets();save(assets,directory/'frozen_assets.json')
    log=directory/'cpu_tests.log'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    summary=re.search(r'\d+ passed[^\n]*',log.read_text())
    if result.returncode or summary is None:
        raise ValueError('All current V4 CPU tests must pass; see cpu_tests.log')
    if hashes != code_hashes() or tests != test_hashes() or feature != validate_complete_feature_cache() or torch.cuda.is_initialized():
        raise ValueError('CPU gate sources changed or CUDA was opened')
    model=initial_model();weights={name:value.detach().clone() for name,value in model.state_dict().items()}
    state=dict(feature,scope='TRAIN_functional_only',epoch=0,checkpoint_eligible=False,config=CONFIG,
        code_sha256=hashes,a0_sha256=A0_SHA,model_config=model.config,model=weights,weights_sha256=fingerprint(weights))
    initial=directory/'initial.pt'
    if initial.exists():
        old=torch.load(initial,map_location='cpu',weights_only=False)
        if any(old.get(key) != value for key,value in state.items() if key!='model') or fingerprint(old['model']) != state['weights_sha256']:
            raise ValueError('Existing V4 initial snapshot differs; preserve before rebuilding')
    else:
        atomic_checkpoint(state,initial)
    report=dict(feature,status='v4_cpu_tests_and_initial_state_verified',code_sha256=hashes,
        initial_checkpoint_sha256=sha(initial),initial_weights_sha256=state['weights_sha256'],
        test_source_sha256=tests,test_log_sha256=sha(log),test_summary=summary.group(),all_cpu_tests_passed=True,
        frozen_assets_sha256=sha(directory/'frozen_assets.json'),no_gpu_used=True,formal_training=False)
    save(report,DIRECTORY/'cpu_gates.json');print(json.dumps(report),flush=True)


def cuda_smoke():
    require_idle_gpu();binding=current_binding()
    cpu=json.loads((DIRECTORY/'cpu_gates.json').read_text())
    torch.set_num_threads(2);dataset=ReferenceOverlayViews('train')
    rows,sources,counts=[],[],{}
    for index,entry in enumerate(dataset.entries):
        family=entry['source_kind']
        if counts.get(family,0)>=2: continue
        row=dataset[index];labels=row['labels']
        positive=labels['localization_utility']&labels['utility_valid']
        negative=~positive&labels['utility_valid']
        if positive.any() and negative.any():
            rows.append(row);counts[family]=counts.get(family,0)+1
            sources.append({key:entry[key] for key in ('source_kind','filename','sha256','overlay_sha256','episode','step')})
        if len(rows)==6: break
    if len(rows)!=6 or len(counts)!=3:
        raise ValueError('Three TRAIN source families with real positive/negative pairs required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=initial_model();initial=fingerprint(model.state_dict())
    if initial != cpu['initial_weights_sha256']:
        raise ValueError('V4 CUDA model differs from CPU initial weights')
    frozen={name:param.detach().cpu().clone() for name,param in model.named_parameters() if not param.requires_grad}
    require_idle_gpu();model=model.cuda().train();batch=to_device(collate_views(rows),'cuda');optimizer=optimizer_for(model)
    history=[];grads={}
    for step in range(6):
        require_idle_gpu(allow_self=True)
        torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        loss,stats=relation_v3_loss(model,model(**batch['network']),batch,refine=step>=3)
        if not torch.isfinite(loss): raise ValueError('Nonfinite actual TRAIN V4 CUDA loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),5.,error_if_nonfinite=True)
        if not norm>0: raise ValueError('No actual V4 CUDA gradient')
        grads={name:float(param.grad.norm()) for name,param in model.named_parameters() if param.grad is not None}
        if step>=3:
            for prefix in ('base.language.','base.scorer.rank_head.','base.match.','base.relation.','category.','risk.'):
                if not any(value>0 for name,value in grads.items() if name.startswith(prefix)):
                    raise ValueError('Missing V4 CUDA gradient '+prefix)
        optimizer.step();torch.cuda.synchronize()
        history.append(dict(step=step,refine=step>=3,loss=float(loss),seconds=time.perf_counter()-started,stats=stats))
    unchanged=all(torch.equal(dict(model.named_parameters())[name].detach().cpu(),value) for name,value in frozen.items())
    if not unchanged or fingerprint(model.state_dict())==initial or current_binding()!=binding:
        raise ValueError('V4 CUDA update/frozen parameters/source check failed')
    report=dict(binding,status='v4_real_TRAIN_CUDA_backward_passed',cpu_gates_sha256=sha(DIRECTORY/'cpu_gates.json'),
        initial_weights_sha256=initial,rank_backward_steps=3,refine_backward_steps=3,all_finite=True,
        frozen_parameters_unchanged=True,sources=sources,history=history,gradients=grads,
        device=torch.cuda.get_device_name(0),max_cuda_memory_gb=torch.cuda.max_memory_allocated()/2**30,
        formal_training=False,benchmark_used=False)
    save(report,DIRECTORY/'cuda_smoke.json');print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu',action='store_true');parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    if args.cpu==args.cuda: raise ValueError('Choose exactly one V4 CPU or CUDA gate')
    if args.cpu: cpu_gates();return
    with (BASE/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cuda_smoke()


if __name__=='__main__': main()
