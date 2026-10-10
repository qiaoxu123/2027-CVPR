"""Auditable continuation, frozen Seen diagnostics and unchanged-controller rollouts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_static_controlled_queue import BASE, ROOT, OLD, FULL
from graphnav.checkpointing import file_sha256


def main():
    out = ROOT / 'runs/static_final_analysis_20261010'
    out.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    status = {'jobs': {}, 'selection': 'Seen R@1 then NE; no Unseen tuning'}
    protected = [FULL/'runs/final_a0_a5_evaluation/geometry_best.pt', OLD/'runs/a3_trajectory_stop_seed0/best.pt',
                 OLD/'runs/a3_replicate_seed0/best.pt', OLD/'runs/a3_replicate_seed1/best.pt']
    status['protected_sha256'] = {str(p):file_sha256(p) for p in protected}

    def save():
        (out/'status.json').write_text(json.dumps(status, indent=2))

    def run(name, cmd):
        status['jobs'][name] = dict(command=cmd, started=time.time(), status='running')
        save()
        with (out/(name+'.log')).open('w') as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status['jobs'][name].update(exit_code=code, finished=time.time(), status='passed' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(name+' failed; stop dependent evaluation')

    save()
    queue = ROOT/'runs/static_completed_20261010/status.json'
    while not json.loads(queue.read_text()).get('completed'):
        if any(v['status']=='failed' for v in json.loads(queue.read_text())['jobs'].values()):
            raise RuntimeError('upstream training failed')
        time.sleep(5)
    # This recipe was added after the interrupted original queue was started.
    flags = ['--variant','gat','--graph-control','reliable-relation']
    for smoke in (True,False):
        name='A4_reliable_relation'+('_smoke' if smoke else '')
        cmd=BASE+flags+['--epochs','1' if smoke else '5','--output',str(out/name)]
        if smoke:cmd+=['--max-batches','1']
        run(name,cmd)
    # Seed1 was the strongest of the original three Seed runs. Keep Seed0 E1 too.
    base1=list(BASE)
    base1[base1.index('--seed')+1]='1'
    base1[base1.index('--lr')+1]='0.00005'
    run('E1_seed1_smoke',base1+['--epochs','6','--max-batches','1','--continue-from',str(protected[-1]),'--output',str(out/'E1_seed1_smoke')])
    run('E1_seed1',base1+['--epochs','15','--continue-from',str(protected[-1]),'--output',str(out/'E1_seed1')])
    models={'A3_seed0':protected[2], 'A3_seed1':protected[3],
            'E1_seed0':ROOT/'runs/static_E1_seed0_lr5e5_20261010/best.pt',
            'E1_seed1':out/'E1_seed1/best.pt'}
    for name in ('A4_full','A4_self','A4_zero_geometry','A4_relation'):
        models[name]=ROOT/'runs/static_E2_E3_20261010'/name/'best.pt'
    for name in ('A3_relation000','A3_relation001','A3_relation005'):
        models[name]=ROOT/'runs/static_completed_20261010'/name/'best.pt'
    models['A4_reliable_relation']=out/'A4_reliable_relation/best.pt'
    entries=[name+'='+str(path) for name,path in models.items()]
    records=FULL/'data/revised_static/graphnav_val_seen.jsonl'
    cache=OLD/'runs/a3_dynamic_seen_fp32/bert_tokens.pt'
    run('static_seen',[sys.executable,'-m','scripts.evaluate_controlled_seen','--records',str(records),
                      '--bert-cache',str(cache),'--checkpoints',*entries,'--output',str(out/'static_seen')])
    cf=ROOT/'runs/static_E1_counterfactuals_20261010'
    run('direction_factorial',[sys.executable,'-m','scripts.evaluate_graph_language_factorial',
        '--original',str(cf/'synthetic_original.jsonl'),'--swapped',str(cf/'synthetic_swapped.jsonl'),
        '--checkpoints',*entries,'--output',str(out/'direction_factorial.json')])
    run('relation_predicates',[sys.executable,'-m','scripts.evaluate_static_relation_predicates',
        '--records',str(records),'--bert-cache',str(cache),'--checkpoints',*entries,
        '--output',str(out/'relation_predicates.json')])
    protocol=json.loads((ROOT/'reports/a3_relation_gat_20261010/E1_best_seen128.json').read_text())
    for name,path in models.items():
        if name in ('A3_seed0','E1_seed0'):
            continue  # Their preserved complete Seen128 artifacts already exist.
        args=dict(protocol['args'],checkpoint=str(path),output=str(out/'seen128'/name),evaluation_concurrency=1)
        args={k:v for k,v in args.items() if v is not None and v is not False}
        cmd=[sys.executable,'-m','scripts.evaluate_closedloop']
        for key,value in args.items():
            cmd+=['--'+key.replace('_','-')]
            if value is not True:cmd+=[str(value)]
        run('seen128_'+name,cmd)
        metrics=json.loads((Path(args['output'])/'metrics.json').read_text())
        if metrics['episode_manifest_sha256']!=protocol['episode_manifest_sha256']:
            raise ValueError('Seen128 episode manifest mismatch')
    for path,sha in status['protected_sha256'].items():
        if file_sha256(path)!=sha:raise ValueError('protected source changed')
    status['completed']=time.time()
    save()


if __name__=='__main__':
    main()
