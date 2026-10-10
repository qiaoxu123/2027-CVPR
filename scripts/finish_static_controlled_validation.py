"""Await predeclared static runs, then frozen-checkpoint Seen diagnostics only."""
import json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
W=ROOT.parent;OLD=W/'2027-CVPR-closedloop-sbf-parity';FULL=W/'2027-CVPR-full-token-rgb'
Q=ROOT/'runs/static_E2_E3_20261010';OUT=ROOT/'runs/static_post_validation_v2_20261010'
ENV=dict(os.environ,PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')


def main():
    OUT.mkdir(exist_ok=False,parents=True)
    def state(status,**extra):(OUT/'status.json').write_text(json.dumps(dict(status=status,time=time.time(),**extra),indent=2))
    state('waiting for static training; no Unseen')
    while True:
        status=json.loads((Q/'status.json').read_text())
        if any(x['status']=='failed' for x in status['jobs'].values()):raise RuntimeError('static training failed; refusing downstream')
        if status.get('completed') and (ROOT/'runs/static_E2_zero_forward_v2_20261010/report.json').exists():break
        time.sleep(5)
    recipes={name:Q/name/'best.pt' for name in ('A4_full','A4_self','A4_zero_geometry','A4_relation','A3_relation001','A3_relation005')}
    recipes['A3_relation000']=ROOT/'runs/static_E2_zero_forward_v2_20261010/best.pt'
    state('all training complete; Seen128 controller validation',checkpoints={k:str(v) for k,v in recipes.items()})
    source=W/'SBFNav-CityNav-Reproduction/runs/sbfnav_main_e20_bs16_s0_20260929/source'
    base=[sys.executable,'scripts/evaluate_closedloop.py','--sbf-source',str(source),'--config',str(source/'configs/sbfnav/main.yaml'),'--data-root',str(W/'DATA/sbfnav-data'),'--split','val_seen','--model','static-transfer','--head-checkpoint',str(OLD/'runs/a3_trajectory_stop_seed0/best.pt'),'--reference-a3-checkpoint',str(FULL/'runs/final_a0_a5_evaluation/geometry_best.pt'),'--head-execution','guarded-blend','--stop-threshold','.3','--max-episodes','128','--evaluation-concurrency','2']
    # Fully report failed recipes too; these Seen128 tests are diagnostics, not
    # an Unseen-based search. Original controller parameters always unchanged.
    def navigation(item):
        name,checkpoint=item
        with (OUT/(name+'_seen128.log')).open('w') as stream:
            code=subprocess.call(base+['--checkpoint',str(checkpoint),'--output',str(OUT/(name+'_seen128'))],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('navigation failed: '+name)
        return name,code
    with ThreadPoolExecutor(2) as pool:results=dict(pool.map(navigation,recipes.items()))
    state('Seen counterfactual evaluation',navigation_status=results)
    command=[sys.executable,'scripts/evaluate_p1_counterfactuals.py','--synthetic',str(OLD/'data/p1_pairs_v2/train.jsonl'),'--real-records',str(FULL/'data/revised_static/graphnav_val_seen.jsonl'),'--max-synthetic','512','--checkpoints']+[k+'='+str(v) for k,v in recipes.items()]+['--output',str(OUT/'counterfactuals')]
    with (OUT/'counterfactuals.log').open('w') as stream:
        code=subprocess.call(command,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
    if code:raise RuntimeError('counterfactual evaluation failed')
    state('completed',navigation_status=results)

if __name__=='__main__':main()
