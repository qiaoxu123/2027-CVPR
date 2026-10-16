"""Predeclared E2/E3 queue: smoke every recipe before full training, Seen only."""
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=next(p for p in ROOT.parents if p.name=='2027-CVPR')
OUT=ROOT/'runs/static_E2_E3_20261010'
OLD=PROJECT/'Ablations/GraphNav/2027-CVPR-closedloop-sbf-parity'
FULL=PROJECT/'Ablations/GraphNav/2027-CVPR-full-token-rgb'
BASE=[sys.executable,str(ROOT/'scripts/train_static_extension.py'),
 '--template',str(FULL/'runs/final_a0_a5_evaluation/geometry_best.pt'),
 '--train',str(FULL/'data/revised_static/graphnav_train_seen.jsonl'),
 '--seen',str(FULL/'data/revised_static/graphnav_val_seen.jsonl'),
 '--train-bert-cache',str(OLD/'runs/a3_dynamic_train_fp32/bert_tokens.pt'),
 '--seen-bert-cache',str(OLD/'runs/a3_dynamic_seen_fp32/bert_tokens.pt'),
 '--seed','0','--batch-size','16','--workers','4','--lr','0.0002']
RECIPES=[('A4_full',['--variant','gat']),('A4_self',['--variant','gat','--graph-control','self']),
 ('A4_zero_geometry',['--variant','gat','--graph-control','zero-geometry']),
 ('A4_relation',['--variant','gat','--graph-control','relation']),
 ('A3_relation001',['--relation-weight','0.01']),('A3_relation005',['--relation-weight','0.05'])]


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    status={'created':time.time(),'recipes':dict(RECIPES),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'jobs':{}}
    def save():
        tmp=OUT/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(OUT/'status.json')
    def run(name,flags,smoke):
        label=name+('_smoke' if smoke else '')
        cmd=BASE+flags+['--epochs','1' if smoke else '5','--output',str(OUT/label)]
        if smoke:cmd+=['--max-batches','1']
        status['jobs'][label]={'command':cmd,'status':'running','started':time.time()};save()
        with (OUT/(label+'.log')).open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            status['jobs'][label]['pid']=proc.pid;save();code=proc.wait()
        status['jobs'][label].update(status='passed' if code==0 else 'failed',exit_code=code,finished=time.time());save()
        if code:raise RuntimeError('failed '+label+'; full queue stops')
    for name,flags in RECIPES:run(name,flags,True)
    status['gate']='waiting E1 full completion before full E2/E3';save()
    while not (ROOT/'runs/static_E1_seed0_lr5e5_20261010/report.json').exists():time.sleep(5)
    status['gate']='all smoke passed; full E2/E3';save()
    for name,flags in RECIPES:run(name,flags,False)
    status['completed']=time.time();save()

if __name__=='__main__':main()
