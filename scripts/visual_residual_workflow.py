"""Finish visual evidence calibration/control evaluation and publish honest live status."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from graphnav.checkpointing import atomic_json
from graphnav.workspace_paths import PROJECT,current_paths

ROOT=Path.cwd()
ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
BASE=current_paths(json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_seen/metrics.json').read_text())['args'])
EXPECTED={'val_seen':2470,'val_unseen':2697,'test_unseen':5281}


def report():
    text=['# 384 全量评测与视觉残差改进','', '状态会自动更新；未完成的 split 不报告整体 SR。','', '|384 split|已完成 / 总数|整体 SR|状态|','|---|---:|---:|---|']
    status={}
    for split,total in EXPECTED.items():
        directory=ROOT/'full_resolution_384'/split;metrics=directory/'metrics.json'
        count=sum(1 for _ in (directory/'predictions.jsonl').open()) if (directory/'predictions.jsonl').exists() else 0
        if metrics.exists():
            data=json.loads(metrics.read_text());assert data['metrics']['episodes']==total
            sr=f"{data['metrics']['SR']*100:.4f}%";state='complete'
        else:sr='待完整结束';state='running'
        status[split]=dict(completed=count,total=total,status=state)
        text.append(f'|{split}|{count} / {total}|{sr}|{state}|')
    text += ['', '冻结原 C Epoch1，实际网格 384×384，stop=0.3；Test 不参与视觉模型选取。目录内保留冻结 manifest、输入/权重 SHA、完整逐 episode 轨迹及续跑记录。', '', '## 视觉残差','', '两种训练候选：几何条件/质量置信度残差，及允许真实可见负证据整体降权的非中心化残差（用原生图文匹配代替 ego 距离捷径）。二者只训练新的残差头，静态主干、轨迹与 Stop 参数冻结；仅用 Train 标签，按原 Train Dev 整组划分选权重。', '', '|候选|训练完整|Dev 静态 R@1|Dev 修正 R@1|','|---|---|---:|---:|']
    for name in ('calibrated_residual','absolute_residual'):
        history=ROOT/name/'history.json'
        if history.exists():
            values=json.loads(history.read_text());best=max(values,key=lambda v:(v['dev']['corrected_R1'],-v['dev']['mass']))
            complete=(ROOT/name/'complete.json').exists()
            text.append(f"|{name}|{complete}|{best['dev']['static_R1']*100:.3f}%|{best['dev']['corrected_R1']*100:.3f}%|")
    evaluations=ROOT/'visual_evaluations';summaries={}
    for candidate in ('calibrated_residual','absolute_residual'):
        text += ['', f'### {candidate} 闭环配对对照','', '|控制|Seen128 SR|全 Seen SR|','|---|---:|---:|']
        for control in ('real','no-belief','zero','shuffle'):
            shown=[]
            for scope in ('seen128','full_seen'):
                file=evaluations/f'{candidate}_{scope}_{control}'/'metrics.json'
                if file.exists():
                    values=json.loads(file.read_text())['metrics'];shown.append(f"{values['SR']*100:.4f}%")
                else:shown.append('待执行')
            text.append(f"|{control}|{shown[0]}|{shown[1]}|")
        real=evaluations/f'{candidate}_full_seen_real'/'predictions.jsonl'
        baseline=evaluations/f'{candidate}_full_seen_no-belief'/'predictions.jsonl'
        if (real.parent/'metrics.json').exists() and (baseline.parent/'metrics.json').exists():
            a=[json.loads(line) for line in real.read_text().splitlines()];b=[json.loads(line) for line in baseline.read_text().splitlines()]
            assert [r['index'] for r in a]==[r['index'] for r in b]
            assert len(a)==2470
            diff=np.asarray([x['metrics']['SR']-y['metrics']['SR'] for x,y in zip(a,b)],np.float32)
            rng=np.random.default_rng(0);samples=[]
            for _ in range(40):samples.extend(diff[rng.integers(len(diff),size=(250,len(diff)))].mean(-1))
            lo,hi=np.quantile(samples,[.025,.975]);helped=int((diff>0).sum());hurt=int((diff<0).sum())
            summaries[candidate]=dict(episodes=len(a),sr_gain_pp=float(diff.mean()*100),bootstrap_95ci_pp=[float(lo*100),float(hi*100)],helped=helped,hurt=hurt,statistically_positive_gain_beyond_static=bool(lo>0))
            text += ['',f"完整 Seen：帮助 {helped} 条，损害 {hurt} 条，SR 差 {diff.mean()*100:+.3f}pp，配对 bootstrap 95%CI [{lo*100:+.3f}, {hi*100:+.3f}]pp。只有区间下界>0 才标为有明确正收益；同时查看 zero/shuffle 结果验证实际 RGB 贡献。"]
    atomic_json(dict(full_resolution=status,visual=summaries,updated=time.time()),ROOT/'live_status.json')
    (ROOT/'RESULTS.md').write_text('\n'.join(text)+'\n')


def run_job(module,arguments,logfile):
    with Path(logfile).open('a') as log:
        process=subprocess.Popen([sys.executable,'-m',module,*arguments],env=ENV,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:report();time.sleep(30)
        if process.returncode:raise RuntimeError(f'{module} failed; see {logfile}')


def main():
    # The first candidate was started interactively; never duplicate its optimizer.
    while not (ROOT/'calibrated_residual/complete.json').exists():
        if (ROOT/'workflow_failure.json').exists():return
        report();time.sleep(30)
    if not (ROOT/'absolute_residual/complete.json').exists():
        if (ROOT/'absolute_residual').exists():raise RuntimeError('Preserve incomplete absolute training for review')
        run_job('scripts.train_calibrated_visual_residual',['--epochs','3','--workers','4','--uncentered','--native-contrast','--shuffle-regularization','--rank-weight','.5','--output','absolute_residual'],'absolute_training.log')
    out=ROOT/'visual_evaluations';out.mkdir(exist_ok=True)
    # Each candidate is selected by Train Dev before any Seen closed-loop score.
    for candidate in ('calibrated_residual','absolute_residual'):
        for scope in ('seen128','full_seen'):
            for control in ('real','no-belief','zero','shuffle'):
                destination=out/f'{candidate}_{scope}_{control}'
                if (destination/'metrics.json').exists():continue
                arguments=['--model','graphnav-v1','--checkpoint',BASE['checkpoint'],'--sbf-source',BASE['sbf_source'],'--config',BASE['config'],'--data-root',BASE['data_root'],'--split','val_seen','--stage1-residual-checkpoint',str(ROOT/candidate/'best.pt'),'--stage1-visual-control',control,'--output',str(destination)]
                if scope=='seen128':arguments+=['--max-episodes','128']
                if destination.exists():arguments.append('--resume')
                run_job('scripts.evaluate_closedloop',arguments,str(destination)+'.log')
    while not all((ROOT/'full_resolution_384'/split/'metrics.json').exists() for split in EXPECTED):
        report();time.sleep(30)
    report();atomic_json(dict(complete=True,finished=time.time()),ROOT/'workflow_complete.json')


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(dict(error=str(error),time=time.time()),ROOT/'workflow_failure.json');raise
