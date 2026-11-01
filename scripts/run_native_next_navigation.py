"""Sequential paired Seen pilots after completed on-policy training; no held-out tuning."""
import json,os,subprocess,sys,time
from pathlib import Path
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.workspace_paths import PROJECT,WORKSPACE


def main():
    completed=Path('aerialvg_native_plain_onpolicy128_sigma8_s0_20261101/complete.json')
    started=time.monotonic()
    while not completed.exists():
        process=subprocess.run(['pgrep','-f','^bash scripts/run_native_onpolicy_followup_20261101.sh$'],capture_output=True,text=True)
        if process.returncode:raise RuntimeError('Follow-up coordinator ended without completed training')
        if time.monotonic()-started>7200:raise TimeoutError('Training wait exceeded2h')
        time.sleep(5)
    diagnostic=Path('aerialvg_native_temporal_train128_20261101.json');report=json.loads(diagnostic.read_text())
    window=int(max(report['windows'],key=lambda k:(report['windows'][k]['R1'],-report['windows'][k]['NE_m'],-int(k))))
    selection=dict(history_window=window,selection='Train-only causal state R1 then lower NE then shorter window; no Seen feedback',diagnostic_sha=file_sha256(diagnostic))
    atomic_json(selection,Path('aerialvg_native_temporal_selection_20261101.json'))
    sbf=PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'
    common=['--sbf-source',str(sbf),'--config',str(sbf/'configs/sbfnav/main.yaml'),'--checkpoint',str(PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt'),'--data-root',str(WORKSPACE/'DATA/sbfnav-data'),'--split','val_seen','--model','graphnav-v1','--max-episodes','128','--evaluation-concurrency','1']
    jobs=[('aerialvg_native_onpolicy_navigation_seen128_20261101','aerialvg_native_plain_onpolicy128_sigma8_s0_20261101/best.pt',1)]
    if window>1:jobs.append(('aerialvg_native_temporal_navigation_seen128_20261101','aerialvg_native_plain_sigma8_s0_20261101/best.pt',window))
    for output,checkpoint,history in jobs:
        with open(output+'.log','w') as stream:
            subprocess.run([sys.executable,'-u','-m','scripts.evaluate_native_object_closedloop',*common,'--native-checkpoint',checkpoint,'--native-history-window',str(history),'--output',output],check=True,stdout=stream,stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,'-m','scripts.summarize_native_navigation','--baseline','aerialvg_original_navigation_seen128_20261101','--candidate',output,'--output',output+'_paired.json'],check=True)


if __name__=='__main__':main()
