"""Wait for full verified Seen; launch one frozen Val-Unseen confirmation only if accepted."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.navigation_acceptance import assess_seen
from graphnav.workspace_paths import PROJECT,WORKSPACE


def main():
    criteria_path=Path('aerialvg_native_validation_acceptance_20261101.json');criteria=json.loads(criteria_path.read_text());criteria_sha=file_sha256(criteria_path)
    root=Path('aerialvg_native_temporal_navigation_full_seen_20261101');verification_path=root/'verification.json';start=time.monotonic()
    while not verification_path.exists():
        process=subprocess.run(['ps','-p','749619','-o','args='],capture_output=True,text=True)
        if 'resume_native_temporal_full_seen_20261101.sh' not in process.stdout:raise RuntimeError('Seen runner ended before full verification')
        if time.monotonic()-start>7200:raise TimeoutError('Full Seen wait exceeded2h')
        time.sleep(10)
    verification=json.loads(verification_path.read_text());paired=Path('aerialvg_native_temporal_full_seen_paired_20261101.json');summary=json.loads(paired.read_text());freeze_path=Path('aerialvg_native_temporal_frozen_seen_candidate_20261101.json')
    if verification['freeze_sha']!=file_sha256(freeze_path) or verification['trace_sha']!=file_sha256(root/'predictions.jsonl') or verification['report_sha']!=file_sha256(root/'metrics.json'):raise ValueError('Verified Seen artifacts changed')
    decision=assess_seen(summary,verification,criteria);decision.update(criteria_sha=criteria_sha,paired_sha=file_sha256(paired),verification_sha=file_sha256(verification_path),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(decision,Path('aerialvg_native_full_seen_decision_20261101.json'));print(json.dumps(decision),flush=True)
    if not decision['accepted_for_heldout']:return
    frozen=json.loads(freeze_path.read_text());frozen.update(seen_validation_complete=True,seen_full_paired_sha=file_sha256(paired),seen_full_verification_sha=file_sha256(verification_path),selection_criteria_sha=criteria_sha,original_freeze_sha=file_sha256(freeze_path),heldout_protocol='Full Val-Unseen, same frozen native checkpoint/history/control, no retuning or Test access')
    selected=Path('aerialvg_native_temporal_seen_selected_20261101.json');atomic_json(frozen,selected)
    sbf=PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source';base=PROJECT/'GraphNav/runs/graphnav_v1_stage1';common=['--sbf-source',str(sbf),'--config',str(sbf/'configs/sbfnav/main.yaml'),'--checkpoint',str(base/'best.pt'),'--data-root',str(WORKSPACE/'DATA/sbfnav-data'),'--split','val_unseen','--model','graphnav-v1','--frozen-final-candidate',str(base/'frozen_final_candidate.json'),'--evaluation-concurrency','1']
    baseline='aerialvg_original_navigation_full_unseen_confirm_20261101';candidate='aerialvg_native_temporal_navigation_full_unseen_confirm_20261101'
    for module,output,extra in [('scripts.evaluate_native_shared_closedloop',baseline,[]),('scripts.evaluate_native_object_closedloop',candidate,['--native-checkpoint',frozen['native_checkpoint'],'--native-history-window',str(frozen['native_history_window']),'--native-control',frozen['native_control'],'--native-frozen-manifest',str(selected)])]:
        with open(output+'.log','w') as stream:subprocess.run([sys.executable,'-u','-m',module,*common,*extra,'--output',output],stdout=stream,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-m','scripts.summarize_native_navigation','--baseline',baseline,'--candidate',candidate,'--output','aerialvg_native_temporal_full_unseen_confirm_paired_20261101.json'],check=True)
    count=json.loads((Path(candidate)/'metrics.json').read_text())['metrics']['episodes']
    subprocess.run([sys.executable,'-m','scripts.verify_native_navigation_result','--directory',candidate,'--frozen',str(selected),'--expected-episodes',str(count),'--require-full'],check=True)


if __name__=='__main__':main()
