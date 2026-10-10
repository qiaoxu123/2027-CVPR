"""Durable full Stage1 A3 initialization, A/B/C training, frozen evaluation and ablations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256,provenance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='runs/graphnav_v1_stage1');p.add_argument('--cache',default='data/graphnav_stage1_frames');p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    workflow=root/'workflow_status.json'
    if workflow.exists() and not a.resume:raise ValueError('explicit workflow resume required')
    status=json.loads(workflow.read_text())if workflow.exists()else dict(jobs={},started=time.time(),provenance=provenance())
    env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(Path.cwd()))
    sources=root/'initial_sources.json';base=json.loads((root/'baseline_seen128/metrics.json').read_text())['args']
    # Make the additional Seen controls part of the durable workflow on resume.
    audit=root/'audit_status.json'
    audit_state=json.loads(audit.read_text())if audit.exists()else{}
    audit_live=False
    if audit_state.get('pid') and audit_state.get('status')not in ('complete','failed'):
        try:os.kill(audit_state['pid'],0);audit_live=True
        except ProcessLookupError:pass
    if audit_state.get('status')!='complete' and not audit_live:
        with (root/'audit_console.log').open('a')as stream:
            subprocess.Popen([sys.executable,'-m','scripts.audit_graphnav_stage1','--output',str(root)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    def save():atomic_json(status,workflow)
    def run(name,command):
        status.update(stage=name,status='running');status['jobs'][name]=dict(command=command,started=time.time(),status='running');save()
        with (root/(name+'.log')).open('a')as f:
            child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT);status['child_pid']=child.pid;save();code=child.wait()
        status['jobs'][name].update(status='passed'if code==0 else'failed',exit_code=code,finished=time.time());save()
        if code:raise RuntimeError(name+' failed; retained log and checkpoints, resume after fixing')
    def evaluate(name,checkpoint,split='val_seen',limit=128,visual='real',history='checkpoint',freeze=None):
        destination=root/name
        if (destination/'metrics.json').exists():
            old=json.loads((destination/'metrics.json').read_text())
            if old['checkpoint_sha256']!=file_sha256(checkpoint):raise ValueError('existing evaluation uses another candidate')
            return old
        if destination.exists():destination.rename(root/(name+'_failed_'+str(int(time.time()))))
        cmd=[sys.executable,'-m','scripts.evaluate_closedloop','--model','graphnav-v1','--sbf-source',base['sbf_source'],'--config',base['config'],'--data-root',base['data_root'],'--checkpoint',str(checkpoint),'--split',split,'--output',str(destination),'--stage1-visual-control',visual,'--stage1-history-control',history,'--stop-threshold','.3','--head-execution','guarded-blend']
        if limit is not None:cmd+=['--max-episodes',str(limit)]
        if freeze:cmd+=['--frozen-final-candidate',str(freeze)]
        run(name,cmd);return json.loads((destination/'metrics.json').read_text())
    try:
        status.update(status='waiting_full_causal_features',stage='features');save()
        while not (Path(a.cache)/'manifest.json').exists()or not json.loads((Path(a.cache)/'manifest.json').read_text())['complete']:time.sleep(5)
        if not json.loads((root/'initial_complete_parity.json').read_text())['poses_and_stop_exact']:raise ValueError('initial strong controller compatibility required')
        for stage in ('A','B','C'):
            if not json.loads((root/('smoke_'+stage+'_v3')/'complete.json').read_text())['smoke']:raise ValueError('CUDA smoke required')
        previous=None
        for stage,epochs in [('A',3),('B',3),('C',5)]:
            destination=root/'stages'/stage
            if not (destination/'complete.json').exists():
                cmd=[sys.executable,'-m','scripts.train_graphnav_stage1','--cache',a.cache,'--sources',str(sources),'--stage',stage,'--output',str(destination),'--epochs',str(epochs),'--batch-size','16','--workers','4','--seed','0','--sbf-source',base['sbf_source'],'--sbf-config',base['config'],'--data-root',base['data_root']]
                if previous:cmd+=['--init',str(previous)]
                if (destination/'last.pt').exists():cmd+=['--resume',str(destination/'last.pt')]
                run('train_'+stage,cmd)
            completed=json.loads((destination/'complete.json').read_text())
            if completed['epochs']!=epochs or completed['smoke']:raise ValueError('full planned stage epochs required')
            previous=destination/'best.pt'
        candidates=[]
        for stage in ('B','C'):
            checkpoint=root/'stages'/stage/'best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            candidates.append((tuple(state['best_key']),stage,checkpoint))
        _,selected_stage,selected=max(candidates,key=lambda x:x[0])
        status['selected_stage']=selected_stage;status['selection']='Best complete B/C checkpoint by Seen128 SR, SPL, Train Dev Recall. No Unseen consulted';save()
        if not (root/'best.pt').exists():shutil.copy2(selected,root/'best.pt')
        if file_sha256(root/'best.pt')!=file_sha256(selected):raise ValueError('selected candidate changed; refuse overwrite')
        if not (root/'last.pt').exists():shutil.copy2(root/'stages/C/last.pt',root/'last.pt')
        checkpoint=root/'best.pt'
        # All diagnostic controls are Seen-only; none changes the selected weights.
        for label,visual,history in [('selected_seen128','real','checkpoint'),('zero_rgb_seen128','zero','checkpoint'),('shuffle_rgb_seen128','shuffle','checkpoint'),('no_belief_seen128','no-belief','checkpoint'),('current_only_seen128','real','current')]:
            evaluate(label,checkpoint,visual=visual,history=history)
        freeze=root/'frozen_final_candidate.json'
        if not freeze.exists():atomic_json(dict(checkpoint_sha256=file_sha256(checkpoint),checkpoint=str(checkpoint),selected_stage=selected_stage,all_stages_complete=True,completed_epochs=dict(A=3,B=3,C=5),visual_control='real',history_control='checkpoint',stop_threshold=.3,active_observation=False,selection=status['selection'],frozen=time.time(),provenance=provenance()),freeze)
        evaluate('full_seen',checkpoint,limit=None)
        evaluate('full_unseen',checkpoint,split='val_unseen',limit=None,freeze=freeze)
        # Frozen independent B0 release candidate, measured after final selection.
        baseline_unseen=root/'baseline_full_unseen'
        if not (baseline_unseen/'metrics.json').exists():
            if baseline_unseen.exists():baseline_unseen.rename(root/('baseline_full_unseen_failed_'+str(int(time.time()))))
            src=json.loads(sources.read_text());cmd=[sys.executable,'-m','scripts.evaluate_closedloop','--model','static-transfer','--sbf-source',base['sbf_source'],'--config',base['config'],'--data-root',base['data_root'],'--checkpoint',src['a3']['path'],'--head-checkpoint',src['head']['path'],'--reference-a3-checkpoint',src['reference']['path'],'--split','val_unseen','--output',str(baseline_unseen),'--stop-threshold','.3','--head-execution','guarded-blend']
            run('baseline_full_unseen',cmd)
        full=Path('../2027-CVPR-full-token-rgb/data/revised_static')
        src=json.loads(sources.read_text())
        for split in ('seen','unseen'):
            for name,path in [('B0',src['a3']['path']),('selected',str(checkpoint))]:
                destination=root/f'static_{name}_{split}.json'
                if destination.exists():continue
                run(f'static_{name}_{split}',[sys.executable,'-m','scripts.evaluate_graphnav_stage1_static','--checkpoint',path,'--records',str(full/f'graphnav_val_{split}.jsonl'),'--source-a3',src['a3']['path'],'--output',str(destination)])
        run('summarize',[sys.executable,'-m','scripts.summarize_graphnav_stage1','--output',str(root)])
        status.update(status='complete',finished=time.time());save()
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),failed=time.time());save();raise


if __name__=='__main__':main()
