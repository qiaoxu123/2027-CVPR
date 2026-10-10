"""Durable one-model workflow; no baseline reruns, no tuning on Unseen."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from graphnav.checkpointing import atomic_json,file_sha256,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','cache','sbf-source','sbf-config','data-root','a3-checkpoint'):p.add_argument('--'+key,required=True)
    p.add_argument('--first-epoch-only',action='store_true',help='Latest user instruction: hold after full epoch1+Seen128')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    state=dict(args=vars(a),provenance=provenance(),started_unix=time.time(),status='running',
        policy='One joint model,planned20 epochs,seed0,batch16,BF16,workers4; Seen-selected Best. No SBF/HETT reruns.')
    def run(stage,command):
        state.update(stage=stage,command=command,stage_started_unix=time.time())
        atomic_json(state,root/'workflow_status.json')
        with (root/(stage+'.log')).open('a') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;atomic_json(state,root/'workflow_status.json')
            returncode=process.wait()
        state.update(stage_finished_unix=time.time(),returncode=returncode)
        atomic_json(state,root/'workflow_status.json')
        if returncode:raise RuntimeError(f'{stage} failed with exit {returncode}; see retained stage log')
    try:
        prep=[sys.executable,'scripts/prepare_joint_active_data.py','--sbf-source',a.sbf_source,
            '--data-root',a.data_root,'--a3-checkpoint',a.a3_checkpoint,'--output',a.cache,'--microbatch','8']
        if Path(a.cache).exists():prep+=['--resume']
        run('prepare_frozen_features',prep)
        training=root/'training'
        train=[sys.executable,'scripts/train_joint_active.py','--train-cache',a.cache,
            '--a3-checkpoint',a.a3_checkpoint,'--output',str(training),'--sbf-source',a.sbf_source,
            '--sbf-config',a.sbf_config,'--data-root',a.data_root,'--epochs','20','--batch-size','16',
            '--workers','4','--seed','0','--lr','0.0001']
        if (training/'last.pt').exists():train+=['--resume',str(training/'last.pt')]
        if a.first_epoch_only:train+=['--stop-after-epoch','1']
        run('joint_training_20e',train)
        if a.first_epoch_only:
            state.update(status='epoch1_complete_awaiting_user_review',finished_unix=time.time())
            atomic_json(state,root/'workflow_status.json');return
        complete=json.loads((training/'training_complete.json').read_text())
        if complete['epochs']!=20 or complete['smoke']:raise ValueError('20 complete full epochs required')
        checkpoint=training/'best.pt'
        import torch
        selected=torch.load(checkpoint,map_location='cpu',weights_only=True)
        frozen=dict(epoch=selected['epoch'],completed_training_epochs=20,
            checkpoint=str(checkpoint.resolve()),checkpoint_sha256=file_sha256(checkpoint),
            selection='Best Seen128 SR then SPL within20 completed epochs; independent of all Unseen results',
            provenance=provenance(),frozen_unix=time.time(),data_version='Revised CityNav',
            no_further_model_or_threshold_tuning=True)
        freeze_path=root/'frozen_final_candidate.json';atomic_json(frozen,freeze_path)
        for split in ('val_seen','val_unseen','test_unseen'):
            target=Path(a.data_root)/'processed_citynav'/f'citynav_{split}.json'
            if not target.is_file():
                atomic_json(dict(split=split,status='blocked_missing_local_annotations',path=str(target)),root/(split+'_missing.json'))
                if split!='test_unseen':raise FileNotFoundError(target)
                state['test_unseen_status']='missing_local_annotations';continue
            destination=root/('final_'+split)
            if (destination/'metrics.json').exists():
                old=json.loads((destination/'metrics.json').read_text())
                if old['checkpoint_sha256']!=frozen['checkpoint_sha256']:raise ValueError('different existing final candidate')
                continue
            command=[sys.executable,'scripts/evaluate_closedloop.py','--model','joint-active',
                '--sbf-source',a.sbf_source,'--config',a.sbf_config,'--checkpoint',str(checkpoint),
                '--data-root',a.data_root,'--split',split,'--output',str(destination)]
            if split=='test_unseen':command+=['--frozen-final-candidate',str(freeze_path)]
            run('final_'+split,command)
        state.update(status='experiments_complete_reports_pending',finished_unix=time.time())
        atomic_json(state,root/'workflow_status.json')
    except BaseException:
        state.update(status='failed',finished_unix=time.time(),error=traceback.format_exc())
        atomic_json(state,root/'workflow_status.json');raise


if __name__=='__main__':main()
