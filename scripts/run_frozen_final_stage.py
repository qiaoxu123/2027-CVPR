"""Wait for fair Seen checks, Commit final candidates, then evaluate once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True);p.add_argument('--proxy',default='http://127.0.0.1:17890')
    a=p.parse_args();status_path=Path('runs/frozen_final_stage.status.json')
    if status_path.exists():raise ValueError('refusing controller status overwrite')
    status=dict(status='waiting_seen',started=time.time())
    def save():status_path.write_text(json.dumps(status,indent=2))
    save()
    try:
        while True:
            try:states={name:json.loads(Path('runs/'+name+'_queue.status.json').read_text())['status'] for name in ('phase20','phase21b','phase22')}
            except json.JSONDecodeError:
                time.sleep(1);continue # Concurrent status writer; no incomplete check can pass.
            if any(state.startswith('failed') for state in states.values()):raise ValueError('prerequisite failed: '+str(states))
            if all(state=='complete' for state in states.values()):break
            time.sleep(15)
        if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='2027-CVPR/graphnav-closedloop-sbf-parity':raise ValueError('wrong experiment branch')
        if subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip():raise ValueError('unrelated staged changes; refusing automatic commit')
        plan='reports/sbf_protocol/final_frozen_evaluation_plan.json'
        subprocess.run([a.python,'scripts/freeze_final_eval.py','--output',plan],check=True)
        state=json.loads(Path(plan).read_text());status.update(status='committing_frozen_candidates',primary_candidate=state['primary_candidate']);save()
        progress=Path('docs/GRAPHNAV_GOAL_PROGRESS.md')
        with progress.open('a') as f:f.write('\n## 最终候选冻结（任何新GraphNav Unseen评估之前）\n\n主候选由完整Seen选择为`'+state['primary_candidate']+'`；权重SHA、逐模块与三个完整训练种子的最终评估方案见`reports/sbf_protocol/final_frozen_evaluation_plan.json`。该清单先Commit再开始Unseen，不根据新Unseen结果调整Checkpoint或配置。公开高度Stop修复已完成Seen确认；所有旧失败结果保留。\n')
        subprocess.run(['git','add',plan,str(progress)],check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Freeze final GraphNav candidates using Seen only before Unseen evaluation'],check=True)
        status['frozen_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();save()
        env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',HTTPS_PROXY=a.proxy,HTTP_PROXY=a.proxy,ALL_PROXY='')
        try:
            pushed=subprocess.run(['git','-c','credential.helper=!gh auth git-credential','-c','http.proxy='+a.proxy,'push','https://github.com/qiaoxu123/2027-CVPR-Dev.git','HEAD:refs/heads/2027-CVPR/graphnav-closedloop-sbf-parity'],env=env,timeout=120)
            push_code=pushed.returncode
        except subprocess.TimeoutExpired:push_code='timeout; local frozen commit preserved, remote retry required'
        status.update(push_returncode=push_code,status='running_frozen_final');save()
        subprocess.run([sys.executable,'scripts/run_experiment_queue.py','--manifest',plan],check=True)
        status.update(status='complete_final_evaluations',finished=time.time());save()
    except Exception as error:
        status.update(status='failed',reason=str(error),finished=time.time());save();raise


if __name__=='__main__':main()
