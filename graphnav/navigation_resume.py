"""Validate and preserve an interrupted native Seen rollout prefix before appending."""
import json,shutil
from pathlib import Path
from .checkpointing import file_sha256,atomic_json


def prepare_resume(root,frozen_path,native_checkpoint,base_checkpoint,history,control):
    root=Path(root);frozen=json.loads(Path(frozen_path).read_text())
    if (root/'metrics.json').exists():raise ValueError('Cannot resume a completed evaluation')
    if file_sha256(native_checkpoint)!=frozen['native_checkpoint_sha'] or file_sha256(base_checkpoint)!=frozen['base_checkpoint_sha'] or history!=frozen['native_history_window'] or control!=frozen['native_control']:raise ValueError('Resume differs from frozen candidate')
    path=root/'predictions.jsonl';raw=path.read_bytes()
    if not raw or not raw.endswith(b'\n'):raise ValueError('Resume prefix must contain complete newline-terminated episodes')
    rows=[json.loads(line) for line in raw.splitlines()];manifest=json.loads((root/'episodes.json').read_text())
    if not 0<len(rows)<len(manifest):raise ValueError('Resume prefix must be nonempty and incomplete')
    for row,episode in zip(rows,manifest):
        if row['episode_id']!=episode['episode_id'] or row['index']!=episode['index']:raise ValueError('Resume prefix identity mismatch')
        if len(row['poses'][0])!=len(episode['start_pose']) or any(abs(x-y)>1e-5 for x,y in zip(row['poses'][0],episode['start_pose'])):raise ValueError('Resume prefix initial pose mismatch')
        diagnostics=row['step_diagnostics']
        if not diagnostics or len(diagnostics)!=len(row['decision_seconds']):raise ValueError('Incomplete episode decisions')
        for step,d in enumerate(diagnostics):
            if d['observation_step']!=step or d['native_object_checkpoint_sha']!=frozen['native_checkpoint_sha'] or d.get('native_object_history_window',1)!=history or d['native_object_control']!=control or d['execution']!=frozen['head_execution'] or d['active_observation'] is not False:raise ValueError('Resume prefix inference controls mismatch')
        if row['termination']=='policy_stop':
            d=diagnostics[-1]
            if d['stop_probability']<frozen['stop_threshold'] or d['predicted_goal_distance']>1e-3 or not d['stop_altitude_guard']:raise ValueError('Resume prefix Stop guard mismatch')
    backup=root/f'predictions.interrupted_{len(rows)}.jsonl'
    if backup.exists():
        if file_sha256(backup)!=file_sha256(path):raise ValueError('Existing prefix backup differs')
    else:shutil.copyfile(path,backup)
    contract=dict(prefix_episodes=len(rows),total_episodes=len(manifest),prefix_sha=file_sha256(path),prefix_backup=backup.name,manifest_sha=file_sha256(root/'episodes.json'),freeze_sha=file_sha256(frozen_path),native_sha=file_sha256(native_checkpoint),base_sha=file_sha256(base_checkpoint),timing_scope='seconds and GPU peak cover resumed session; per-decision timings retain both sessions')
    old=root/'resume_contract.json'
    if old.exists():shutil.copyfile(old,root/f'resume_contract.before_{len(rows)}.json')
    atomic_json(contract,old);return contract


def load_prefix(root,expected_manifest):
    root=Path(root);contract=json.loads((root/'resume_contract.json').read_text());path=root/'predictions.jsonl'
    if file_sha256(path)!=contract['prefix_sha'] or file_sha256(root/'episodes.json')!=contract['manifest_sha']:raise ValueError('Resume inputs changed after validation')
    if json.loads((root/'episodes.json').read_text())!=expected_manifest:raise ValueError('Resume dataset/episode selection differs')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if len(rows)!=contract['prefix_episodes']:raise ValueError('Resume prefix count mismatch')
    return rows,contract
