"""Same evaluator; capture all steps of first three public queries for real figure sources."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from graphnav import diagnostic_fields
from scripts import evaluate_closedloop

def capture(policy,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);box={}
    # IDs are logger metadata only, never forwarded to the policy/model.
    manifest=json.loads((directory.parent/'episodes.json').read_text())
    predictions=directory.parent/'predictions.jsonl'
    completed=sum(1 for _ in predictions.open('rb')) if predictions.exists() else 0
    cursor=completed-1;original_set=policy.set_episode
    def set_episode(*args,**kwargs):
        nonlocal cursor
        cursor+=1
        return original_set(*args,**kwargs)
    policy.set_episode=set_episode
    def hook(module,args,result):
        if box.get('capture'):box['field']=result
    handle=policy.model.register_forward_hook(hook);original=policy.predict
    def predict(state,instruction):
        key=hashlib.sha256((policy.record['map_name']+'\n'+instruction).encode()).hexdigest()[:16]
        box['capture']=0<=cursor<3
        result=original(state,instruction)
        if box['capture']:
            field=box.pop('field');o=state.current_observation
            path=directory/f'episode{cursor:05d}_{key}_step{int(state.step):03d}.npz'
            if not path.exists():
                np.savez_compressed(path,episode_id=json.dumps(manifest[cursor]['episode_id']),episode_index=int(manifest[cursor]['index']),static_logits=field.static.belief_logits[0].float().cpu().numpy(),belief_logits=field.belief.belief_logits[0].float().cpu().numpy(),visual_delta=field.delta[0].float().cpu().numpy(),observed=field.observed[0].cpu().numpy(),world_xy=policy.world[0].cpu().numpy(),bgr=o['bgr'],footprint_world=o['footprint_world'],pose=np.asarray(state.pose),step=int(state.step),grid_size=policy.args.grid_size,map_name=policy.record['map_name'],instruction=instruction)
            result.diagnostics['saved_dense_field']=str(path)
        box['capture']=False
        return result
    policy.predict=predict
    def remove():policy.predict=original;policy.set_episode=original_set;handle.remove()
    return remove

if __name__=='__main__':
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    diagnostic_fields.install_field_capture=capture
    evaluate_closedloop.main()
