"""Save actual first-observation fields and supplied photo; accepts no labels."""
import hashlib
from pathlib import Path
import numpy as np


def install_field_capture(policy, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    box={}
    def hook(module, inputs, result):
        if box.get('capture'):
            box['field']=result
    handle=policy.model.register_forward_hook(hook)
    original=policy.predict
    def predict(state,instruction):
        box['capture']=state.step==0
        prediction=original(state,instruction)
        if box['capture']:
            field=box.pop('field')
            name=hashlib.sha256((policy.record['map_name']+'\n'+instruction).encode()).hexdigest()[:16]
            observation=state.current_observation
            np.savez_compressed(directory/(name+'.npz'),
                static_logits=field.static.belief_logits[0].float().cpu().numpy(),
                belief_logits=field.belief.belief_logits[0].float().cpu().numpy(),
                visual_delta=field.delta[0].float().cpu().numpy(),
                world_xy=policy.world[0].float().cpu().numpy(),
                bgr=observation['bgr'],footprint_world=observation['footprint_world'],
                grid_size=policy.args.grid_size,map_name=policy.record['map_name'],instruction=instruction)
        box['capture']=False
        return prediction
    policy.predict=predict
    def remove():
        policy.predict=original
        handle.remove()
    return remove
