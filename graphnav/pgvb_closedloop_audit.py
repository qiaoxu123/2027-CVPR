"""Read-only audit artifacts for frozen PGVB closed loops, never policy inputs."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from .landmark_visual_only import visual_only_world_belief


def pack_field(field):
    if field is None:return {}
    return {name:value[0].detach().float().cpu().numpy().copy() for name,value in (
        ('logits',field.belief_logits),('candidate_ids',field.candidate_ids),
        ('candidate_xy',field.candidate_xy),('candidate_context',field.candidate_context))}


class DecisionRecorder:
    """Called after/alongside forward; detached snapshots cannot change actions."""
    def __init__(self,policy,output):
        self.policy=policy;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.snapshot={};self.evidence=None;self.policy.audit_callback=self.capture

    def capture(self,original,final,evidence):
        self.snapshot={}
        for label,field in [('original',original),('final',final)]:
            self.snapshot.update({label+'_'+k:v for k,v in pack_field(field).items()})
        if evidence is not None:
            uv,scores,valid=evidence
            v=visual_only_world_belief(local_uv=uv,scores=scores,valid=valid)
            self.snapshot.update(pgvb_probability=v['probabilities'][0].cpu().numpy().copy(),
                                 pgvb_covered=v['covered'][0].cpu().numpy().copy(),
                                 local_uv=uv[0].cpu().numpy().copy(),scores=scores[0].cpu().numpy().copy(),valid=valid[0].cpu().numpy().copy())

    def predict(self,state,instruction):
        self.snapshot={}
        prediction=self.policy.predict(state,instruction)
        obs=state.current_observation;stem=f'step_{state.step:02d}'
        self.snapshot.update(footprint_world=np.asarray(obs['footprint_world']),pose=np.asarray(state.pose),bounds=np.asarray(self.policy.record['world_bounds']))
        np.savez_compressed(self.output/(stem+'.npz'),**self.snapshot)
        cv2.imwrite(str(self.output/(stem+'.png')),np.asarray(obs['bgr']))
        prediction.diagnostics.update(audit_artifact=str(self.output/(stem+'.npz')),
            rgb_sha256=hashlib.sha256(np.asarray(obs['bgr']).tobytes()).hexdigest(),
            observation_step=int(state.step),camera_footprint_world=np.asarray(obs['footprint_world']).tolist(),
            rgb_shape=list(np.asarray(obs['bgr']).shape))
        return prediction
