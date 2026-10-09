"""Preserved A3 with a bounded waypoint policy; no learned Trajectory/Stop claim."""
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import torch

from .candidate_geometry import FullCandidateDataset, collate_full
from .full_experiment import _batch_inputs, new_model
from .token_language import FrozenBERTTokens
from .landmark_alias import resolve_references


def public_static_record(episode, transform, catalog, *, max_landmarks=16):
    bounds=np.array([transform.x_min,transform.y_min,transform.x_max,transform.y_max])
    landmarks=[]
    reference_names=episode.reference_names or tuple(lm.name for lm in episode.references)
    # Preserve all homonymous source fragments as in the verified A3 converter;
    # collapsing them to SBF's largest polygon would silently change the model input.
    resolved=resolve_references(reference_names,catalog.objects[episode.map_name],
        max_landmarks=max_landmarks,match_mode='fuzzy')
    for landmark in resolved:
        landmarks.append(dict(name=landmark['name'],object_type=landmark['object_type'],aliases=landmark['aliases'],
            contour=((np.asarray(landmark['contour'])-bounds[:2])/(bounds[2:]-bounds[:2])).tolist()))
    return dict(map_name=episode.map_name,instruction=episode.instruction,
        world_bounds=bounds.tolist(),physical_map_sizes=(bounds[2:]-bounds[:2]).tolist(),
        landmarks=landmarks,dataset_version='revised',
        # Packing scaffold, never a real target and never included in _batch_inputs.
        target_xy=[0.,0.])


class A3WaypointPolicy:
    def __init__(self, checkpoint, scratch_directory, *, device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.args=SimpleNamespace(**payload['args'])
        self.model=new_model(self.args,torch.device(device)).eval()
        self.model.load_state_dict(payload['model'],strict=True)
        self.encoder=FrozenBERTTokens(name=self.args.text_backbone,device=device,
            max_length=getattr(self.args,'max_text_length',96))
        self.device=torch.device(device)
        self.scratch=Path(scratch_directory);self.scratch.mkdir(parents=True,exist_ok=True)
        self.cache={}
        self.field_cache={}
        self.record=None

    def set_episode(self, episode, transform, catalog):
        self.record=public_static_record(episode,transform,catalog,max_landmarks=self.args.max_landmarks)

    @torch.no_grad()
    def frozen_field(self):
        key=json.dumps(self.record,sort_keys=True)
        if key not in self.field_cache:
            self.field_cache.clear() # One episode field; avoid retaining GPU tensors across the split.
            path=self.scratch/'current_public_query.jsonl'
            path.write_text(json.dumps(self.record)+'\n')
            dataset=FullCandidateDataset(path,grid_size=self.args.grid_size,
                max_landmarks=self.args.max_landmarks,max_roads=self.args.max_roads,
                max_relations=self.args.max_relations,prior_sigma_m=self.args.prior_sigma_m,
                rule_sigma_m=self.args.rule_sigma_m)
            batch=collate_full([dataset[0]])
            inputs,tokens,mask,names=_batch_inputs(batch,self.encoder,self.device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                field=self.model(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            self.field_cache[key]=(field,inputs)
        return self.field_cache[key]

    @torch.no_grad()
    def predict(self,state,instruction):
        key=json.dumps(self.record,sort_keys=True)
        if key not in self.cache:
            field,inputs=self.frozen_field()
            top=int(field.belief_logits[0].argmax())
            uv=inputs['grid_xy'][0,top].float().cpu().numpy()
            bounds=np.array(self.record['world_bounds'])
            self.cache[key]=bounds[:2]+uv*(bounds[2:]-bounds[:2])
        # No target z: retain current legal altitude; shared controller enforces limits.
        xy=self.cache[key]
        return SimpleNamespace(xyz=np.array([*xy,state.pose[2]]),should_stop=False,
            diagnostics={'policy':'A3 Top1 bounded waypoint; untrained Trajectory/Stop disabled'})


class A3TrajectoryPolicy(A3WaypointPolicy):
    """Frozen A3 and a trained candidate-aligned head, using public observations."""
    def __init__(self,checkpoint,head_checkpoint,scratch_directory,*,device='cuda',stop_threshold=.5):
        super().__init__(checkpoint,scratch_directory,device=device)
        from .trajectory_policy import CandidateTrajectoryStop
        from .checkpointing import file_sha256
        state=torch.load(head_checkpoint,map_location='cpu',weights_only=True)
        cache=Path(state['args']['cache'])/'manifest.json'
        manifest=json.loads(cache.read_text())
        if manifest['protocol']['checkpoint_sha256']!=file_sha256(checkpoint):
            raise ValueError('trajectory head and static A3 checkpoint differ')
        if state['protocol']['observed_fraction']!=0.:
            raise ValueError('geometry/pose runtime only accepts matching zero visual-history inputs')
        if not 0<stop_threshold<1:raise ValueError('Stop threshold must be in (0,1)')
        self.stop_threshold=stop_threshold
        self.head=CandidateTrajectoryStop(hidden=state['hidden']).to(self.device).eval()
        self.head.load_state_dict(state['model'],strict=True)

    @torch.no_grad()
    def predict(self,state,instruction):
        import math
        from sbfnav.mapdata import GROUND_LEVEL
        field,_=self.frozen_field();probabilities=field.belief_logits.float().softmax(-1)
        inputs=dict(candidate_xy=field.candidate_xy.float(),candidate_context=field.candidate_context.float(),
            candidate_logits=field.belief_logits.gather(1,field.candidate_ids).float(),
            pose_world=torch.as_tensor(state.pose,dtype=torch.float32,device=self.device)[None],
            bounds=torch.tensor(self.record['world_bounds'],device=self.device,dtype=torch.float32)[None],
            ground_level=torch.tensor([GROUND_LEVEL[self.record['map_name']]],device=self.device),
            belief_entropy=(-(probabilities*probabilities.clamp_min(1e-8).log()).sum(-1)/math.log(probabilities.shape[-1])),
            observed_fraction=torch.zeros(1,device=self.device))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):output=self.head(**inputs)
        candidate,mode,waypoint,stop=output.select();probability=float(stop.float().sigmoid()[0])
        return SimpleNamespace(xyz=waypoint[0].float().cpu().numpy(),should_stop=probability>=self.stop_threshold,
            diagnostics=dict(policy='Frozen A3 + learned trajectory/rank/Stop; geometry/pose only',
                candidate=int(candidate[0]),mode=int(mode[0]),stop_probability=probability))
