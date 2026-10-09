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
        self.record=None

    def set_episode(self, episode, transform, catalog):
        self.record=public_static_record(episode,transform,catalog,max_landmarks=self.args.max_landmarks)

    @torch.no_grad()
    def predict(self,state,instruction):
        key=json.dumps(self.record,sort_keys=True)
        if key not in self.cache:
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
            top=int(field.belief_logits[0].argmax())
            uv=inputs['grid_xy'][0,top].float().cpu().numpy()
            bounds=np.array(self.record['world_bounds'])
            self.cache[key]=bounds[:2]+uv*(bounds[2:]-bounds[:2])
        # No target z: retain current legal altitude; shared controller enforces limits.
        xy=self.cache[key]
        return SimpleNamespace(xyz=np.array([*xy,state.pose[2]]),should_stop=False,
            diagnostics={'policy':'A3 Top1 bounded waypoint; untrained Trajectory/Stop disabled'})
