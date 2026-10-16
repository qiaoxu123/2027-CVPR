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
        self.visited_goal_cells=set()

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
            self.public_node_emb=names
            self.public_entities=dataset._prepared(0)[0]['entities']
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
    def __init__(self,checkpoint,head_checkpoint,scratch_directory,*,device='cuda',stop_threshold=.5,execution='joint',explore_candidates=0):
        super().__init__(checkpoint,scratch_directory,device=device)
        self.load_trajectory_head(checkpoint,head_checkpoint,stop_threshold=stop_threshold,execution=execution,explore_candidates=explore_candidates)

    def load_trajectory_head(self,checkpoint,head_checkpoint,*,stop_threshold=.5,execution='joint',explore_candidates=0):
        from .trajectory_policy import CandidateTrajectoryStop
        from .checkpointing import file_sha256,resolve_checkpoint_resource
        state=torch.load(head_checkpoint,map_location='cpu',weights_only=True)
        cache=resolve_checkpoint_resource(state['args']['cache'],head_checkpoint)/'manifest.json'
        if file_sha256(cache)!=state['input_fingerprint']['cache_manifest']['sha256']:
            raise ValueError('trajectory head source cache manifest changed')
        manifest=json.loads(cache.read_text())
        if manifest['protocol']['checkpoint_sha256']!=file_sha256(checkpoint):
            raise ValueError('trajectory head and static A3 checkpoint differ')
        if state['protocol']['observed_fraction']!=0.:
            raise ValueError('geometry/pose runtime only accepts matching zero visual-history inputs')
        if not 0<stop_threshold<1:raise ValueError('Stop threshold must be in (0,1)')
        self.stop_threshold=stop_threshold
        if execution not in ('joint','fixed-goal','fixed-goal-no-stop','direct-goal','guarded-path','guarded-blend','guarded-direct'):
            raise ValueError('unknown trajectory execution ablation')
        self.execution=execution
        if not 0<=explore_candidates<=8:raise ValueError('bounded exploration candidates must be between0 and8')
        if explore_candidates and not execution.startswith('guarded-'):raise ValueError('exploration requires predicted-goal guard')
        self.explore_candidates=explore_candidates
        self.stop_positive_weight=state['protocol']['stop_positive_weight']
        self.head=CandidateTrajectoryStop(hidden=state['hidden']).to(self.device).eval()
        self.head.load_state_dict(state['model'],strict=True)

    @torch.no_grad()
    def predict(self,state,instruction):
        field,_=self.frozen_field()
        return self.predict_field(state,field)

    @torch.no_grad()
    def predict_field(self,state,field):
        import math
        from sbfnav.mapdata import GROUND_LEVEL
        probabilities=field.belief_logits.float().softmax(-1)
        inputs=dict(candidate_xy=field.candidate_xy.float(),candidate_context=field.candidate_context.half().float(),
            candidate_logits=field.belief_logits.gather(1,field.candidate_ids).float(),
            pose_world=torch.as_tensor(state.pose,dtype=torch.float32,device=self.device)[None],
            bounds=torch.tensor(self.record['world_bounds'],device=self.device,dtype=torch.float32)[None],
            ground_level=torch.tensor([GROUND_LEVEL[self.record['map_name']]],device=self.device),
            belief_entropy=(-(probabilities*probabilities.clamp_min(1e-8).log()).sum(-1)/math.log(probabilities.shape[-1])),
            observed_fraction=torch.zeros(1,device=self.device))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):output=self.head(**inputs)
        from .trajectory_policy import calibrated_stop_probability
        candidate,mode,waypoint,stop=output.select()
        if self.execution!='joint':
            limit=min(getattr(self,'explore_candidates',0),field.candidate_ids.shape[1])
            available=[i for i in range(limit) if int(field.candidate_ids[0,i]) not in self.visited_goal_cells] if limit else [0]
            available=available or [0]
            for position,index in enumerate(available):
                candidate=torch.tensor([index],dtype=torch.long,device=self.device)
                mode=output.mode_logits[:,index].argmax(-1)
                waypoint=output.paths_world[torch.arange(1,device=self.device),candidate,mode,0]
                stop=output.stop_logits[:,index]
                goal=inputs['bounds'][:,:2]+inputs['candidate_xy'][:,index]*(inputs['bounds'][:,2:]-inputs['bounds'][:,:2])
                if self.execution in ('direct-goal','guarded-direct','guarded-path','guarded-blend'):
                    if self.execution in ('direct-goal','guarded-direct'):waypoint=torch.cat((goal,inputs['pose_world'][:,2:3]),-1)
                    else:
                        from .trajectory_policy import convergent_first_waypoint
                        waypoint,_=convergent_first_waypoint(inputs['pose_world'],goal,waypoint,blend=.25 if self.execution=='guarded-blend' else 1.)
                distance=float((goal-inputs['pose_world'][:,:2]).norm())
                if limit and distance<=1e-3 and float(calibrated_stop_probability(stop,self.stop_positive_weight)[0])<self.stop_threshold:
                    # Plan another hypothesis, without erasing any unobserved belief.
                    self.visited_goal_cells.add(int(field.candidate_ids[0,index]))
                    if position+1<len(available):continue
                break
        probability=float(calibrated_stop_probability(stop,self.stop_positive_weight)[0])
        predicted_goal_distance=float((goal-inputs['pose_world'][:,:2]).norm()) if self.execution.startswith('guarded-') else None
        from .trajectory_policy import stop_arrival_allowed
        arrival_guard_m=getattr(self,'stop_arrival_guard_m',1e-3)
        arrival_guard=stop_arrival_allowed(predicted_goal_distance,arrival_guard_m)
        height=float(state.pose[2])-GROUND_LEVEL[self.record['map_name']]
        altitude_guard=10.-1e-5<=height<=100.+1e-5
        return SimpleNamespace(xyz=waypoint[0].float().cpu().numpy(),should_stop=(probability>=self.stop_threshold and self.execution!='fixed-goal-no-stop' and arrival_guard and altitude_guard),
            diagnostics=dict(policy='Frozen A3 + learned trajectory/rank/Stop; geometry/pose only',
                execution=self.execution,candidate=int(candidate[0]),mode=int(mode[0]),stop_probability=probability,predicted_goal_distance=predicted_goal_distance,stop_altitude_guard=altitude_guard,
                stop_arrival_guard_m=arrival_guard_m,stop_arrival_guard_passed=arrival_guard,
                explored_goal_cells=len(self.visited_goal_cells),selected_goal_xy=(inputs['bounds'][:,:2]+inputs['candidate_xy'][torch.arange(1,device=self.device),candidate]*(inputs['bounds'][:,2:]-inputs['bounds'][:,:2]))[0].float().cpu().tolist()))
