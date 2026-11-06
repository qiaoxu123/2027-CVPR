"""V3 frozen causal observation interface. No target, ID, trajectory or GT crops."""
import json
import hashlib
import numpy as np
from pathlib import Path
import cv2
import torch
from torch.nn import functional as F
from scripts.unified_pgvb_v3_pilot import (_load_base,blind_public_landmarks,
    STATIC_KEYS,V3_FORMAT,_depth_patch_tensor)
from .unified_pgvb_v3 import UnifiedPGVBV3
from .token_language import FrozenBERTTokens
from .appearance_matching import ObservedSigLIPMatchingEncoder
from .candidate_geometry import FullCandidateDataset,collate_full
from .full_experiment import _batch_inputs
from .checkpointing import file_sha256
from .full_localization import GoalField

class V3CausalBelief:
    def __init__(self,source_v2,checkpoint,directory,device='cuda',skip_rgb_encoder=False):
        self.device=torch.device(device);self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        bundle=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if bundle.get('format')!=V3_FORMAT or bundle['manifest']['source_v2_sha256']!=file_sha256(source_v2):
            raise ValueError('V3 source/checkpoint mismatch')
        geom,self.cfg=_load_base(source_v2,device);self.model=UnifiedPGVBV3(geom).to(device)
        self.model.load_state_dict(bundle['model'],strict=True);self.model.eval().requires_grad_(False)
        self.bert=FrozenBERTTokens(self.cfg.text_backbone,device=device,max_length=self.cfg.max_text_length)
        self.skip_rgb_encoder=bool(skip_rgb_encoder)
        self.native=None if self.skip_rgb_encoder else ObservedSigLIPMatchingEncoder(device=device,microbatch=1)
        self.frames=[];self.frame_keys=set();self.record=None

    @torch.no_grad()
    def start(self,*,map_name,instruction,world_bounds,public_objects):
        # Explicit whitelist: callers cannot forward a dataset record/target metadata.
        record=dict(map_name=map_name,instruction=instruction,world_bounds=list(world_bounds),physical_map_sizes=(np.asarray(world_bounds[2:])-np.asarray(world_bounds[:2])).tolist(),landmarks=[],dataset_version='revised',target_xy=[0.,0.])
        blind_public_landmarks([record],{map_name:public_objects},self.cfg.max_landmarks)
        self.record=record;self.frames=[];self.frame_keys=set()
        path=self.directory/'public_record.jsonl';path.write_text(json.dumps(record)+'\n')
        ds=FullCandidateDataset(path,grid_size=28,max_landmarks=self.cfg.max_landmarks,max_roads=self.cfg.max_roads,max_relations=self.cfg.max_relations,prior_sigma_m=self.cfg.prior_sigma_m,rule_sigma_m=self.cfg.rule_sigma_m,deduplicate_queries=False)
        static,tokens,mask,names=_batch_inputs(collate_full([ds[0]]),self.bert,self.device,'geometry')
        self.static={k:static[k] for k in STATIC_KEYS};self.tokens=tokens.half().float();self.mask=mask;self.names=names.half().float()
        self.text=torch.zeros(1,1024,device=self.device) if self.skip_rgb_encoder else self.native.texts([instruction]).half().float();self.bounds=torch.tensor([world_bounds],device=self.device,dtype=torch.float32)

    @torch.no_grad()
    def observe(self,*,observation,pose,geometry_only=False):
        if self.record is None:raise RuntimeError('Call start with public inputs')
        key=ObservedSigLIPMatchingEncoder.image_key(observation['bgr'])+hashlib.sha256(
            np.asarray(observation['footprint_world'],dtype=np.float64).tobytes()+np.asarray(pose,dtype=np.float64).tobytes()
            +np.asarray(observation['depth_valid'],dtype=np.bool_).tobytes()).hexdigest()
        depth,valid=_depth_patch_tensor(observation)
        if self.skip_rgb_encoder or not self.record['landmarks']:valid.zero_()
        if key not in self.frame_keys:
            if valid.any():
                image=self.native.processor(images=[cv2.cvtColor(observation['bgr'],cv2.COLOR_BGR2RGB)],return_tensors='pt').to(self.device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                    patches=self.native.model.vision_model(**image).last_hidden_state[0]
                patches=patches.half().float()
            else:patches=torch.zeros(576,1024,device=self.device)
            # One real observation now, at most five causal visited views thereafter.
            frame=dict(key=key,rgb=patches,depth=depth.to(self.device).half().float(),valid=valid.to(self.device),
                       footprint=torch.as_tensor(observation['footprint_world'],device=self.device,dtype=torch.float32),
                       pose=torch.as_tensor(pose,device=self.device,dtype=torch.float32))
            self.frames.append(frame);self.frame_keys.add(key)
            if len(self.frames)>5:
                removed=self.frames.pop(0);self.frame_keys.remove(removed['key'])
        kw=dict(static_inputs=self.static,bert_tokens=self.tokens,bert_mask=self.mask,landmark_embeddings=self.names,siglip_text=self.text,
                rgb_patches=torch.stack([f['rgb'] for f in self.frames])[None],depth_patches=torch.stack([f['depth'] for f in self.frames])[None],
                patch_valid=torch.stack([f['valid'] for f in self.frames])[None],footprints_world=torch.stack([f['footprint'] for f in self.frames])[None],
                frame_poses=torch.stack([f['pose'] for f in self.frames])[None],world_bounds=self.bounds)
        self.last_output=self.model(**kw)
        return self.last_output

    @staticmethod
    def controller_field(output,*,geometry_only=False,top_k=20):
        """Gather coordinates/logits/context from the SAME dense global IDs."""
        logits=(output.geometry_global if geometry_only else output.fused_logits).flatten(1)
        ids=logits.topk(min(top_k,logits.shape[1]),-1).indices
        xy=torch.stack(((ids%384+.5)/384,(ids//384+.5)/384),-1)
        dense=output.geometry_context
        if dense is None or dense.shape[1]!=784:raise ValueError('Original dense geometry context required')
        context=F.grid_sample(dense.transpose(1,2).reshape(len(dense),-1,28,28),
                              (xy*2-1)[:,None],align_corners=False,padding_mode='border')[:,:,0].transpose(1,2)
        return GoalField(belief_logits=logits,candidate_ids=ids,candidate_xy=xy,candidate_context=context,
                         visual_coverage=output.observed_mask.flatten(1).float())

class V3NavigationPolicy:
    """Unchanged original Trajectory/Stop applied to V3's aligned dense GoalField."""
    def __init__(self,source_v2,checkpoint,stage1,directory,*,geometry_only=False):
        from .trajectory_policy import CandidateTrajectoryStop
        self.belief=V3CausalBelief(source_v2,checkpoint,directory,skip_rgb_encoder=geometry_only)
        self.device=self.belief.device;self.geometry_only=geometry_only
        source=torch.load(source_v2,map_location='cpu',weights_only=True)
        stage=torch.load(stage1,map_location='cpu',weights_only=True)
        if source['config']['source']['stage1_sha256']!=file_sha256(stage1):
            raise ValueError('Original frozen controller/source provenance mismatch')
        if stage['protocol']['name']!='graphnav-v1-stage1' or stage['protocol'].get('active_observation'):
            raise ValueError('Expected original non-active controller protocol')
        weights={k[len('trajectory.'):]:v for k,v in stage['model'].items() if k.startswith('trajectory.')}
        self.head=CandidateTrajectoryStop(hidden=self.belief.cfg.hidden).to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict(weights,strict=True)
        self.stop_threshold=.3;self.stop_positive_weight=20.;self.execution='guarded-blend';self.explore_candidates=0
        self.record=None;self.visited_goal_cells=set()

    def set_episode(self,episode,transform,catalog):
        bounds=[transform.x_min,transform.y_min,transform.x_max,transform.y_max]
        # Ignore annotation-linked reference IDs; blind public-name binding is identical to V3 TRAIN.
        objects=catalog.objects[episode.map_name]
        self.belief.start(map_name=episode.map_name,instruction=episode.instruction,world_bounds=bounds,
                          public_objects={str(k):v for k,v in objects.items()})
        self.record=self.belief.record;self.visited_goal_cells=set()

    @torch.no_grad()
    def predict(self,state,instruction):
        from .a3_runtime import A3TrajectoryPolicy
        if instruction!=self.record['instruction']:raise ValueError('Instruction changed within episode')
        output=self.belief.observe(observation=state.current_observation,pose=state.pose)
        field=self.belief.controller_field(output,geometry_only=self.geometry_only)
        prediction=A3TrajectoryPolicy.predict_field(self,state,field)
        prediction.diagnostics.update(policy='UnifiedPGVB-V3 + unchanged original Trajectory/Stop',
            geometry_only=self.geometry_only,observed_cells=int(output.observed_mask.sum()),
            causal_history_frames=len(self.belief.frames),raw_yaw=float(state.pose[3]),
            goal_top1_world=(self.belief.bounds[0,:2]+field.candidate_xy[0,0]*(self.belief.bounds[0,2:]-self.belief.bounds[0,:2])).cpu().tolist(),
            visual_delta_max=float((output.fused_logits-output.geometry_global).abs().max()))
        return prediction
