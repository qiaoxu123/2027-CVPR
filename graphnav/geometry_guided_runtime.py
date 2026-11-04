"""Opt-in stage1 landmark regression; frozen geometry, learned observed likelihood."""
import numpy as np
import shapely
import torch
from .stage1_runtime import GraphNavStage1Policy
from .a3_runtime import A3TrajectoryPolicy
from .candidate_geometry import FullCandidateDataset,collate_full
from .full_experiment import _batch_inputs
from .geometry_guided_visual import LandmarkVisualBelief,visual_world_beliefs
from .landmark_alias import resolve_references
from .landmark_region_supervision import named_union,public_type
from .native_dense_regions import native_dense_regions,sparse_view
from .stage1 import preserved_proposal_order
from .full_localization import GoalField


def fused_controller_field(base,grid_uv,energy,projection_supported):
    # Retain the original head's logit gauge: do not feed normalized geometry
    # log probabilities to a head trained on original raw A3 scores.
    if not projection_supported or not bool(energy.any()):return base
    logits=base.belief_logits.float()+energy
    ids=preserved_proposal_order(logits,base.candidate_ids)
    dense=base.dense_context
    return GoalField(logits,ids,grid_uv.gather(1,ids[...,None].expand(-1,-1,2)),
        dense.gather(1,ids[...,None].expand(-1,-1,dense.shape[-1])),dense_context=dense)


class GeometryGuidedLandmarkPolicy(GraphNavStage1Policy):
    def __init__(self,checkpoint,visual_checkpoint,directory):
        super().__init__(checkpoint,directory)
        state=torch.load(visual_checkpoint,weights_only=True,map_location='cpu')
        self.learned_visual=LandmarkVisualBelief().to(self.device).eval().requires_grad_(False)
        self.learned_visual.load_state_dict(state['model'])

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        entities=resolve_references(episode.reference_names,catalog.objects[episode.map_name],max_landmarks=self.args.max_landmarks,match_mode='fuzzy')
        self.region=named_union([e['contour'] for e in entities])
        self.public_type=public_type([e['object_type'] for e in entities])
        phrases={'Building':'an aerial photograph of a building','Road':'an aerial photograph of a road',
            'Ground':'an aerial photograph of a grassy ground area','Parking':'an aerial photograph of a parking lot'}
        self.name_text=self.native.texts([episode.instruction]).detach().clone()
        self.type_text=self.native.texts([phrases[self.public_type]]).detach().clone() if self.public_type else None

    @torch.no_grad()
    def predict(self,state,instruction,*,use_visual=True):
        from scripts.run_native_grid_pilot import pool
        import cv2
        if instruction!=self.record['instruction']:raise ValueError('Query mismatch')
        if self.inputs is None:
            import json
            path=self.directory/'current_public_query.jsonl';path.write_text(json.dumps(self.record)+'\n')
            ds=FullCandidateDataset(path,grid_size=self.args.grid_size,max_landmarks=self.args.max_landmarks,max_roads=self.args.max_roads,
                max_relations=self.args.max_relations,prior_sigma_m=self.args.prior_sigma_m,rule_sigma_m=self.args.rule_sigma_m)
            self.inputs,tokens,mask,names=_batch_inputs(collate_full([ds[0]]),self.bert,self.device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):self.base=self.model.a3(self.inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            self.bounds=torch.tensor(self.record['world_bounds'],device=self.device,dtype=torch.float32)[None]
        field=self.base;support=False;energy_range=0.;raw_goal=None
        if self.type_text is not None:
            obs=state.current_observation
            sparse=sparse_view(native_dense_regions(dict(bgr=obs['bgr'],depth_valid=obs['depth_valid'],footprint_world=obs['footprint_world'])))
            pixels=self.native.processor(images=[cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB)],return_tensors='pt').to(self.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):patches=self.native.model.vision_model(**pixels).last_hidden_state[0]
            features=pool(self.native,patches,sparse['masks'])[None].to(self.device)
            bounds=self.bounds[0].cpu().numpy();uv=torch.from_numpy((sparse['world']-bounds[:2])/(bounds[2:]-bounds[:2]))[None].to(self.device)
            valid=torch.from_numpy(sparse['valid'])[None].to(self.device)&(uv>=0).all(-1)&(uv<=1).all(-1)
            inside=shapely.distance(shapely.points(sparse['world']),self.region)<=.001
            support=bool((torch.from_numpy(inside).to(self.device)[None]&valid).any())
            score=self.learned_visual(image_features=features,name_text=self.name_text,type_text=self.type_text,valid=valid)
            evidence=visual_world_beliefs(self.base.belief_logits,uv,score,valid,torch.tensor([support],device=self.device))
            energy_range=float(evidence['energy'].max()-evidence['energy'].min())
            grid_world=self.bounds[:,None,:2]+self.inputs['grid_xy']*(self.bounds[:,None,2:]-self.bounds[:,None,:2])
            if evidence['has_view'][0]:raw_goal=grid_world[0,evidence['raw'][0].argmax()].cpu().tolist()
            if use_visual:field=fused_controller_field(self.base,self.inputs['grid_xy'],evidence['energy'],support)
        prediction=A3TrajectoryPolicy.predict_field(self,state,field)
        prediction.diagnostics.update(geometry_frozen=True,learned_visual_enabled=use_visual,
            public_projection_support=support,public_type=self.public_type,raw_visual_goal_xy=raw_goal,visual_energy_range=energy_range,
            fallback_reason='mixed_public_type' if self.type_text is None else 'no_projection_support' if not support else None)
        return prediction
