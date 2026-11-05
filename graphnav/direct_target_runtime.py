"""Trained direct-target adapter; unchanged StableGoal and Trajectory/Stop."""
import json
import time
from pathlib import Path
import torch
from .unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy
from .direct_target_data import load_model, relation_geometry
from .a3_runtime import public_static_record, A3TrajectoryPolicy
from .pgvb_observation import encode_landmark_observation
from .local_landmark_anchor import named_public_polygons
from .checkpointing import file_sha256


class UnifiedPGVBDirectTargetPolicy(UnifiedPGVBNamedLandmarkPolicy):
    def __init__(self, stage1, original_v2, trained, directory, *, method='RGB'):
        super().__init__(stage1, original_v2, directory, mode='tracked-stable')
        payload=torch.load(trained,map_location='cpu',weights_only=True)
        if payload.get('format')!='unified-pgvb-direct-target-v1' or payload.get('direct_target_ready') is not True:
            raise ValueError('A fully trained target checkpoint is required')
        if payload['training_config']['source_sha256']!=file_sha256(original_v2):
            raise ValueError('Target model and original integrated source differ')
        self.unified=load_model(payload).eval().requires_grad_(False)
        if method not in ('Geometry','Relation','RGB'):raise ValueError('Unknown direct-target ablation')
        self.method=method

    def set_direct_episode(self, episode, transform, catalog, public_task):
        if public_task['instruction']!=episode.instruction or public_task['map_name']!=episode.map_name:
            raise ValueError('Language-conditioned input mismatch')
        if public_task.get('target_GT_used') is not False:
            raise ValueError('Only public GT-free task specification is allowed')
        # Task whitelist, not a private dataset row or target destination.
        if set(public_task)-{'index','episode','instruction_index','map_name','instruction',
                            'anchors','structured','target_GT_used','world_bounds','start_pose'}:
            raise ValueError('Unexpected metadata in online task')
        self.direct_spec=public_task
        self.record=public_static_record(episode,transform,catalog,max_landmarks=self.args.max_landmarks)
        self.inputs=None;self.visited_goal_cells=set();self.acquisition.reset();self.stable_goal.reset();self.unified.reset_episode()
        a=public_task['anchors'][0]
        self._landmark=dict(name=a['landmark']['name'],typ=a['landmark']['type'],contours=a['contours'])
        self.instruction_spec=None

    @torch.no_grad()
    def predict(self,state,instruction):
        if instruction!=self.record['instruction']:raise ValueError('Instruction changed')
        self._initialize_geometry_inputs()
        start=time.perf_counter();observed=None;meta=None;tracking=None;reason='ablation_no_RGB'
        if self.method=='RGB':
            observed,reason,meta=encode_landmark_observation(state.current_observation,self._landmark,
                self.record,self.native,self.device,return_metadata=True,allow_offscreen_anchor=True)
        geometry=relation_geometry(self.direct_spec,self.inputs['grid_xy'][0].cpu().numpy())[None].to(self.device)
        ids=torch.tensor([[a['predicate_id'] for a in self.direct_spec['anchors']]],device=self.device)
        category=self.direct_spec['structured']['target']['category']
        semantic=' '.join([category]+[a['value'] for a in self.direct_spec['structured']['target']['attributes']])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=self.unified.forward_direct_target(static_inputs=self.inputs,tokens=self.tokens,
                token_mask=self.mask,node_emb=self.names,relation_geometry=geometry,relation_ids=ids,
                relation_valid=torch.ones_like(ids,dtype=torch.bool),observed=observed,
                target_text=self.native.texts([semantic]))
            if observed is not None:
                kwargs={k:v for k,v in observed.items() if k!='local_uv'}
                scores=self.unified.pgvb(**kwargs)
                shifted=dict(kwargs,image_features=kwargs['image_features'].roll(shifts=scores.shape[1]//2,dims=1))
                counter=self.unified.pgvb(**shifted)
                tracking=self.acquisition.observe(scores.float(),counter.float(),meta['anchor_labels'],kwargs['valid'])
        key={'Geometry':'geometry_field','Relation':'relation_field','RGB':'field'}[self.method]
        field=result[key]
        active=bool(self.method=='RGB' and observed is not None and result['coverage'].any())
        field,goal_audit=self.stable_goal.select(field,self.inputs['grid_xy'],tracking=tracking,visual_active=active)
        prediction=A3TrajectoryPolicy.predict_field(self,state,field)
        torch.cuda.synchronize()
        prediction.diagnostics.update(method=self.method,RGB_reason=reason,visual_active=active,
            tracking=tracking,stable_goal=goal_audit,decision_seconds=time.perf_counter()-start,
            geometry_top1=int(result['geometry_field'].belief_logits.argmax()),
            relation_top1=int(result['relation_field'].belief_logits.argmax()),
            full_top1=int(result['field'].belief_logits.argmax()),
            belief_top1=int(field.belief_logits.argmax()))
        if getattr(self,'direct_audit_callback',None):
            self.direct_audit_callback(state,result,field)
        return prediction
