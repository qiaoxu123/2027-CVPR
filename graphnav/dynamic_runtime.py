"""Causal current-pixel evidence and residual goal belief in the shared environment."""
from types import SimpleNamespace
import numpy as np
import torch
from .a3_runtime import A3WaypointPolicy
from .candidate_geometry import relative_entity_features


class NativeObservationBuilder:
    """Environment boundary: render only the actual current pose, never teacher poses."""
    def __init__(self,builder,rgbd,directory):
        from .renderer import CityNavObservedViewRenderer
        self.builder=builder
        self.renderer=CityNavObservedViewRenderer(rgbd,directory,image_size=384)

    def transform(self,map_name):return self.builder.transform(map_name)

    def build_from_poses(self,episode,poses):
        from sbfnav.mapdata import GROUND_LEVEL
        state=self.builder.build_from_poses(episode,poses)
        # Source raster stays environment-private; policy receives one visited crop.
        observation=self.renderer.current_rgbd(episode.map_name,poses[-1],GROUND_LEVEL[episode.map_name])
        observation.pop('surface_height_m',None)
        return SimpleNamespace(pose=state.pose,current_observation=observation,step=len(poses)-1)


def contour_matches(first,second):
    first=np.asarray(first);second=np.asarray(second)
    if len(first)>1 and np.linalg.norm(first[-1]-first[0])<1e-4:first=first[:-1]
    if len(second)>1 and np.linalg.norm(second[-1]-second[0])<1e-4:second=second[:-1]
    return first.shape==second.shape and np.allclose(first,second,atol=.02,rtol=0)


class DynamicWaypointPolicy(A3WaypointPolicy):
    def __init__(self,checkpoint,dynamic_checkpoint,visual_checkpoint,directory,*,device='cuda',history=False,visual=True,observed_cache=None):
        super().__init__(checkpoint,directory,device=device)
        from .dynamic_belief import DynamicGoalBelief
        from .checkpointing import file_sha256
        state=torch.load(dynamic_checkpoint,map_location='cpu',weights_only=True)
        if state['input_fingerprint']['checkpoint']['sha256']!=file_sha256(checkpoint):
            raise ValueError('dynamic and A3 checkpoints differ')
        if not state['args']['associate_observed_refs']:raise ValueError('actual visual node association checkpoint required')
        vector_root=state['args']['train_vectors']
        import json
        manifest=json.loads((__import__('pathlib').Path(vector_root)/'manifest.json').read_text())
        if manifest['checkpoint_sha256']!=file_sha256(visual_checkpoint):
            raise ValueError('dynamic and visual encoder checkpoints differ')
        self.dynamic=DynamicGoalBelief(self.model,vision_dim=256).to(self.device).eval()
        self.dynamic.load_state_dict(state['model'],strict=True)
        self.use_graph=state['args']['variant']!='visual'
        self.use_visual=visual and state['args']['variant']!='graph'
        self.use_history=history
        self.encoder_visual=None
        if self.use_visual:
            from .observed_landmarks import ObservedLandmarkEncoder
            self.encoder_visual=ObservedLandmarkEncoder(visual_checkpoint,observed_cache or directory+'/observed_vectors',device=device)

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        self.landmarks=[]
        from scripts.prepare_landmark_views import objects_by_map
        self.landmarks=objects_by_map({episode.map_name:catalog.objects[episode.map_name]})[episode.map_name]
        self.evidence={};self.geometry={};self.pose_history=[]
        if self.encoder_visual:self.encoder_visual.start_episode()

    @torch.no_grad()
    def predict(self,state,instruction):
        base,inputs=self.frozen_field()
        b,n=inputs['entity_valid'].shape;q=len(inputs['grid_xy'][0]);dim=256
        ref=torch.zeros(b,n,dim,device=self.device);rc=torch.zeros(b,n,device=self.device)
        aux=torch.zeros(b,32,dim,device=self.device);ac=torch.zeros(b,32,device=self.device)
        ag=torch.zeros(b,q,32,10,device=self.device)
        current={}
        if self.use_visual:
            rois,vectors=self.encoder_visual.observe(state.current_observation,self.landmarks,step=state.step)
            for roi,vector in zip(rois,vectors):
                key=roi.object_id
                current[key]=dict(vector=vector.float()*roi.confidence,mass=roi.confidence,
                    confidence=roi.confidence,pixels=roi.quality['visible_pixels'],roi=roi)
                if key in self.evidence:
                    old=self.evidence[key];old['vector']+=vector.float()*roi.confidence;old['mass']+=roi.confidence
                    old['confidence']=max(old['confidence'],roi.confidence);old['pixels']=max(old['pixels'],roi.quality['visible_pixels'])
                else:self.evidence[key]={**current[key],'vector':current[key]['vector'].clone()}
        groups=self.evidence if self.use_history else current
        bounds=np.asarray(self.record['world_bounds']);world=inputs['grid_xy'][0].float().cpu().numpy()*(bounds[2:]-bounds[:2])+bounds[:2]
        selected=sorted(groups,key=lambda k:(-groups[k]['confidence']*groups[k]['pixels'],k))[:32]
        extra=0
        for key in selected:
            item=groups[key];roi=item['roi'];vector=(item['vector']/max(item['mass'],1e-8)).to(self.device)
            matches=[i for i,entity in enumerate(self.public_entities) if entity['name']==roi.name and contour_matches(entity['geometry_m'],roi.contour_world)]
            if matches:
                for node in matches:ref[0,node]=vector;rc[0,node]=item['confidence']
            else:
                if key not in self.geometry:self.geometry[key]=torch.as_tensor(relative_entity_features(world,[dict(geometry_m=roi.contour_world,kind='surface')],1)[:,0],device=self.device)
                aux[0,extra]=vector;ac[0,extra]=item['confidence'];ag[0,:,extra]=self.geometry[key];extra+=1
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            field=self.dynamic(inputs,tokens=None,token_mask=None,node_emb=self.public_node_emb,
                base_field=base,use_graph=self.use_graph,use_visual=self.use_visual,
                visual_vectors=ref,visual_confidence=rc,observed_vectors=aux,observed_confidence=ac,observed_geometry=ag)
        top=int(field.belief_logits[0].argmax());uv=inputs['grid_xy'][0,top].float().cpu().numpy()
        goal=bounds[:2]+uv*(bounds[2:]-bounds[:2]);self.pose_history.append(np.asarray(state.pose).tolist())
        return SimpleNamespace(xyz=np.array([*goal,state.pose[2]]),should_stop=False,
            diagnostics=dict(policy='A3 residual belief + bounded waypoint',graph=self.use_graph,visual=self.use_visual,
                history=self.use_history,observed_nodes=int((rc>0).sum()),auxiliary_nodes=extra,
                observed_regions=len(groups),goal_xy=goal.tolist(),candidate=top,
                graph_gate=float(self.dynamic.graph_gate.tanh()),visual_gate=float(self.dynamic.visual_gate.tanh())))
