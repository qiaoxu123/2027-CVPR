"""Causal current-pixel evidence and residual goal belief in the shared environment."""
from types import SimpleNamespace
from collections import OrderedDict
import hashlib
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
        poses=np.asarray(poses,dtype=np.float64)
        if poses.ndim!=2 or poses.shape[1]!=4 or not len(poses) or not np.isfinite(poses).all():
            raise ValueError('poses must be a finite nonempty Nx4 history')
        # Source raster stays environment-private; policy receives one visited crop.
        # The dynamic interface never consumes SBF's 9-channel canvas: avoid
        # rebuilding its entire explored footprint and semantic masks per step.
        observation=self.renderer.current_rgbd(episode.map_name,poses[-1],GROUND_LEVEL[episode.map_name])
        observation.pop('surface_height_m',None)
        return SimpleNamespace(pose=poses[-1].copy(),current_observation=observation,step=len(poses)-1)


def contour_matches(first,second):
    first=np.asarray(first);second=np.asarray(second)
    if len(first)>1 and np.linalg.norm(first[-1]-first[0])<1e-4:first=first[:-1]
    if len(second)>1 and np.linalg.norm(second[-1]-second[0])<1e-4:second=second[:-1]
    return first.shape==second.shape and np.allclose(first,second,atol=.02,rtol=0)


def observed_grid_cells(world,observation):
    """Actual pixel support only; outside/black pixels stay unknown, not negative."""
    from .landmark_projection import world_to_frame
    bgr=observation['bgr'];h,w=bgr.shape[:2]
    pixels=world_to_frame(world,observation['footprint_world'],bgr.shape)
    inside=np.isfinite(pixels).all(-1)&(pixels[:,0]>=0)&(pixels[:,0]<=w-1)&(pixels[:,1]>=0)&(pixels[:,1]<=h-1)
    supported=np.zeros(len(world),dtype=bool)
    ij=np.rint(pixels[inside]).astype(int)
    supported[inside]=(bgr[ij[:,1],ij[:,0]].max(-1)>0)
    return supported


def graph_control_inputs(inputs, control):
    """Inference-only mechanism ablation, preserving the original A3 inputs."""
    if control == 'full':return inputs
    if control == 'self':
        valid=inputs['node_valid'].bool();n=valid.shape[1]
        adjacency=torch.eye(n,device=valid.device,dtype=torch.bool)[None]&valid[:,:,None]&valid[:,None,:]
        return dict(inputs,adjacency=adjacency)
    if control == 'zero-edge-features':return dict(inputs,edges=torch.zeros_like(inputs['edges']))
    raise ValueError('unknown graph mechanism control')


def visual_control_vectors(rois,vectors,control):
    if control=='full':return vectors
    if control=='zero-vectors':return torch.zeros_like(vectors)
    if control=='same-type-shuffle':
        result=vectors.clone();groups={}
        for i,roi in enumerate(rois):groups.setdefault(roi.object_type,[]).append(i)
        for indices in groups.values():
            if len(indices)>1:result[indices]=vectors[indices[-1:]+indices[:-1]]
        return result
    raise ValueError('unknown visual mechanism control')


class DynamicWaypointPolicy(A3WaypointPolicy):
    def __init__(self,checkpoint,dynamic_checkpoint,visual_checkpoint,directory,*,device='cuda',history=False,visual=True,observed_cache=None,head_checkpoint=None,stop_threshold=.3,execution='guarded-blend',explore_candidates=0,graph_control='full',visual_control='full'):
        super().__init__(checkpoint,directory,device=device)
        from .dynamic_belief import DynamicGoalBelief
        from .checkpointing import file_sha256,resolve_checkpoint_resource
        state=torch.load(dynamic_checkpoint,map_location='cpu',weights_only=True)
        if state['input_fingerprint']['checkpoint']['sha256']!=file_sha256(checkpoint):
            raise ValueError('dynamic and A3 checkpoints differ')
        if not state['args']['associate_observed_refs']:raise ValueError('actual visual node association checkpoint required')
        vector_root=state['args']['train_vectors']
        import json
        from pathlib import Path
        manifest_path=resolve_checkpoint_resource(vector_root,dynamic_checkpoint)/'manifest.json'
        if file_sha256(manifest_path)!=state['input_fingerprint']['train_vectors']['sha256']:
            raise ValueError('dynamic training visual vector manifest changed')
        manifest=json.loads(manifest_path.read_text())
        if manifest['checkpoint_sha256']!=file_sha256(visual_checkpoint):
            raise ValueError('dynamic and visual encoder checkpoints differ')
        self.dynamic=DynamicGoalBelief(self.model,vision_dim=256).to(self.device).eval()
        self.dynamic.load_state_dict(state['model'],strict=True)
        self.use_graph=state['args']['variant']!='visual'
        self.use_visual=visual and state['args']['variant']!='graph'
        self.use_history=history
        if graph_control not in ('full','self','zero-edge-features'):raise ValueError('unknown graph mechanism control')
        self.graph_control=graph_control
        if visual_control not in ('full','zero-vectors','same-type-shuffle'):raise ValueError('unknown visual mechanism control')
        if visual_control!='full' and not self.use_visual:raise ValueError('visual control requires observed visual branch')
        self.visual_control=visual_control
        # Pure public map geometry, independent of query, goal or observation history.
        self.public_geometry_cache=OrderedDict()
        self.encoder_visual=None
        if self.use_visual:
            from .observed_landmarks import ObservedLandmarkEncoder
            self.encoder_visual=ObservedLandmarkEncoder(visual_checkpoint,observed_cache or directory+'/observed_vectors',device=device)
        self.has_trajectory=head_checkpoint is not None
        if self.has_trajectory:
            from .a3_runtime import A3TrajectoryPolicy
            A3TrajectoryPolicy.load_trajectory_head(self,checkpoint,head_checkpoint,stop_threshold=stop_threshold,execution=execution,explore_candidates=explore_candidates)

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        self.landmarks=[]
        from scripts.prepare_landmark_views import objects_by_map
        from .observed_landmarks import LandmarkSpatialIndex
        self.landmarks=LandmarkSpatialIndex(objects_by_map({episode.map_name:catalog.objects[episode.map_name]})[episode.map_name])
        self.evidence={};self.geometry={};self.pose_history=[];self.observed_cells=None
        self.static_dynamic_field=None
        if self.encoder_visual:self.encoder_visual.start_episode()

    @torch.no_grad()
    def predict(self,state,instruction):
        base,inputs=self.frozen_field()
        dynamic_inputs=graph_control_inputs(inputs,getattr(self,'graph_control','full'))
        if not self.use_visual:
            if self.static_dynamic_field is None:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                    self.static_dynamic_field=self.dynamic(dynamic_inputs,tokens=None,token_mask=None,node_emb=self.public_node_emb,
                        base_field=base,use_graph=self.use_graph,use_visual=False)
            return self.execute_field(state,self.static_dynamic_field,inputs,dict(observed_nodes=0,auxiliary_nodes=0,observed_regions=0))
        b,n=inputs['entity_valid'].shape;q=len(inputs['grid_xy'][0]);dim=256
        ref=torch.zeros(b,n,dim,device=self.device);rc=torch.zeros(b,n,device=self.device)
        aux=torch.zeros(b,32,dim,device=self.device);ac=torch.zeros(b,32,device=self.device)
        ag=torch.zeros(b,q,32,10,device=self.device)
        current={}
        if self.use_visual:
            rois,vectors=self.encoder_visual.observe(state.current_observation,self.landmarks,step=state.step)
            vectors=visual_control_vectors(rois,vectors,getattr(self,'visual_control','full'))
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
        if self.observed_cells is None:self.observed_cells=np.zeros(q,dtype=bool)
        self.observed_cells|=observed_grid_cells(world,state.current_observation)
        selected=sorted(groups,key=lambda k:(-groups[k]['confidence']*groups[k]['pixels'],k))[:32]
        extra=0
        for key in selected:
            item=groups[key];roi=item['roi'];vector=(item['vector']/max(item['mass'],1e-8)).to(self.device)
            matches=[i for i,entity in enumerate(self.public_entities) if entity['name']==roi.name and contour_matches(entity['geometry_m'],roi.contour_world)]
            if matches:
                node=matches[-1];ref[0,node]=vector;rc[0,node]=item['confidence']
            else:
                if key not in self.geometry:
                    if not hasattr(self,'public_geometry_cache'):self.public_geometry_cache=OrderedDict()
                    public_key=hashlib.sha256(world.astype(np.float32).tobytes()+roi.contour_world.astype(np.float32).tobytes()).hexdigest()
                    if public_key not in self.public_geometry_cache:
                        self.public_geometry_cache[public_key]=relative_entity_features(world,[dict(geometry_m=roi.contour_world,kind='surface')],1)[:,0].astype(np.float16)
                    self.public_geometry_cache.move_to_end(public_key)
                    self.geometry[key]=torch.as_tensor(self.public_geometry_cache[public_key],device=self.device)
                    if len(self.public_geometry_cache)>4096:self.public_geometry_cache.popitem(last=False)
                aux[0,extra]=vector;ac[0,extra]=item['confidence'];ag[0,:,extra]=self.geometry[key].half().float();extra+=1
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            field=self.dynamic(dynamic_inputs,tokens=None,token_mask=None,node_emb=self.public_node_emb,
                base_field=base,use_graph=self.use_graph,use_visual=self.use_visual,
                visual_vectors=ref,visual_confidence=rc,observed_vectors=aux,observed_confidence=ac,observed_geometry=ag)
        return self.execute_field(state,field,inputs,dict(observed_nodes=int((rc>0).sum()),auxiliary_nodes=extra,observed_regions=len(groups),
            observed_candidate_fraction=float(self.observed_cells.mean())))

    def execute_field(self,state,field,inputs,evidence_diagnostics):
        bounds=np.asarray(self.record['world_bounds'])
        top=int(field.belief_logits[0].argmax());uv=inputs['grid_xy'][0,top].float().cpu().numpy()
        goal=bounds[:2]+uv*(bounds[2:]-bounds[:2]);self.pose_history.append(np.asarray(state.pose).tolist())
        diagnostics=dict(policy='A3 residual belief + bounded waypoint',graph=self.use_graph,visual=self.use_visual,
                history=self.use_history,graph_control=getattr(self,'graph_control','full'),visual_control=getattr(self,'visual_control','full'),**evidence_diagnostics,goal_xy=goal.tolist(),candidate=top,
                graph_gate=float(self.dynamic.graph_gate.tanh()),visual_gate=float(self.dynamic.visual_gate.tanh()))
        if getattr(self,'has_trajectory',False):
            from .a3_runtime import A3TrajectoryPolicy
            prediction=A3TrajectoryPolicy.predict_field(self,state,field)
            prediction.diagnostics.update({k:v for k,v in diagnostics.items() if k not in ('candidate','goal_xy')},
                belief_argmax_cell=top,belief_goal_xy=goal.tolist(),trajectory_execution=self.execution)
            return prediction
        return SimpleNamespace(xyz=np.array([*goal,state.pose[2]]),should_stop=False,diagnostics=diagnostics)
