"""Opt-in frozen local-position runtime using unchanged original Trajectory/Stop.

Only PublicEpisode, current observations and public text/map bindings are read.
No raster access, teacher trajectory, goal labels or target-instance identifiers.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .a3_runtime import A3TrajectoryPolicy
from .candidate_geometry import FullCandidateDataset,collate_full,relative_entity_features
from .full_experiment import _batch_inputs
from .full_localization import GoalField
from .local_chain_observation import local_patch_regions
from .local_chain_position import LocalChainPositionHead
from .observed_patch_scales import masked_native_pool
from .stage1 import preserved_proposal_order
from .stage1_runtime import GraphNavStage1Policy
from scripts.analyze_instruction_landmark_association import PublicNames


def local_goal_field(base,global_xy,local_uv,logits,local_valid):
    if len(logits)!=1:raise ValueError('online field packing expects one episode')
    if not local_valid.any():return base
    valid=local_valid[0].nonzero().flatten()
    cells=base.belief_logits.shape[-1];grid=math.isqrt(cells)
    if grid*grid!=cells:raise ValueError('square static field required')
    uv=local_uv[:,valid].float()
    packed=torch.cat((logits[:,:cells],logits[:,cells+valid]),-1)
    xy=torch.cat((global_xy.float(),uv),1)
    context=base.dense_context.float()
    h=context.shape[-1]
    local_context=F.grid_sample(context.transpose(1,2).reshape(1,h,grid,grid),
                                (uv*2-1)[:,None],align_corners=False,padding_mode='border').squeeze(2).transpose(1,2)
    dense=torch.cat((context,local_context),1)
    ids=preserved_proposal_order(packed,base.candidate_ids)
    return GoalField(packed,ids,xy.gather(1,ids[...,None].expand(-1,-1,2)),
                     dense.gather(1,ids[...,None].expand(-1,-1,h)),dense_context=dense,
                     visual_coverage=torch.cat((torch.zeros_like(base.belief_logits.float()),torch.ones(1,len(valid),device=logits.device)),1))


class LocalChainPolicy(GraphNavStage1Policy):
    def __init__(self,checkpoint,local_checkpoint,public_roles,directory,*,device='cuda',control='real'):
        super().__init__(checkpoint,directory,device=device)
        if control not in ('real','zero','shuffle','no_geometry','full_text','a3'):
            raise ValueError('unknown frozen local intervention')
        state=torch.load(local_checkpoint,map_location='cpu',weights_only=True)
        self.position=LocalChainPositionHead().to(device).eval().requires_grad_(False)
        self.position.load_state_dict(state['model'],strict=True)
        roles=json.loads(Path(public_roles).read_text())
        self.roles={(r['map_name'],r['instruction']):r for r in roles}
        self.control=control

    def set_episode(self,episode,transform,catalog):
        super().set_episode(episode,transform,catalog)
        mapped=PublicNames(catalog.objects).match(episode.map_name,episode.instruction)
        ids=list(dict.fromkeys(oid for m in mapped for oid in m['public_ids']))[:16]
        self.public_entities=[dict(kind='region',geometry_m=np.asarray(catalog.objects[episode.map_name][int(oid)]['contour'],np.float32)) for oid in ids]
        role=self.roles.get((episode.map_name,episode.instruction))
        self.role_fallback=role is None
        self.target_phrase=role['target_text'] if role else episode.instruction
        self.context_phrase=role['context_text'] if role else episode.instruction

    @torch.no_grad()
    def predict(self,state,instruction):
        if instruction!=self.record['instruction']:raise ValueError('language changed within episode')
        if self.inputs is None:
            path=self.directory/'current_public_query.jsonl';path.write_text(json.dumps(self.record)+'\n')
            ds=FullCandidateDataset(path,grid_size=self.args.grid_size,max_landmarks=self.args.max_landmarks,
                max_roads=self.args.max_roads,max_relations=self.args.max_relations,
                prior_sigma_m=self.args.prior_sigma_m,rule_sigma_m=self.args.rule_sigma_m)
            self.inputs,self.tokens,self.mask,self.names=_batch_inputs(collate_full([ds[0]]),self.bert,self.device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                self.base=self.model.a3(self.inputs,variant='geometry',tokens=self.tokens,token_mask=self.mask,node_emb=self.names)
            self.bounds=np.array(self.record['world_bounds'],np.float32)
            self.target_vector=self.native.texts([self.target_phrase]).detach().clone()
            self.context_vector=self.native.texts([self.context_phrase]).detach().clone()
        if self.control=='a3':
            prediction=A3TrajectoryPolicy.predict_field(self,state,self.base)
            prediction.diagnostics.update(local_control='a3',local_hypotheses=0)
            return prediction
        observation=state.current_observation
        world,masks,valid,quality=local_patch_regions(observation)
        features=torch.zeros(144,3,1024,device=self.device)
        if valid.any():
            import cv2
            pixels=self.native.processor(images=[cv2.cvtColor(observation['bgr'],cv2.COLOR_BGR2RGB)],return_tensors='pt').to(self.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                patches=self.native.model.vision_model(**pixels).last_hidden_state[0]
                supported=masks.any(-1)
                pooled=masked_native_pool(self.native.model.vision_model.head,patches,torch.from_numpy(masks[supported]).to(self.device))
                features[torch.from_numpy(supported).to(self.device)]=F.normalize(pooled.float(),dim=-1)
        if self.control=='zero':features.zero_()
        if self.control=='shuffle':
            ids=torch.from_numpy(np.flatnonzero(valid)).to(self.device)
            features[ids]=features[ids.flip(0)].clone()
        geometry=np.zeros((144,31),np.float32)
        if self.public_entities:
            pair=relative_entity_features(world,self.public_entities,len(self.public_entities))
            geometry=np.concatenate((pair.mean(1),pair.min(1),pair.max(1),np.full((144,1),len(self.public_entities)/16,np.float32)),-1)
        if self.control=='no_geometry':geometry.fill(0)
        uv=torch.from_numpy((world-self.bounds[:2])/(self.bounds[2:]-self.bounds[:2])).to(self.device)[None]
        valid_tensor=torch.from_numpy(valid).to(self.device)[None]
        # Match cached training quantization, without changing model weights.
        logits=self.position(prior_logits=self.base.belief_logits.float(),local_uv=uv,
            image_features=features.half()[None],target_text=(self.context_vector if self.control=='full_text' else self.target_vector).half(),
            context_text=self.context_vector.half(),public_geometry=torch.from_numpy(geometry).half().to(self.device)[None],
            quality=torch.from_numpy(quality).to(self.device)[None],local_valid=valid_tensor)
        field=local_goal_field(self.base,self.inputs['grid_xy'],uv,logits,valid_tensor)
        prediction=A3TrajectoryPolicy.predict_field(self,state,field)
        prediction.diagnostics.update(policy='Frozen local-position hypotheses + unchanged original Trajectory/Stop',
            local_control=self.control,local_hypotheses=int(valid.sum()),role_fallback=self.role_fallback,
            camera_yaw_rad=observation.get('camera_yaw_rad'),agent_yaw_rad=float(state.pose[3]),
            posterior_hypotheses=int(field.belief_logits.shape[-1]),
            global_prior_top1_xy=(self.bounds[:2]+self.inputs['grid_xy'][0,self.base.belief_logits[0].argmax()].float().cpu().numpy()*(self.bounds[2:]-self.bounds[:2])).tolist(),
            selected_local=bool(field.candidate_ids[0,0]>=self.base.belief_logits.shape[-1]),
            active_observation=False)
        return prediction
