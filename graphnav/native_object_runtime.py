"""Causal AerialVG observation encoder and native-query navigation belief adapter."""
import argparse,sys
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
from torchvision.transforms import functional as VF
from .workspace_paths import WORKSPACE
from .object_proposals import box_valid_fraction,select_proposals
from .object_query_belief import ObjectQueryRanker
from .calibrated_visual_residual import corrected_field
from .checkpointing import file_sha256


class NativeObjectEncoder:
    def __init__(self,device='cuda'):
        repo=WORKSPACE/'third_party/AerialVG';sys.path.insert(0,str(repo))
        from model import build_model
        from model.AerialVG import ms_deform_attn
        from util.slconfig import SLConfig
        from util.utils import clean_state_dict
        class Fallback:
            @staticmethod
            def apply(value,shapes,start,locations,weights,step):
                return ms_deform_attn.multi_scale_deformable_attn_pytorch(value,shapes,locations,weights)
        ms_deform_attn.MultiScaleDeformableAttnFunction=Fallback
        args=SLConfig.fromfile(str(repo/'config/config_cfg.py'));args.device=str(device)
        args.text_encoder_type='/home/20261024_1/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
        self.model=build_model(args)
        with torch.serialization.safe_globals([argparse.Namespace]):state=torch.load(WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth',map_location='cpu',weights_only=True)
        self.model.load_state_dict(clean_state_dict(state['model']),strict=True);self.model.to(device).eval().requires_grad_(False);self.device=torch.device(device);self.captured={}
        def capture(module,inputs,output):
            if inputs[0].shape[1]==args.num_queries:
                self.captured=dict(features=inputs[0].detach(),text=inputs[1]['encoded_text'].detach(),text_mask=inputs[1]['text_token_mask'].detach(),native_logits=output.detach())
        self.model.class_embed[-1].register_forward_hook(capture)
        # Pair reasoning is applied after geometry-guided selection by the trained head.
        # Replacing relation scores with native scores preserves detector boxes/features.
        self.model.relation_transformer.forward=lambda features,td:self.model.class_embed[-1](features,td)

    @torch.inference_mode()
    @torch.autocast('cuda',enabled=False)
    def encode(self,observation,instruction):
        for key in ('bgr','depth_valid','footprint_world'):
            if key not in observation:raise ValueError('Actual current observation required: '+key)
        bgr=observation['bgr'];height,width=bgr.shape[:2]
        image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));image=VF.resize(image,800,max_size=1333)
        tensor=VF.normalize(VF.to_tensor(image),[.485,.456,.406],[.229,.224,.225]).to(self.device)
        self.captured.clear();prediction=self.model(tensor[None],captions=[instruction.lower().strip().rstrip('.')+'.'])
        captured=self.captured;raw=captured['features'][0];mask=captured['text_mask'][0]
        # Official ContrastiveEmbed pads logits to256 tokens; td mask has actual L.
        # Padding is already -inf, matching official Top-K over sigmoid scores.
        ids=captured['native_logits'][0].sigmoid().amax(-1).topk(15).indices
        # Match feature-cache precision: native Top15 FP32, remaining query features FP16.
        features=raw.half().float();features[ids]=raw[ids]
        boxes=prediction['pred_boxes'][0].float();corners=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
        homography=cv2.getPerspectiveTransform(corners,np.asarray(observation['footprint_world'],np.float32))
        centers=cv2.perspectiveTransform((boxes[:,:2].cpu().numpy()*np.array([width-1,height-1],np.float32))[None],homography)[0]
        quality=box_valid_fraction(boxes.cpu().numpy(),observation['depth_valid'])
        return dict(features=features,text=captured['text'][0].float(),text_mask=mask,boxes=boxes,world_centers=torch.tensor(centers,device=self.device),quality=torch.tensor(quality,device=self.device))


@torch.autocast('cuda',enabled=False)
def prepare_native_frame(encoded,grid_world,base_logits,observation,sigma=8.):
    """Same GT-free selection, features, kernel and support as offline preparation."""
    device=base_logits.device;features=encoded['features'];text=encoded['text'];mask=encoded['text_mask'];boxes=encoded['boxes'];centers=encoded['world_centers'];quality=encoded['quality']
    native=(features@text.T)[:,mask].sigmoid().amax(-1)
    nearest=torch.cat([torch.cdist(grid_world.float(),block.float()).argmin(0) for block in centers.split(64)],0)
    prior=base_logits[nearest];ids=select_proposals(boxes,native,quality,15,prior,True)
    count=len(ids);f=torch.zeros(15,256,device=device);g=torch.zeros(15,8,device=device);world=torch.zeros(15,2,device=device);valid=torch.zeros(15,dtype=torch.bool,device=device)
    f[:count]=features[ids].half().float();world[:count]=centers[ids];valid[:count]=True
    g[:count]=torch.stack(((prior[ids]-base_logits.amax()).clamp(-20,0)/5,torch.logit(native[ids].clamp(1e-5,1-1e-5))/5,quality[ids],boxes[ids,2],boxes[ids,3],boxes[ids,0],boxes[ids,1],boxes[ids,2]*boxes[ids,3]),-1)
    height,width=observation['depth_valid'].shape;pixels=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    homography=cv2.getPerspectiveTransform(np.asarray(observation['footprint_world'],np.float32),pixels)
    projected=cv2.perspectiveTransform(grid_world.float().cpu().numpy()[None],homography)[0]
    inside=((projected>=0)&(projected<=np.array([width-1,height-1]))).all(-1);rounded=np.clip(np.rint(projected).astype(int),[0,0],[width-1,height-1])
    observed=torch.tensor(inside&observation['depth_valid'][rounded[:,1],rounded[:,0]],device=device)
    kernel=torch.exp(-torch.cdist(grid_world.float(),world).square()/(2*sigma**2))*observed[:,None]*valid[None]
    return dict(features=f[None],text=text.half().float()[None],text_mask=mask[None],geometry=g[None],valid=valid[None],query_world=world[None],kernel=kernel[None],observed=observed[None],proposal_ids=ids)


def install_native_object_belief(policy,path,base_checkpoint,control='real',history_window=1):
    if control not in ('real','zero','shuffle','no_relation'):raise ValueError('Unknown native object control')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not (Path(path).parent/'complete.json').exists():raise ValueError('Native object training must complete before navigation evaluation')
    if state['kind']!='native-aerialvg-object-query-belief-v1' or state['config']['base_sha']!=file_sha256(base_checkpoint):raise ValueError('Native object checkpoint/base mismatch')
    native_sha=file_sha256(path)
    config=state['config'];ranker=ObjectQueryRanker(relation=config['relation']).to(policy.device);ranker.load_state_dict(state['model']);ranker.eval().requires_grad_(False)
    if control=='no_relation':ranker.relation=None
    encoder=NativeObjectEncoder(policy.device) if control!='zero' else None
    policy.model.add_module('native_object_ranker',ranker)
    # This is a replacement visual branch. Old SigLIP matcher inputs are unused under
    # no-belief, so avoid computing a second visual backbone in this policy instance.
    policy.native.model.to('cpu')
    policy.native.texts=lambda texts:torch.zeros(len(texts),1024,device=policy.device)
    def empty_visual(observations,worlds):
        return [(torch.zeros(len(w),3,1024,dtype=torch.float16),torch.zeros(len(w),3,dtype=torch.bool),torch.zeros(len(w),3,3)) for w in worlds]
    policy.visual.encode=empty_visual
    from .temporal_native_evidence import CausalResidualWindow
    memory=CausalResidualWindow(history_window)
    original_forward=policy.model.forward;original_predict=policy.predict;context={}
    def forward(**inputs):
        inputs=dict(inputs);inputs['visual_control']='no-belief';base=original_forward(**inputs)
        if control=='zero':return base
        static=base.static;bounds=inputs['bounds'][0];world=bounds[:2]+inputs['static_inputs']['grid_xy'][0]*(bounds[2:]-bounds[:2])
        frame=prepare_native_frame(context['encoded'],world,static.belief_logits[0].float(),context['observation'],sigma=config.get('kernel_sigma',20.))
        if control=='shuffle':
            ids=frame['valid'][0].nonzero().flatten();frame['features'][0,ids]=frame['features'][0,ids.flip(0)].clone()
            logits=(frame['features'][0]@frame['text'][0].T).masked_fill(~frame['text_mask'][0][None],-1e4).amax(-1)
            frame['geometry'][0,:,1]=logits/5
        q,null=ranker(frame['features'],frame['text'],frame['text_mask'],frame['geometry'],frame['valid'])
        sink=getattr(policy,'native_collection_sink',None)
        if sink is not None:
            # Public observation tensors only. Labels are attached after rollout.
            sink(dict(frame=frame,base_logits=static.belief_logits,grid_world=world,
                      query_logits=q,null_logits=null,pose=context['pose'],step=context['step']))
        _,delta=ranker.update_belief(static.belief_logits,q,null,frame['kernel'],frame['observed'],frame['valid'],bound=config['bound'])
        delta=memory.update(delta,frame['observed'])
        base.delta=delta;base.belief=corrected_field(static,delta,inputs['static_inputs']['grid_xy']);context['delta_max']=float(delta.abs().max());context['queries']=int(frame['valid'].sum());context['observed_ids']=frame['observed'][0].nonzero().flatten().cpu().tolist()
        return base
    policy.model.forward=forward
    @torch.no_grad()
    def predict(state,instruction):
        if state.step==0:memory.reset()
        context['observation']=state.current_observation
        context['pose']=state.pose;context['step']=state.step
        if encoder is not None:context['encoded']=encoder.encode(state.current_observation,instruction)
        prediction=original_predict(state,instruction)
        prediction.diagnostics.update(native_object_control=control,native_object_history_window=history_window,native_object_queries=context.get('queries',0),native_object_delta_max=context.get('delta_max',0.),native_object_checkpoint=str(path),native_object_checkpoint_sha=native_sha,native_object_fusion_gain=float(ranker.fusion_gain.tanh()),native_observed_grid_ids=context.get('observed_ids',[]))
        return prediction
    policy.predict=predict
