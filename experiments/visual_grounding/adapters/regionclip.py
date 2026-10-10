"""Actual RegionCLIP RN50 C4 region features using official Python modules.

No Detectron2 compiled extension is loaded. Official ROIAlign is itself a
Torchvision wrapper. ``images`` uses a full-image box on each observed ROI;
``regions`` accepts an observed source frame and projected XYXY boxes. Polygon
interiors are not segmentation masks. The latter reproduces the official GT
box feature route; proposal generation is a separate protocol.
"""
from __future__ import annotations
import ast
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

COMMIT = '4b8513b56e24827e3d6468e1f2105869f35c2d0b'
WEIGHT_SHA256 = 'afbc58d7d4bd96e2b10cc0d1928a912ed519c3b3215df7e5f3285e4bb09c7c58'

SOURCE_SHA256 = {'detectron2/modeling/backbone/clip_backbone.py': '8d8ca2c4e0487360590aafe70b2b65f02ac3625f8937e3da1abb7d588acf135b', 'detectron2/modeling/roi_heads/clip_roi_heads.py': '3ec9e4bc94549f1c810f6f970289c23d2c64fc069f1c1cb45acee645e19d58ef', 'detectron2/layers/roi_align.py': 'f6e8ba49495f892c702c3616b2760c16821dca9288e094eedc1cc36da05ffc36', 'detectron2/layers/batch_norm.py': 'fbf8ca0353476c0720a04dacafc4087eb30405a6aa59beb222ec345719c0333d', 'LICENSE': '2641e367636c2e4bdf95a1804025d7c957f8ea248c81aba007404cb7eb0e523e'}

def _sha256(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def _verify_weights(path):
    actual=_sha256(path)
    if actual != WEIGHT_SHA256:
        raise ValueError('RegionCLIP official RN50 checkpoint SHA256 mismatch: '+actual)
    return actual

def _official_namespace(repo):
    """Compile unchanged official class definitions, excluding package imports.

    This avoids importing unavailable legacy Detectron2 compiled operators.
    All math in FrozenBN, C4, res5, attention pooling and language tower is
    the actual pinned RegionCLIP source, not a renamed OpenAI visual model.
    """
    ns = dict(torch=torch, nn=nn, F=F, np=np, OrderedDict=OrderedDict,
              Tuple=Tuple, Union=Union, Backbone=nn.Module)
    files = [(repo/'detectron2/layers/batch_norm.py', {'FrozenBatchNorm2d'}),
             (repo/'detectron2/modeling/backbone/clip_backbone.py',
              {'Bottleneck', 'AttentionPool2d', 'ModifiedResNet', 'LayerNorm',
               'QuickGELU', 'ResidualAttentionBlock', 'Transformer', 'CLIPLangEncoder'})]
    for file, names in files:
        tree = ast.parse(file.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), 'exec'), ns)
    return ns

class RegionCLIPEncoder:
    def __init__(self, checkpoint, official_repo, device='cuda', batch_size=8):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.repo = Path(official_repo)
        actual_weight_sha=_verify_weights(checkpoint)
        source_hashes={file:_sha256(self.repo/file) for file in SOURCE_SHA256}
        if source_hashes != SOURCE_SHA256:
            raise ValueError('Pinned official RegionCLIP source SHA256 mismatch')
        ns = _official_namespace(self.repo)
        self.visual = ns['ModifiedResNet']((3,4,6,3),1024,32,224,64,
            out_features=['res4'],freeze_at=0,depth=50,pool_vec=False,create_att_pool=True)
        self.language = ns['CLIPLangEncoder'](1024,224,(3,4,6,3),64,None,
            77,49408,512,8,12,['res5'],0)
        weights = torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
        for module, prefix in [(self.visual,'backbone.'),(self.language,'lang_encoder.')]:
            state = {k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}
            module.load_state_dict(state, strict=True)
            module.to(self.device).eval().requires_grad_(False)
        from open_clip.tokenizer import SimpleTokenizer
        self.tokenizer = SimpleTokenizer()
        self.mean = torch.tensor([.48145466,.4578275,.40821073],device=self.device)[:,None,None]
        self.std = torch.tensor([.26862954,.26130258,.27577711],device=self.device)[:,None,None]
        self.metadata = dict(model='RegionCLIP RN50',official_commit=COMMIT,
            weights_sha256=actual_weight_sha,source_sha256=source_hashes,weights_license='not separately specified in model zoo',license='Apache-2.0 (code)',embedding_dim=1024,
            preprocessing='RGB ResizeShortestEdge 800 max1333; CLIP normalization',
            roi_route='official C4 -> torchvision ROIAlignV2 14 -> official res5 -> official attention pool',
            proposal_status='not_evaluated',proposal_recall=None)

    def _preprocess(self, bgr):
        from PIL import Image
        h,w = bgr.shape[:2]
        scale = min(800/min(h,w),1333/max(h,w))
        nw,nh = int(w*scale+.5),int(h*scale+.5)
        rgb = np.asarray(Image.fromarray(np.ascontiguousarray(bgr[:,:,::-1])).resize((nw,nh),Image.Resampling.BILINEAR))
        x = torch.from_numpy(rgb.copy()).permute(2,0,1).to(self.device).float()/255
        return (x-self.mean)/self.std, (nw/w,nh/h)

    @torch.inference_mode()
    def _regions_one(self, source_bgr, boxes_xyxy):
        import hashlib
        key=(source_bgr.shape,hashlib.sha256(np.ascontiguousarray(source_bgr).data).digest())
        if getattr(self, '_frame_key', None) != key:
            x,(sx,sy) = self._preprocess(source_bgr)
            self._frame_c4 = self.visual(x[None])['res4']
            self._frame_scale = (sx,sy)
            self._frame_key = key
        sx,sy=self._frame_scale
        boxes = torch.as_tensor(boxes_xyxy,dtype=torch.float32,device=self.device).reshape(-1,4)
        boxes = boxes*torch.tensor([sx,sy,sx,sy],device=self.device)
        c4 = self._frame_c4
        pooled = roi_align(c4,[boxes],(14,14),spatial_scale=1/16,sampling_ratio=0,aligned=True)
        features = self.visual.attnpool(self.visual.layer4(pooled))
        return F.normalize(features.float(),dim=-1).cpu()

    @torch.inference_mode()
    def regions(self, frames_bgr, boxes_xyxy):
        """One bbox per source frame; a single frame may also have many boxes."""
        if isinstance(frames_bgr, np.ndarray):
            return self._regions_one(frames_bgr, boxes_xyxy)
        if len(frames_bgr) != len(boxes_xyxy):
            raise ValueError('One bbox is required per source frame')
        # Group identical observed frames so each C4 map is computed once.
        import hashlib
        grouped = {}
        for index, (frame, box) in enumerate(zip(frames_bgr, boxes_xyxy)):
            key = (frame.shape, hashlib.sha256(np.ascontiguousarray(frame).data).digest())
            grouped.setdefault(key, [frame, [], []])
            grouped[key][1].append(box)
            grouped[key][2].append(index)
        output = torch.empty((len(frames_bgr),1024))
        for frame, boxes, indices in grouped.values():
            output[indices] = self._regions_one(frame, boxes)
        return output

    @torch.inference_mode()
    def images(self, images_bgr):
        vectors=[]
        for image in images_bgr:
            h,w = image.shape[:2]
            vectors.append(self._regions_one(image,[[0,0,w,h]]))
        return torch.cat(vectors) if vectors else torch.empty((0,1024))

    @torch.inference_mode()
    def texts(self, texts):
        vectors=[]
        for start in range(0,len(texts),self.batch_size):
            tokens=self.tokenizer(texts[start:start+self.batch_size]).to(self.device)
            vectors.append(F.normalize(self.language.encode_text(tokens).float(),dim=-1).cpu())
        return torch.cat(vectors) if vectors else torch.empty((0,1024))

    @torch.inference_mode()
    def verify_roi_align(self, source_bgr, boxes_xyxy):
        """Compare direct path with pinned official ROIAlign wrapper on real C4."""
        namespace={'nn':nn,'roi_align':roi_align}
        path=self.repo/'detectron2/layers/roi_align.py'
        exec(compile(path.read_text(),str(path),'exec'),namespace)
        x,(sx,sy)=self._preprocess(source_bgr)
        boxes=torch.as_tensor(boxes_xyxy,device=self.device,dtype=torch.float32)*torch.tensor([sx,sy,sx,sy],device=self.device)
        rois=torch.cat([torch.zeros((len(boxes),1),device=self.device),boxes],1)
        c4=self.visual(x[None])['res4']
        a=roi_align(c4,rois,(14,14),1/16,0,True)
        b=namespace['ROIAlign']((14,14),1/16,0,True)(c4,rois)
        fa=self.visual.attnpool(self.visual.layer4(a))
        fb=self.visual.attnpool(self.visual.layer4(b))
        return {'pool_max_abs_difference':float((a-b).abs().max()),
                'embedding_max_abs_difference':float((fa-fb).abs().max()),
                'finite':bool(torch.isfinite(fa).all()),'shape':list(fa.shape)}

    def enable_proposals(self, checkpoint, topk=100):
        """Load official axis-aligned offline RPN from the same checkpoint.

        Legacy unused rotated/deformable C++ ops fail loudly if called. Official
        horizontal RPN uses supported Torchvision NMS; no substitute detector.
        """
        _verify_weights(checkpoint)
        import sys, types
        from PIL import Image
        sys.path.insert(0,str(self.repo.resolve()))
        sys.modules.setdefault('detectron2._C',types.ModuleType('detectron2._C'))
        def unavailable(*args,**kwargs):
            raise RuntimeError('Legacy rotated operator unavailable in axis-aligned RPN compatibility path')
        for name in ['nms_rotated','box_iou_rotated','roi_align_rotated_forward','roi_align_rotated_backward']:
            if not hasattr(torch.ops.detectron2,name):
                setattr(torch.ops.detectron2,name,unavailable)
        if not hasattr(np.lib,'pad'): np.lib.pad=np.pad
        if not hasattr(Image,'LINEAR'): Image.LINEAR=Image.BILINEAR
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        cfg=get_cfg()
        cfg.merge_from_file(str(self.repo/'configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_zsinf.yaml'))
        cfg.MODEL.DEVICE=str(self.device)
        cfg.MODEL.MASK_ON=False
        cfg.MODEL.CLIP.TEXT_EMB_PATH=None
        cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG=str(self.repo/'configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml')
        cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST=topk
        model=build_model(cfg).eval().requires_grad_(False)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
        for module,prefix in [(model.offline_backbone,'offline_backbone.'),
                              (model.offline_proposal_generator,'offline_proposal_generator.')]:
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        self.proposal_model=model
        self.metadata['proposal_status']='official_offline_rpn_available'
        self.metadata['proposal_topk']=topk

    @torch.inference_mode()
    def proposals(self, frame_bgr):
        """Original observed-image XYXY boxes and official RPN objectness logits."""
        from detectron2.data.transforms import ResizeShortestEdge
        rgb=np.ascontiguousarray(frame_bgr[:,:,::-1])
        transform=ResizeShortestEdge([800,800],1333).get_transform(rgb)
        resized=transform.apply_image(rgb)
        inputs=[{'image':torch.as_tensor(np.ascontiguousarray(resized.transpose(2,0,1)))}]
        model=self.proposal_model
        images=model.offline_preprocess_image(inputs)
        features=model.offline_backbone(images.tensor)
        proposals,_=model.offline_proposal_generator(images,features,None)
        boxes=proposals[0].proposal_boxes.tensor.cpu().numpy()
        boxes=transform.inverse().apply_box(boxes)
        return torch.from_numpy(boxes),proposals[0].objectness_logits.cpu()

    @torch.inference_mode()
    def verify_official_region_route(self, frame_bgr, boxes_xyxy):
        """Compare adapter with instantiated official ROI head end-to-end.

        Requires enable_proposals(). Prediction layers are not used by this
        feature extraction route, matching tools/extract_region_features.py.
        """
        from detectron2.data.transforms import ResizeShortestEdge
        from detectron2.structures import Boxes
        model=self.proposal_model
        model.backbone.load_state_dict(self.visual.state_dict(),strict=True)
        rgb=np.ascontiguousarray(frame_bgr[:,:,::-1])
        transform=ResizeShortestEdge([800,800],1333).get_transform(rgb)
        resized=transform.apply_image(rgb)
        inputs=[{'image':torch.as_tensor(np.ascontiguousarray(resized.transpose(2,0,1)))}]
        images=model.preprocess_image(inputs)
        features=model.backbone(images.tensor)
        boxes=torch.from_numpy(transform.apply_box(np.asarray(boxes_xyxy,dtype=np.float32))).to(self.device)
        pooled=model.roi_heads._shared_roi_transform([features[f] for f in model.roi_heads.in_features],[Boxes(boxes)],model.backbone.layer4)
        official=F.normalize(model.backbone.attnpool(pooled).float(),dim=-1).cpu()
        adapter=self.regions(frame_bgr,boxes_xyxy)
        return dict(max_absolute_difference=float((official-adapter).abs().max()),
                    cosine_similarity=F.cosine_similarity(official,adapter).tolist(),
                    shape=list(adapter.shape),finite=bool(torch.isfinite(official).all()),
                    official_route='ResizeShortestEdge->CLIPFastRCNN.preprocess_image->backbone->CLIPRes5ROIHeads._shared_roi_transform->attention pooling',
                    passed=bool(torch.allclose(official,adapter,atol=2e-5,rtol=2e-4)))
