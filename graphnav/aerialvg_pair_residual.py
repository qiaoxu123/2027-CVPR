"""Observed navigation adapter around the actual official AerialVG pair module.

The upstream module is imported without the detector package's eager imports.
This reuses pretrained pair reasoning, not the detector's image pyramid or boxes.
"""
import argparse
import importlib
import sys
import types
from pathlib import Path
import torch
from torch import nn
from .workspace_paths import WORKSPACE


def official_relation_class():
    source=WORKSPACE/'third_party/AerialVG/model/AerialVG'
    if not (source/'relation.py').is_file():raise FileNotFoundError('Download official AerialVG into Workspace/third_party/AerialVG first')
    namespace='_graphnav_official_aerialvg'
    if namespace not in sys.modules:
        package=types.ModuleType(namespace);package.__path__=[str(source)];sys.modules[namespace]=package
    return importlib.import_module(namespace+'.relation').CrossSelfRelationTransformer


class AerialVGPairResidual(nn.Module):
    def __init__(self,context_dim=128,vision_dim=1024,checkpoint=None,topk=15):
        super().__init__();self.topk=topk
        self.pairs=official_relation_class()(d_model=256,num_heads=8,num_layers=3)
        checkpoint=Path(checkpoint) if checkpoint else WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth'
        with torch.serialization.safe_globals([argparse.Namespace]):state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        weights={k.removeprefix('relation_transformer.'):v for k,v in state['model'].items() if k.startswith('relation_transformer.')}
        self.pairs.load_state_dict(weights,strict=True)
        self.pairs.requires_grad_(False)
        self.visual=nn.Sequential(nn.Linear(vision_dim+context_dim+2,256),nn.LayerNorm(256))
        self.language=nn.Sequential(nn.Linear(768,256),nn.LayerNorm(256))
        self.gain=nn.Parameter(torch.zeros(()))

    def forward(self,current,history,context,xy,tokens,token_mask,known,static_logits):
        if tokens is None or token_mask is None:raise ValueError('Official AerialVG adapter requires full instruction tokens and their mask')
        output=torch.zeros_like(static_logits,dtype=torch.float32)
        for b in range(len(output)):
            ids=known[b].nonzero().flatten()
            if not len(ids) or not token_mask[b].any():continue
            ids=ids[static_logits[b,ids].topk(min(self.topk,len(ids))).indices]
            features=torch.cat(((current[b,ids]+history[b,ids])*.5,context[b,ids],xy[b,ids]),-1)
            visual=self.visual(features.float())[None]
            language=self.language(tokens[b:b+1].float())
            mask=token_mask[b:b+1].bool()
            pair_logits=self.pairs(visual,dict(encoded_text=language,text_token_mask=mask))[0]
            # Official logits are padded with -inf to length 256. Exclude those entries.
            scores=pair_logits[:,:mask.shape[1]][:,mask[0]].float().mean(-1)/16.
            output[b,ids]=scores.tanh()*self.gain.tanh()
        return output
