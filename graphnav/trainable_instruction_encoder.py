"""Differentiable instruction BERT with immutable cached public-name embeddings."""
import copy
import torch
from .token_language import FrozenBERTTokens

class TrainableInstructionBERT(FrozenBERTTokens):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.frozen_names=copy.deepcopy(self.model).eval().requires_grad_(False)
  self.model.requires_grad_(True)
  if getattr(self.model,"pooler",None) is not None:self.model.pooler.requires_grad_(False)
 def encode_tokens(self,texts):
  # Never reuse pretrained instruction features when optimizing BERT.
  batch=self.tokenizer(list(texts),truncation=True,padding=True,max_length=self.max_length,return_tensors='pt').to(self.device)
  states=self.model(**batch).last_hidden_state
  return states,batch['attention_mask'].bool()
 def encode(self,texts):
  # Public names retain the exact frozen source embedding throughout training.
  missing=[t for t in texts if t not in self.cache]
  if missing:
   unique=list(dict.fromkeys(missing))
   with torch.no_grad():
    for start in range(0,len(unique),64):
     names=unique[start:start+64];batch=self.tokenizer(names,truncation=True,padding=True,max_length=self.max_length,return_tensors='pt').to(self.device);states=self.frozen_names(**batch).last_hidden_state
     for i,name in enumerate(names):
      length=int(batch['attention_mask'][i].sum());self.cache[name]=(states[i,:length].detach().cpu().clone(),batch['attention_mask'][i,:length].bool().cpu().clone())
  return torch.stack([self.cache[t][0][0].float() for t in texts]).to(self.device).detach()
