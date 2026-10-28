"""Small public-only outcome head; private supervision lives in a separate loss."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .risk_probe import FEATURE_NAMES
from .revision_fusion import bounded_belief_mix,MAX_GATE

NEUTRAL,GAIN,HARM=0,1,2


class DirectRiskHead(nn.Module):
    def __init__(self,mean,std,class_weights):
        super().__init__()
        mean=torch.as_tensor(mean,dtype=torch.float32).clone();std=torch.as_tensor(std,dtype=torch.float32).clone()
        weights=torch.as_tensor(class_weights,dtype=torch.float32).clone()
        if mean.shape!=(len(FEATURE_NAMES),) or std.shape!=mean.shape or weights.shape!=(3,):raise ValueError('Risk head normalization/class schema changed')
        if not all(torch.isfinite(t).all() for t in (mean,std,weights)) or (std<0).any() or (weights<=0).any():raise ValueError('Invalid TRAIN normalization/weights')
        self.register_buffer('mean',mean);self.register_buffer('std',std.clamp_min(1e-6));self.register_buffer('class_weights',weights)
        self.linear=nn.Linear(len(FEATURE_NAMES),3)

    def forward(self,features):
        if features.ndim!=2 or features.shape[-1]!=len(FEATURE_NAMES) or not torch.isfinite(features).all():raise ValueError('Invalid public risk feature input')
        return self.linear((features.float()-self.mean)/self.std)

    def corrected_logits(self,logits):
        # Remove the explicit balanced-CE class bias. This does NOT guarantee
        # calibrated probabilities under distribution shift or refinement loss.
        return logits-self.class_weights.log()


def outcome_classes(outcomes):
    if outcomes.ndim!=1 or not ((outcomes==0)|(outcomes==1)|(outcomes==-1)).all():raise ValueError('Invalid private outcome labels')
    return torch.where(outcomes<0,HARM,torch.where(outcomes>0,GAIN,NEUTRAL)).long()


def head_loss(logits,classes,class_weights,*,refine=False):
    classification=F.cross_entropy(logits,classes,weight=class_weights)
    probability=(logits-class_weights.log()).softmax(-1)[:,GAIN]
    cost=(probability*(10*(classes==HARM).float()-(classes==GAIN).float())).mean()
    loss=classification+(.2*cost if refine else 0.)
    return loss,dict(classification=float(classification.detach()),risk_cost=float(cost.detach()))


def apply_direct_risk(base,proposal,corrected_logits,available,parameters):
    """The only two actions are exact A0 or the fixed.25 proposal mixture."""
    minimum_gain=float(parameters.get('minimum_gain',1.));maximum_harm=float(parameters.get('maximum_harm',0.));temperature=float(parameters.get('temperature',1.))
    if not all(math.isfinite(x) for x in (minimum_gain,maximum_harm,temperature)) or not 0<=minimum_gain<=1 or not 0<=maximum_harm<=1 or temperature<=0:
        raise ValueError('Invalid independent risk thresholds')
    if corrected_logits.shape!=(3,) or not torch.isfinite(corrected_logits).all():raise ValueError('Invalid threeclass risk scores')
    scores=(corrected_logits.float()/temperature).softmax(-1)
    gain=float(scores[GAIN]);harm=float(scores[HARM]);active=bool(parameters.get('enabled',False) and available and gain>=minimum_gain and harm<=maximum_harm)
    gate=MAX_GATE if active else 0.;mixed=bounded_belief_mix(base,proposal,gate) if active else base
    return mixed,dict(updated=not torch.equal(base,mixed),gate=gate,gain_score=gain,harm_score=harm)
