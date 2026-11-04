"""Distill on the actual 784-cell output, not a sampled-candidate surrogate."""
import torch
from .geometry_guided_visual import landmark_visual_loss,visual_world_beliefs


def world_landmark_visual_loss(model,batch):
    parts=landmark_visual_loss(model,batch)
    score=model(image_features=batch['image_features'],name_text=batch['name_text'],type_text=batch['type_text'],valid=batch['valid'])
    fields=visual_world_beliefs(batch['geometry_logits'],batch['local_uv'],score,batch['valid'],batch['projection_supported'])
    covered=fields['covered'];active=fields['has_view']
    world_kl=score.sum()*0
    if active.any():
        mask=covered[active]
        geometry=batch['geometry_logits'][active].detach().float()
        g=geometry.masked_fill(~mask,float('-inf')).log_softmax(-1).masked_fill(~mask,0)
        v=fields['raw'][active].log_softmax(-1).masked_fill(~mask,0)
        world_kl=(g.exp()*mask*(g-v)).sum(-1).mean()
    gp=batch['geometry_logits'].detach().float().log_softmax(-1)
    fp=fields['fused'].log_softmax(-1)
    preservation=(gp.exp()*(gp-fp)).sum(-1).mean()
    distill=world_kl+preservation
    return dict(total=parts['region']+.25*distill+.1*parts['contrast'],region=parts['region'],
        distill=distill,contrast=parts['contrast'],contrast_rows=parts['contrast_rows'],
        conditional_world_kl=world_kl,global_preservation_kl=preservation)
