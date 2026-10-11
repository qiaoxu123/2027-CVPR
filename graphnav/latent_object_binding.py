"""Optional P6 Object Belief -> 28x28 Goal adapter; existing navigation unchanged.

Caller supplies only public geometry/belief and already validated causal node
observables. No target identity, endpoint, annotation or future image is accepted.
"""
import numpy as np
import torch
from experiments.latent_object_binding.inference import prepare,posterior
from experiments.latent_object_binding.factors import Geometry

class LatentObjectGrounder:
    def __init__(self,graph,weights=None,blend=.25):
        self.graph=graph;self.weights=weights;self.blend=float(blend);self.geometry=None;self.frame_key=None
    @torch.no_grad()
    def __call__(self,instruction,public,observed_evidence=None):
        if len(public['grid_xy'])!=784:raise ValueError('unchanged 28x28 candidate interface required')
        b=np.asarray(public['bounds']);world=b[:2]+np.asarray(public['grid_xy'])*(b[2:]-b[:2]);key=world.tobytes()
        if key!=self.frame_key:self.geometry=Geometry(self.graph,world);self.frame_key=key
        p=prepare(self.graph,instruction,public,visual=observed_evidence,geometry=self.geometry);r=posterior(p,self.weights,blend=self.blend)
        return dict(object_ids=list(self.graph.ids),object_belief=r['object_belief'].cpu(),goal_logits=r['logits'].cpu(),goal_heatmap=r['heatmap'].cpu().reshape(28,28),confidence=float(r['known_mass']),unknown=p['unknown'],variables=p['language'].get('variables',[]),reference_posteriors=p['reference_posteriors'],assignments=p['assignments'],assignment_posterior=r['assignment_posterior'].cpu())
