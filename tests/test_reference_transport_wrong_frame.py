import pytest
import torch
from torch import nn
from scripts.train_reference_evidence_transport import evaluate


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real GPU forwarding regression')
def test_wrong_frame_control_reaches_model_and_preserves_original_image_input():
    class Spy(nn.Module):
        def __init__(self):super().__init__();self.saw_donor=False
        def forward(self,**kwargs):
            self.saw_donor='transport_image_features' in kwargs and bool((kwargs['transport_image_features']==1).all())
            assert bool((kwargs['image_features']==0).all())
            out=torch.zeros(1,786,device=kwargs['image_features'].device);out[:,3 if self.saw_donor else 0]=1
            base=torch.zeros_like(out);base[:,0]=1
            return out,torch.zeros(1,784,device=out.device),base
    world=torch.zeros(786,2);world[3]=torch.tensor([50.,50.])
    row=dict(prior_logits=torch.zeros(784),local_uv=torch.zeros(2,2),image_features=torch.zeros(2,3,8),target_text=torch.zeros(8),context_text=torch.zeros(8),
        public_geometry=torch.zeros(2,31),quality=torch.ones(2,6),local_valid=torch.ones(2,dtype=torch.bool),candidate_world=world,target_world=torch.tensor([50.,50.]),
        frame_key='f',task_key='t',category='Car',full_text=torch.zeros(8),target_phrase='car')
    spy=Spy();result=evaluate(spy,[row],control='wrong-frame',donors={'f':torch.ones(2,3,8)})
    assert spy.saw_donor
    assert result['metrics']['goal_r1']==1.
