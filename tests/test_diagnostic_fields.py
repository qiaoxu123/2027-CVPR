from types import SimpleNamespace
import numpy as np
import torch
from graphnav.diagnostic_fields import install_field_capture


def test_capture_saves_real_initial_field_and_restores_policy(tmp_path):
    class Model(torch.nn.Module):
        def forward(self):
            field=torch.arange(4,dtype=torch.float32)[None]
            return SimpleNamespace(static=SimpleNamespace(belief_logits=field),
                belief=SimpleNamespace(belief_logits=field+1),delta=torch.ones_like(field))
    model=Model()
    def predict(state,instruction):
        model()
        return 'prediction'
    policy=SimpleNamespace(model=model,predict=predict,args=SimpleNamespace(grid_size=2),
        record=dict(map_name='public_map'),world=torch.zeros(1,4,2))
    observation=dict(bgr=np.zeros((384,384,3),np.uint8),footprint_world=np.zeros((4,2)))
    state=SimpleNamespace(step=0,current_observation=observation)
    remove=install_field_capture(policy,tmp_path)
    assert policy.predict(state,'public instruction')=='prediction'
    saved=list(tmp_path.glob('*.npz'));assert len(saved)==1
    data=np.load(saved[0]);np.testing.assert_equal(data['belief_logits'],[1,2,3,4])
    assert data['bgr'].shape==(384,384,3)
    remove();assert policy.predict is predict
    assert len(model._forward_hooks)==0
