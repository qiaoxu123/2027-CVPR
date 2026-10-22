import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np,torch
from scripts.evaluate_grid384 import capture
from scripts.train_graphnav_grid384 import TrainingSampler

class Policy:
 def __init__(self):
  self.model=torch.nn.Identity();self.args=S(grid_size=2);self.world=torch.zeros(1,4,2)
 def set_episode(self,*args):self.record={'map_name':'map'}
 def predict(self,state,instruction):
  self.model(S(static=S(belief_logits=torch.zeros(1,4)),belief=S(belief_logits=torch.ones(1,4)),delta=torch.ones(1,4),observed=torch.ones(1,4,dtype=torch.bool)))
  return S(diagnostics={})

class CaptureTests(unittest.TestCase):
 def test_duplicate_queries_do_not_mix_episodes(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'episodes.json').write_text(json.dumps([dict(episode_id=['map',i,0],index=i) for i in range(4)]))
   p=Policy();remove=capture(p,root/'fields')
   for ep in range(4):
    p.set_episode(None)
    for step in range(ep+1):
     state=S(step=step,pose=[ep,step,10],current_observation={'bgr':np.zeros((2,2,3),np.uint8),'footprint_world':np.zeros((4,2))})
     result=p.predict(state,'identical language')
     if ep<3:
      a=np.load(result.diagnostics['saved_dense_field']);self.assertEqual(int(a['episode_index']),ep);self.assertEqual(int(a['step']),step)
     else:self.assertNotIn('saved_dense_field',result.diagnostics)
   self.assertEqual(len(list((root/'fields').glob('*.npz'))),6);remove()
 def test_sampler_resume_preserves_remaining_order(self):
  sampler=TrainingSampler.__new__(TrainingSampler);sampler.base=[7,6,0,2,5,1,3,4,11,10,9,8];sampler.dev={1};sampler.data=S(manifest={'protocol':{'episode_indices':[0,1,2]}})
  full=list(sampler);sampler.skip=3;self.assertEqual(list(sampler),full[3:])

if __name__=='__main__':unittest.main()
