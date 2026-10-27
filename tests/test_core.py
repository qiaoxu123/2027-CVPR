import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from geometry import relation_features, triad_features, sample_surface, choose_candidates
from fusion import local_mix
from labels import ranking_loss, associate


class CoreTests(unittest.TestCase):
    def test_relation_direction_distance_and_translation(self):
        c=np.array([[10.,0.],[0.,10.]])
        r=np.array([[0.,0.]])
        a=relation_features(c,r,[2,3],[0],[True,True],[True])
        b=relation_features(c+123,r+123,[2,3],[0],[True,True],[True])
        np.testing.assert_allclose(a,b,atol=1e-6)
        self.assertGreater(a[0,0,3],.99)
        self.assertGreater(a[1,0,2],.99)
        self.assertNotEqual(a[0,0,8],a[1,0,8])

    def test_missing_depth_values_are_ignored(self):
        c=np.array([[1.,2.]])
        a=relation_features(c,np.array([[0.,0.]]),[1e9],[-1e9],[False],[False])
        b=relation_features(c,np.array([[0.,0.]]),[0],[0],[False],[False])
        np.testing.assert_array_equal(a,b)

    def test_between(self):
        x,pairs=triad_features(np.array([[5.,0.],[5.,10.]]),np.array([[0.,0.],[10.,0.]]))
        self.assertEqual(pairs.tolist(),[[0,1]])
        self.assertAlmostEqual(float(x[0,0,0]),.5)
        self.assertLess(float(x[0,0,1]),float(x[1,0,1]))

    def test_surface_invalid_partial_and_constant(self):
        z=np.ones((40,40),np.float32)*12
        poly=np.array([[5,5],[25,5],[25,25],[5,25]],np.float32)
        s=sample_surface(z,np.ones_like(z,bool),poly)
        self.assertTrue(s['valid']);self.assertEqual(s['z'],12.)
        self.assertFalse(sample_surface(z,np.zeros_like(z,bool),poly)['valid'])
        self.assertFalse(sample_surface(z,np.ones_like(z,bool),poly-20)['valid'])

    def test_rank_unknown_is_not_negative(self):
        a=torch.tensor([1.,0.,5.],requires_grad=True)
        loss=ranking_loss(a,torch.tensor([1.,0.,0.]),torch.tensor([True,True,False]))
        loss.backward();self.assertEqual(float(a.grad[-1]),0.)
        self.assertAlmostEqual(float(loss.detach()),float((torch.logsumexp(a[:2],0)-a[0]).detach()),places=6)

    def test_ambiguous_object_not_hard_negative(self):
        poly=np.array([[0,0],[10,0],[10,10],[0,10]],np.float32)
        result=associate([poly],[0],[(0,0,poly),(1,0,poly)],0)
        self.assertFalse(bool(result['rank_label_mask'][0]))

    def test_local_mix_preserves_mass_outside_and_zero_identity(self):
        base=torch.tensor([0.,1.,2.,3.])
        grid=torch.tensor([[0.,0.],[1.,0.],[2.,0.],[20.,0.]])
        mask=torch.tensor([[True,True,False,False],[False,True,True,False]])
        same=local_mix(base,grid,grid[:2],torch.tensor([2.,-1.]),mask,0.)
        self.assertIs(same['logits'],base)
        out=local_mix(base,grid,grid[:2],torch.tensor([2.,-1.]),mask,.5)
        a=base.softmax(0);b=out['logits'].softmax(0)
        self.assertTrue(torch.allclose(a[3:],b[3:],atol=1e-7))
        self.assertAlmostEqual(float(a[:3].sum()),float(b[:3].sum()),places=6)
        self.assertGreater(float((a-b).abs().max()),1e-3)

    def test_selection_keeps_semantic_alternative(self):
        xy=np.array([[float(i),0.] for i in range(12)])
        ids=choose_candidates(xy,np.arange(12)[::-1],np.arange(12),4)
        self.assertIn(0,ids);self.assertIn(11,ids)

if __name__=='__main__':unittest.main()
