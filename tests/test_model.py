import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from model import LocalRelationNetwork


def inputs(n=3,l=2):
    torch.manual_seed(17)
    return dict(tokens=torch.randn(9,768),token_mask=torch.ones(9,dtype=torch.bool),
        candidate_rgb=torch.randn(n,32),reference_rgb=torch.randn(l,32),
        names=torch.randn(l,5,768),name_mask=torch.ones(l,5,dtype=torch.bool),
        candidate_depth=torch.randn(n,2,32,32),reference_depth=torch.randn(l,2,32,32),
        candidate_geometry=torch.randn(n,12),reference_geometry=torch.randn(l,12),
        pair=torch.randn(n,l,18),triad=torch.randn(n,l*(l-1)//2,4),
        triad_ids=torch.tensor([(a,b) for a in range(l) for b in range(a+1,l)],dtype=torch.long).reshape(-1,2),
        offset_limit=torch.ones(n,2)*5)


class ModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2);self.model=LocalRelationNetwork(rgb_dim=32).eval()
    def test_empty_lists_and_backward(self):
        for n,l in [(0,0),(3,0),(0,2),(3,2)]:
            o=self.model(**inputs(n,l));self.assertEqual(o['scores'].shape,(n,))
            self.assertTrue(all(torch.isfinite(v).all() for v in o.values() if torch.is_tensor(v)))
        o=self.model(**inputs());sum(o[k].sum() for k in ['scores','direction','height','offset','match_logit']).backward()
        self.assertGreater(float(self.model.score.weight.grad.norm()),0)
    def test_tokens_and_distance_have_effect(self):
        x=inputs();a=self.model(**x)['scores'];x['tokens']=-x['tokens']
        b=self.model(**x)['scores'];self.assertGreater(float((a-b).abs().max().detach()),1e-5)
        x['pair'][...,4]+=3
        self.assertGreater(float((b-self.model(**x)['scores']).abs().max().detach()),1e-5)
    def test_invalid_depth_fill_has_no_effect(self):
        x=inputs();x['candidate_depth'][:,1]=0;x['reference_depth'][:,1]=0
        a=self.model(**x)['scores'];x['candidate_depth'][:,0]=1e9;x['reference_depth'][:,0]=-1e9
        self.assertTrue(torch.allclose(a,self.model(**x)['scores'],atol=1e-6))
    def test_candidate_permutation(self):
        x=inputs();a=self.model(**x);order=torch.tensor([2,0,1]);y=dict(x)
        for k in ['candidate_rgb','candidate_depth','candidate_geometry','pair','triad','offset_limit']:y[k]=x[k][order]
        b=self.model(**y);self.assertTrue(torch.allclose(a['scores'][order],b['scores'],atol=1e-5))
    def test_all_masked_language_rejected(self):
        x=inputs();x['token_mask'][:]=False
        with self.assertRaises(ValueError):self.model(**x)
    def test_unknown_relation_height_fill_has_no_effect(self):
        x=inputs();x['pair'][...,12]=0
        a=self.model(**x)['scores'];x['pair'][...,8:11]=1e8
        self.assertTrue(torch.allclose(a,self.model(**x)['scores'],atol=1e-6))
    def test_reference_permutation_and_between_symmetry(self):
        x=inputs();a=self.model(**x);order=torch.tensor([1,0]);y=dict(x)
        for key in ['reference_rgb','reference_depth','reference_geometry','names','name_mask']:y[key]=x[key][order]
        y['pair']=x['pair'][:,order];y['triad']=x['triad'].clone();y['triad'][...,0]=1-y['triad'][...,0]
        b=self.model(**y)
        self.assertTrue(torch.allclose(a['scores'],b['scores'],atol=1e-5))

if __name__=='__main__':unittest.main()
