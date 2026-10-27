import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train import audit_groups, select_epoch
from labels import objective
from model import LocalRelationNetwork
from test_model import inputs
from calibrate import calibration_eligible


class TrainingTests(unittest.TestCase):
    def test_split_overlap_rejected(self):
        row={'target_instance':['map',3]}
        with self.assertRaises(ValueError):audit_groups({'train':[row],'dev':[row],'calibration':[]})
    def test_degrading_epoch_not_selected(self):
        self.assertFalse(select_epoch({'eligible':True,'new_errors':1,'rank1':1.,'mrr':1.,'xy_error':0.,'epoch':1},None))
        self.assertTrue(select_epoch({'eligible':True,'new_errors':0,'rank1':.5,'mrr':.6,'xy_error':3.,'epoch':1},None))
    def test_no_supervision_is_finite(self):
        torch.set_num_threads(2);m=LocalRelationNetwork(rgb_dim=32);x=inputs(n=0,l=0);o=m(**x)
        y=dict(ranking=torch.empty(0),rank_label_mask=torch.empty(0,dtype=torch.bool),angle=torch.empty(0,2),angle_valid=torch.empty(0,dtype=torch.bool),
            distance=torch.empty(0),height=torch.empty(0),height_valid=torch.empty(0,dtype=torch.bool),match=torch.tensor(0.),match_valid=torch.tensor(False),
            offset=torch.empty(0,2),offset_valid=torch.empty(0,dtype=torch.bool))
        loss,_=objective(o,y,True);self.assertTrue(torch.isfinite(loss));loss.backward()
    def test_calibration_cannot_bypass_failed_dev_or_smoke(self):
        state=dict(diagnostic_only=False,metrics=dict(eligible=True,new_errors=1))
        self.assertFalse(calibration_eligible(state,20,20,20))
        state['metrics']['new_errors']=0
        self.assertTrue(calibration_eligible(state,20,20,20))
        self.assertFalse(calibration_eligible(state,20,20,20,smoke=True))

if __name__=='__main__':unittest.main()
