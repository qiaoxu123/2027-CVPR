"""CPU regressions for the predeclared precision gate, not model benchmarks."""
import math
import torch
from experiments.visual_grounding.verify_precision import compare_features


def test_precision_gate_rejects_score_error_despite_aligned_features():
    fp=torch.tensor([[1.,0.]])
    bf=torch.tensor([[math.sqrt(1-.02**2),.02]])
    text=torch.tensor([[0.,1.]])
    result=compare_features(fp,bf,text,text)
    assert result['image_cosine_min']>.995
    assert result['score_abs_error_max']>.01
    assert not result['passed']


def test_precision_gate_rejects_feature_rotation_with_small_score_error():
    fp=torch.tensor([[1.,0.]])
    bf=torch.tensor([[.994,math.sqrt(1-.994**2)]])
    result=compare_features(fp,bf,fp,fp)
    assert result['score_abs_error_max']<.01
    assert result['image_cosine_min']<.995
    assert not result['passed']


def test_near_tie_rank_change_is_reported_without_failing_precision():
    fp=torch.tensor([[1.,0.]])
    bf=torch.nn.functional.normalize(torch.tensor([[1.,-.0005]]),dim=-1)
    text=torch.nn.functional.normalize(torch.tensor([[1.,.001],[1.,-.001]]),dim=-1)
    result=compare_features(fp,bf,text,text)
    assert result['passed']
    assert result['top1_text_rank_changes']==1
    assert result['fp32_top1_top2_score_gaps']==[0.]


def test_nonfinite_embedding_fails_precision_gate():
    fp=torch.tensor([[1.,0.]])
    result=compare_features(fp,torch.tensor([[float('nan'),0.]]),fp,fp)
    assert not result['finite']
    assert not result['passed']
