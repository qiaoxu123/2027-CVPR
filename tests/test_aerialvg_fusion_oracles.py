import torch
from scripts.diagnose_aerialvg_fusion_oracles import one_hot_log_scores
from graphnav.object_query_belief import ObjectQueryRanker


def test_oracle_posterior_exactly_preserves_mass_including_zero_and_one():
    q=torch.zeros(1,3);null=torch.zeros(1);valid=torch.ones(1,3,dtype=torch.bool)
    for mass in [0.,.25,1.]:
        logits,n=one_hot_log_scores(q,null,valid,1,mass);p=torch.cat((logits,n[:,None]),-1).softmax(-1)
        assert torch.isfinite(p).all()
        assert abs(float(p[0,:-1].sum())-mass)<1e-6
        assert float(p[0,0])==0 and float(p[0,2])==0


def test_zero_mass_oracle_cannot_change_belief_or_create_nan():
    model=ObjectQueryRanker();model.fusion_gain.data.fill_(1.)
    valid=torch.ones(1,2,dtype=torch.bool);q,n=one_hot_log_scores(torch.zeros(1,2),torch.zeros(1),valid,0,0.)
    base=torch.tensor([[2.,0.,-1.]]);kernel=torch.tensor([[[0.,1.],[1.,0.],[0.,0.]]]);observed=torch.ones(1,3,dtype=torch.bool)
    out,delta=model.update_belief(base,q,n,kernel,observed,valid)
    assert torch.equal(out,base) and torch.equal(delta,torch.zeros_like(base))


def test_relaxed_box_exposes_unobserved_prior_floor_without_claiming_realizability():
    from scripts.diagnose_aerialvg_fusion_oracles import relaxed_margin
    base=torch.tensor([4.,0.]);legal=torch.tensor([False,True])
    assert relaxed_margin(base,legal,torch.tensor([False,True]),1.)==-3.
    assert relaxed_margin(base,legal,torch.tensor([True,True]),3.)==2.


def test_oracle_artifact_reproduces_model_and_respects_relaxed_upper_bound():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'reports/aerialvg_diagnostics_20261101'
    oracle=json.loads((root/'fusion_oracles_v2.json').read_text());original=json.loads((root/'candidate_summary.json').read_text())
    assert oracle['checkpoint_unchanged'] and oracle['checkpoint_sha']==original['checkpoint_sha256']
    for part in ['train','dev']:
        counts=oracle['summary'][part]['counts'];actual=original['splits'][part]['all']['fusion']['a0_guided_spatial/top15']
        for key in ['states','updated_correct','corrected','damaged']:assert counts[key]==actual[key]
        rows=oracle['details'][part];assert len(rows)==counts['failed_correct_rank']
        unresolved=[r for r in rows if not r['any_positive_single_query_full_mass_repairs'] and not r['GT_anchor_full_mass_repairs']]
        assert len(unresolved)==counts['neither_query_nor_GT_full_mass_repairs']
        for row in rows:
            if row['any_positive_single_query_full_mass_repairs'] or row['GT_anchor_full_mass_repairs']:
                assert row['relaxed_full_mass_margin']>=-1e-6
    assert oracle['summary']['dev']['counts']['failed_correct_rank']==37
