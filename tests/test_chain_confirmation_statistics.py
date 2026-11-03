from scripts.report_chain_confirmation import intervals


def test_identical_paired_results_have_zero_uncertainty():
    result=intervals([0,0,0,0],['a','a','b','c'])
    assert result['target_bootstrap95']==[0.,0.]
    assert result['map_bootstrap95']==[0.,0.]
    assert result['maps']==3


def test_constant_paired_loss_is_not_hidden_by_map_resampling():
    result=intervals([-1,-1,-1,-1],['a','a','b','c'])
    assert result['target_bootstrap95']==[-1.,-1.]
    assert result['map_bootstrap95']==[-1.,-1.]
