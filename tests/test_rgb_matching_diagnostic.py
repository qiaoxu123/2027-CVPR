from scripts.evaluate_observed_rgb_matching import retrieval


def test_retrieval_ties_do_not_favor_positive_position():
    for positive in range(3):
        assert retrieval([.1,.1,.1],positive)['r1']==1/3


def test_retrieval_margin_uses_a_different_instance():
    result=retrieval([.7,.2,.4],0)
    assert result['r1']==1 and abs(result['margin']-.3)<1e-8
