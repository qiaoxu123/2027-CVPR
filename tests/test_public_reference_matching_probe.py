import pytest
from scripts.probe_public_reference_matching import top1_credit


def test_ties_do_not_select_first_public_id():
    assert top1_credit([1,1,1],[True,False,False]) == pytest.approx(1/3)
    assert top1_credit([1,1,1],[False,False,True]) == pytest.approx(1/3)


def test_positive_alternative_set_and_empty_validation():
    assert top1_credit([.1,.9,.8],[False,True,True]) == 1
    assert top1_credit([.9,.8],[False,True]) == 0
    with pytest.raises(ValueError):top1_credit([.5],[False])
