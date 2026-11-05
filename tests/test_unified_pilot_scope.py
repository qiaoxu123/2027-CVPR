import pytest
from graphnav.evaluation_scope import validate_unified_scope

def test_full_default_and_explicit_pilot():
    validate_unified_scope('val_seen',None,False)
    validate_unified_scope('val_seen',128,True)

@pytest.mark.parametrize('split,n,pilot',[
 ('val_seen',128,False),('val_seen',None,True),('val_seen',0,True),
 ('val_unseen',128,True),('test_unseen',128,True)])
def test_reject_unintended_scope(split,n,pilot):
    with pytest.raises(ValueError):validate_unified_scope(split,n,pilot)
