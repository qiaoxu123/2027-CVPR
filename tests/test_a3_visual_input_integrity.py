"""Guard regression: partial or misaligned P4 data must never be silently zipped."""
import pytest
from experiments.visual_grounding.evaluate_a3_reranking import validate_aligned_query_rows


def rows():
    return [dict(query_index=i,map_name='map_a',source_episode_index=i+10) for i in range(3)]


def test_complete_rows_align_with_dense_query_count():
    assert validate_aligned_query_rows(rows(),rows(),expected_count=3)==3


def test_missing_evidence_row_is_rejected_before_zip():
    with pytest.raises(ValueError,match='row count mismatch'):
        validate_aligned_query_rows(rows(),rows()[:-1])


@pytest.mark.parametrize('key,value',[('query_index',9),('map_name','map_b'),('source_episode_index',99)])
def test_mismatched_query_identity_is_rejected(key,value):
    evidence=rows();evidence[1][key]=value
    with pytest.raises(ValueError,match=key+' mismatch'):
        validate_aligned_query_rows(rows(),evidence)


def test_missing_dense_query_is_rejected_even_when_both_lists_short():
    with pytest.raises(ValueError,match='dense query count'):
        validate_aligned_query_rows(rows()[:-1],rows()[:-1],expected_count=3)


def test_duplicate_query_identity_is_rejected():
    static=rows();static[2]['query_index']=1
    with pytest.raises(ValueError,match='Duplicate'):
        validate_aligned_query_rows(static,static)


def test_reordered_evidence_is_rejected():
    with pytest.raises(ValueError,match='query_index mismatch'):
        validate_aligned_query_rows(rows(),list(reversed(rows())))
