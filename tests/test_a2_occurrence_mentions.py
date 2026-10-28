import re

import numpy as np
import pytest

from experiments.a2_waldo_landmark_belief.occurrence_mentions import occurrence_mention_spans


def refs(*names):
    return [dict(name=name, aliases=[name], linked=True) for name in names]


def offsets(text):
    return [[m.start(), m.end()] for m in re.finditer(r'\S+', text)]


def test_separate_street_and_building_mentions_both_remain_usable():
    text = 'On Castle Street beside 12 Castle Street'
    masks, valid = occurrence_mention_spans(text, offsets(text), refs('Castle Street','12 Castle Street'))
    assert valid.tolist() == [True, True]
    assert np.flatnonzero(masks[0]).tolist() == [1,2]
    assert np.flatnonzero(masks[1]).tolist() == [4,5,6]


def test_only_longer_public_name_is_not_also_a_separate_street_mention():
    text = 'Beside 12 Castle Street'
    masks, valid = occurrence_mention_spans(text, offsets(text), refs('Castle Street','12 Castle Street'))
    assert valid.tolist() == [False, True]
    assert not masks[0].any()


def test_exact_same_name_identity_ambiguity_stays_unconfirmed():
    text = 'Beside Castle Street'
    masks, valid = occurrence_mention_spans(text, offsets(text), refs('Castle Street','Castle Street'))
    assert not valid.any() and not masks.any()


def test_conflicting_occurrence_does_not_erase_an_independent_public_alias():
    text = 'Library beside Town Hall'
    references = refs('Library','Library')
    references[0]['aliases'].append('Town Hall')
    masks, valid = occurrence_mention_spans(text, offsets(text), references)
    assert valid.tolist() == [True, False]
    assert np.flatnonzero(masks[0]).tolist() == [2,3]


def test_crossing_names_are_ambiguous_without_a_containing_public_name():
    text = 'York Road Station'
    masks, valid = occurrence_mention_spans(text, offsets(text), refs('York Road','Road Station'))
    assert not valid.any() and not masks.any()


def test_truncated_long_name_cannot_become_a_short_name_match():
    text = 'Beside Castle Street Methodist Church'
    short_offsets = offsets(text)[:3]
    masks, valid = occurrence_mention_spans(text, short_offsets,
        refs('Castle Street','Castle Street Methodist Church'))
    assert not valid.any() and not masks.any()


def test_unlinked_or_fuzzy_names_do_not_gain_evidence():
    text = 'Next to CASTLE STREET, not Castlemain'
    references = refs('Castle Street','Castle','Castlemain Road')
    references[0]['linked'] = False
    _, valid = occurrence_mention_spans(text, offsets(text), references)
    assert valid.tolist() == [False, True, False]


def test_nonoverlapping_legacy_outputs_match_exactly_and_input_is_unchanged():
    from experiments.a2_waldo_landmark_belief.mentions import landmark_mention_spans
    text = 'Library next to Market'
    references = refs('Library','Market')
    old = landmark_mention_spans(text, offsets(text), references)
    new = occurrence_mention_spans(text, offsets(text), references)
    for a,b in zip(old,new): np.testing.assert_array_equal(a,b)
    assert references == refs('Library','Market')


@pytest.mark.parametrize('bad_offsets', [[[0,999]], [[2,1]], [[.5,2.]], [[-1,1]]])
def test_invalid_token_offsets_are_rejected(bad_offsets):
    with pytest.raises(ValueError): occurrence_mention_spans('Library',bad_offsets,refs('Library'))


def test_empty_references_have_aligned_empty_masks():
    masks, valid = occurrence_mention_spans('Library',[[0,7]],[])
    assert masks.shape == (0,1) and valid.shape == (0,)
