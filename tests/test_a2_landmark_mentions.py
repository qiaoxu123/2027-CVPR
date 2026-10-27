"""Fail-closed mapping of instruction references to BERT token offsets."""
from experiments.a2_waldo_landmark_belief.mentions import landmark_mention_spans


def test_distinct_references_have_separate_spans():
    text = "Find a car east of the Church and north of the Library."
    words = [(m.start(), m.end()) for m in __import__("re").finditer(r"\S+", text)]
    mask, valid = landmark_mention_spans(
        text, words,
        [
            dict(name="Church", linked=True),
            dict(name="Library", linked=True),
        ],
    )
    assert valid.tolist() == [True, True]
    assert mask[0].sum() == 1 and mask[1].sum() == 1
    assert not (mask[0] & mask[1]).any()


def test_ambiguous_names_fail_closed():
    text = "Go east of Bank."
    offsets = [(0, 2), (3, 7), (8, 10), (11, 15), (15, 16)]
    mask, valid = landmark_mention_spans(
        text, offsets, [
            dict(name="Bank", linked=True),
            dict(name="Bank", linked=True),
        ],
    )
    assert not valid.any() and not mask.any()


def test_missing_or_unlinked_mention_does_not_invent_landmark():
    text = "Near Church"
    mask, valid = landmark_mention_spans(
        text, [(0, 4), (5, 11)],
        [dict(name="Library", linked=True),
         dict(name="Church", linked=False)],
    )
    assert valid.tolist() == [False, False]
    assert not mask.any()


def test_partial_truncated_landmark_mention_fails_closed():
    text='north of Central Library'
    mask,valid=landmark_mention_spans(text,[(0,0),(0,5),(6,8),(9,16),(0,0)],[dict(name='Central Library',linked=True)])
    assert not valid.any() and not mask.any()


def test_unicode_matching_preserves_original_offset_positions():
    text='near Straße Church'
    mask,valid=landmark_mention_spans(text,[(0,4),(5,11),(12,18)],[dict(name='Straße Church',linked=True)])
    assert valid.tolist()==[True] and mask.tolist()==[[False,True,True]]
