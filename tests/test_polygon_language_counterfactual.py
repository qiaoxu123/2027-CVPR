from graphnav.polygon_language_counterfactual import swap_single_cardinal


def test_unique_direction_term_preserves_public_name_and_other_text():
    text="The white house north of One Stop, beside the blue car."
    swap=swap_single_cardinal(text,["Building One Stop"])
    assert swap is not None
    assert swap["old"]=="north"
    assert swap["new"]=="south"
    assert swap["mutated"]=="The white house south of One Stop, beside the blue car."


def test_multiple_direction_terms_abstain():
    assert swap_single_cardinal("north of A but east of B",[]) is None


def test_public_landmark_name_phrase_never_rewritten():
    text="Find the building called North of Heaven."
    assert swap_single_cardinal(text,["Building North of Heaven"]) is None


def test_ambiguous_left_or_behind_not_forced_to_world_cardinal():
    assert swap_single_cardinal("The blue car right of the house",[]) is None
    assert swap_single_cardinal("The blue car behind the store",[]) is None
