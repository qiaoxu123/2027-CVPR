from graphnav.landmark_ablation import instruction_landmarks_only


def test_remove_only_unbound_landmarks_and_preserve_mentioned_fragments():
    rows = [dict(name='Church',aliases=[]),dict(name='School',aliases=[]),
            dict(name='Park',aliases=[]),dict(name='Church',aliases=[])]
    record = dict(instruction='The building near Church and School',landmarks=rows)
    filtered,audit = instruction_landmarks_only(record,['Church','School'])
    assert [row['name'] for row in filtered['landmarks']]==['Church','School','Church']
    assert audit['removed_names']==['Park'] and not audit['target_used']
    assert len(record['landmarks'])==4


def test_instruction_binding_aliases_survive_abbreviation_and_fuzzy_resolution():
    row=dict(name='Willmore Road',aliases=['Wilmore Road'])
    record=dict(instruction='Near Wilmore Road',landmarks=[row])
    filtered,audit=instruction_landmarks_only(record,['Wilmore Road'])
    assert filtered==record and not audit['removed_names']
    filtered,audit=instruction_landmarks_only(record,[])
    assert not filtered['landmarks']
