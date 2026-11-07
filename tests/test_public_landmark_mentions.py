from graphnav.public_landmark_mentions import PublicLandmarkMentions,add_mention_aliases

P=[[0,0],[10,0],[10,10],[0,10]]
def obj(name,kind='TrafficRoad'):return dict(name=name,object_type=kind,contour=P)

def test_road_abbreviations_overlaps_and_original_aliases():
    catalog={'map':{'1':obj('Orchard Road'),'2':obj('Garden Road'),'3':obj('Orchard Rd / Garden Rd')}}
    matcher=PublicLandmarkMentions(catalog);mentions=matcher.match('map','A house north of Orchard Rd / Garden Rd.')
    names={n for x in mentions for n in x['public_names']}
    assert names=={'Orchard Road','Garden Road','Orchard Rd / Garden Rd'}
    r=add_mention_aliases({'landmarks':[{'name':'Orchard Road','aliases':[]}]},mentions)
    assert 'Orchard Rd' in r['landmarks'][0]['aliases']

def test_ambiguous_road_stems_do_not_choose_an_instance():
    matcher=PublicLandmarkMentions({'map':{'1':obj('Orchard Road'),'2':obj('Orchard Street')}})
    assert not matcher.match('map','The building between Orchard and the park.')
    assert {n for m in matcher.match('map','On Orchard Street.') for n in m['public_names']}=={'Orchard Street'}

def test_homonymous_polygons_all_survive_without_target_inputs():
    matcher=PublicLandmarkMentions({'map':{'1':obj('Orchard Road'),'2':obj('Orchard Road')}})
    m=matcher.match('map','A building on Orchard Rd.');assert set(m[0]['public_ids'])=={'1','2'}
    assert all('target' not in k for k in m[0])

def test_conservative_typo_matching_is_name_only():
    matcher=PublicLandmarkMentions({'map':{'1':obj('Evergreen Road')}})
    assert {n for m in matcher.match('map','The car on Evergren Road.') for n in m['public_names']}=={'Evergreen Road'}

def test_original_span_restores_existing_direction_relation_parser():
    from graphnav.semantic_geometry import parse_spatial_relations
    instruction='A building north of Orchard Rd.'
    row={'landmarks':[{'name':'Orchard Road','aliases':[]}]}
    assert not parse_spatial_relations(instruction,row['landmarks'])
    mentions=PublicLandmarkMentions({'map':{'1':obj('Orchard Road')}}).match('map',instruction)
    add_mention_aliases(row,mentions)
    assert any(r.kind=='north' for r in parse_spatial_relations(instruction,row['landmarks']))

def test_street_name_does_not_refer_to_embedded_church_name():
    matcher=PublicLandmarkMentions({'map':{'1':obj('St Example Street'),'2':obj('St Example',kind='Building')}})
    names={n for m in matcher.match('map','The house on St Example Street.') for n in m['public_names']}
    assert names=={'St Example Street'}

def test_church_name_does_not_refer_to_embedded_street_name():
    matcher=PublicLandmarkMentions({'map':{'1':obj('Garden Street'),'2':obj('Garden Street Methodist Church',kind='Building')}})
    names={n for m in matcher.match('map','Beside Garden Street Methodist Church.') for n in m['public_names']}
    assert names=={'Garden Street Methodist Church'}

def test_color_size_words_do_not_become_road_references():
    matcher=PublicLandmarkMentions({'map':{'1':obj('Silver Street'),'2':obj('Short Street'),'3':obj('Garden Road')}})
    names={n for m in matcher.match('map','A silver car on the short side near Garden Road.') for n in m['public_names']}
    assert names=={'Garden Road'}

def test_bare_road_stem_does_not_rename_an_explicit_building():
    matcher=PublicLandmarkMentions({'map':{'1':obj('Oakley Street')}})
    assert not matcher.match('map','Near the road, in front of the Oakley Building.')

def test_leading_article_does_not_create_a_second_nested_entity():
    matcher=PublicLandmarkMentions({'map':{'1':obj('The Hall','Building'),'2':obj('Hall Court','Building')}})
    names={n for m in matcher.match('map','A house near the Hall Court.') for n in m['public_names']}
    assert names=={'Hall Court'}
