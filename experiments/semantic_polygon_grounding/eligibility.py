"""Attribute-binding opportunity audit; no fabricated reference truth labels."""
from collections import Counter
import gzip
import json
import re

from .audit import OUT, COLORS, save
from graphnav.landmark_alias import canonical_name


def named_attribute_trigger(instruction, name):
    """A narrow lexical flag, never a visual/instance supervision label."""
    text=canonical_name(instruction);name=re.escape(canonical_name(name))
    pattern=rf'\b(?:{COLORS})\s+(?:(?:building|church|house|roofed)\s+)?{name}\b'
    pattern2=rf'\b{name}\s+(?:building\s+)?with\s+(?:a\s+)?(?:{COLORS})\s+(?:roof|facade|walls?)\b'
    return bool(re.search(pattern,text) or re.search(pattern2,text))


def audit_rows(rows):
    counts=Counter();examples=[]
    for r in rows:
        counts['queries']+=1
        ids_ambiguous=any(len(b.get('candidates',b.get('candidate_ids',[])))>1 for b in r['bindings'])
        counts['any_homonymous_ids']+=ids_ambiguous
        counts['named_attribute_trigger']+=any(named_attribute_trigger(r['instruction'],b['phrase']) for b in r['bindings'])
        if 'audit' not in r:
            # Raw queries use a name-keyed public candidate catalogue loaded below.
            continue
        opportunities=[]
        for b in r['bindings']:
            buildings=[c for c in b['candidates'] if c['object_type']=='Building' and c['valid_polygon']]
            observed=[c for c in buildings if c['credible_observed']]
            if len(buildings)>=2:
                opportunities.append(dict(phrase=b['phrase'],candidate_building_ids=[c['object_id'] for c in buildings],
                                          credible_building_ids=[c['object_id'] for c in observed],
                                          attribute_trigger=named_attribute_trigger(r['instruction'],b['phrase'])))
        counts['multiple_buildings_under_same_name']+=bool(opportunities)
        counts['multiple_visible_buildings_under_same_name']+=any(len(x['credible_building_ids'])>=2 for x in opportunities)
        counts['colored_name_and_multiple_visible_buildings']+=any(x['attribute_trigger'] and len(x['credible_building_ids'])>=2 for x in opportunities)
        counts['static_error_with_multiple_visible_buildings']+=not r['metrics']['goal_r1'] and any(len(x['credible_building_ids'])>=2 for x in opportunities)
        counts['parsed_any_relation']+=bool(r['parsed_relations'])
        counts['parsed_between']+=any(x['kind']=='between' for x in r['parsed_relations'])
        counts['parsed_cardinal']+=any(x['kind'] in ['east','west','north','south','northeast','northwest','southeast','southwest'] for x in r['parsed_relations'])
        counts['anonymous_color_reference_trigger']+=any(re.search(rf'\b(?:{COLORS})\b',p) for p in r['audit']['anonymous_reference_phrases'])
        if opportunities: examples.append(dict(sample_id=r['sample_id'],family_id=r['family_id'],map_name=r['map_name'],instruction=r['instruction'],static_error=not r['metrics']['goal_r1'],opportunities=opportunities,reference_instance_truth=None))
    return dict(counts),examples


def run():
    result={};examples={}
    for split in ['train','dev','seen']:
        rows=json.loads(gzip.open(OUT/f'{split}_family_predictions.json.gz','rt').read())
        result[split],examples[split]=audit_rows(rows)
    save(OUT/'rgb_binding_opportunities.json',dict(primary_family=result,limits='Lexical trigger counts are conservative lower-bound flags, not validated attribute or reference identity labels.',reference_instance_human_truth_families=0))
    save(OUT/'same_named_building_cases.json',examples)
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
