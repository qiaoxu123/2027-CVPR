"""TRAIN-only waterfall over all public named landmarks and relation bindings."""
import argparse
from collections import Counter
from pathlib import Path

from graphnav.checkpointing import atomic_json
from graphnav.native_multireference import full_public_graph
from graphnav.semantic_geometry import canonical
from graphnav.stage1_data import Stage1Dataset
from scripts.train_native_preserving_relation_gate import checked_metadata
from scripts.train_object_query_belief import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['train','stage1-cache','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if Path(args.output).exists():raise ValueError('Use a new audit directory')
    checked_metadata(args.train,'train')
    static=Stage1Dataset(args.stage1_cache,verify=False).static
    lookup={}
    for i,record in enumerate(static.records):lookup.setdefault((record['map_name'],record['instruction']),i)
    _,rows=load(args.train)
    states,types=Counter(),Counter();seen=set();group_flags={};examples=[];public_entity_total=0;raw_landmark_total=0
    for row in rows:
        key=(row['map_name'],row['instruction']);index=lookup[key]
        graph=full_public_graph(static,index)
        relations=graph['relations']
        unique=[r for r in relations if r.anchors and all(len(a)==1 for a in r.anchors)]
        distance=(row['query_world'].float()-row['labels']['target_world'][:2].float()).norm(dim=-1)
        positive=(distance<=20)&row['valid']
        text=' '+canonical(row['instruction'])+' '
        mentioned=[e['name'] for e in graph['entities'] if canonical(e['name']) and ' '+canonical(e['name'])+' ' in text]
        flags=dict(all=True,parsed=bool(relations),unique_relation=bool(unique),
                   ambiguous_only=bool(relations) and not unique,
                   no_supported_relation=not relations,
                   unique_with_positive=bool(unique) and bool(positive.any()),
                   unique_missing_positive=bool(unique) and not bool(positive.any()),
                   ranking_supervised=bool(unique) and bool(positive.any()) and bool((row['valid']&~positive).any()),
                   genuine_multi_anchor=any(len(r.anchors)>1 for r in unique),
                   direct_name_mention=bool(mentioned))
        states.update(k for k,v in flags.items() if v)
        group_flags.setdefault(key,set()).update(k for k,v in flags.items() if v)
        if key not in seen:
            seen.add(key)
            types.update(r.kind for r in relations)
            public_entity_total+=len(graph['entities']);raw_landmark_total+=len(static.records[index].get('landmarks',[]))
            if mentioned and not unique and len(examples)<10:
                examples.append(dict(instruction=row['instruction'],public_names=mentioned,
                                     relation_kinds=[r.kind for r in relations],ambiguous=flags['ambiguous_only']))
    groups=Counter(flag for flags in group_flags.values() for flag in flags)
    atomic_json(dict(states=dict(states),group_count_semantics='at least one state in the group meets each flag; coverage/missing coverage can overlap',instruction_groups=dict(groups),parsed_relation_kinds_by_group=dict(types),
                     public_entity_occurrences_across_instruction_groups=public_entity_total,
                     raw_landmark_occurrences_across_instruction_groups=raw_landmark_total,
                     name_mention_definition='direct canonical whole-name phrase; aliases not counted by this audit proxy',
                     examples=examples),Path(args.output))


if __name__=='__main__':main()
