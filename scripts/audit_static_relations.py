"""Read-only grounded parser coverage audit; no targets are inspected."""
import argparse,json,re
from pathlib import Path
from collections import Counter
from graphnav.semantic_geometry import build_entities,parse_spatial_relations,canonical
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();seen=set();counts=Counter();missed=[];kinds=Counter()
    triggers={'cardinal':r'\b(?:north|south|east|west|northeast|northwest|southeast|southwest|northern|southern|eastern|western)\b','distance':r'\b(?:near|nearby|beside|close|next|far|away|meters|metres)\b','between':r'\bbetween\b','along_road':r'\b(?:along|alongside|road|street|lane|intersection)\b','sequence':r'\b(?:first|second|third|last|after|before|beyond|past)\b','egocentric':r'\b(?:left|right|front|behind)\b'}
    for line in Path(a.records).read_text().splitlines():
        r=json.loads(line);key=(r.get('map_name',''),r['instruction'])
        if key in seen:continue
        seen.add(key);text=canonical(r['instruction'])
        entities=build_entities(r.get('landmarks',[])[:16],[],r['world_bounds'])
        relations=parse_spatial_relations(r['instruction'],entities)[:8]
        counts['queries']+=1;counts['parsed_queries']+=bool(relations);counts['no_bound_entities']+=not entities
        kinds.update(x.kind for x in relations)
        hit=[name for name,pattern in triggers.items() if re.search(pattern,text)]
        for name in hit:
            counts[name+'_trigger_queries']+=1;counts[name+'_unparsed_queries']+=not relations
        if hit and not relations and len(missed)<100:
            missed.append({'map_name':key[0],'instruction':r['instruction'],'triggers':hit,'bound_names':[e['name'] for e in entities]})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'source':a.records,'sha256':file_sha256(a.records),'counts':dict(counts),'parsed_kind_counts':dict(kinds),'parsed_query_fraction':counts['parsed_queries']/counts['queries'],'unparsed_examples':missed,'limitations':'Keyword triggers include object names and attributes; they are not annotated relation truth. No targets read. Egocentric phrases cannot become world XY directions. No invented roads.'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
