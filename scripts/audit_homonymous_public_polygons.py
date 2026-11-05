"""Count within-map name multiplicity; never infer physical identity from names."""
import json
from collections import defaultdict,Counter
from pathlib import Path
from scripts.analyze_instruction_landmark_association import normalise,GENERIC_NAMES


def audit(catalog,rows):
    groups=defaultdict(list)
    for map_name,objects in catalog.items():
        for oid,obj in objects.items():
            name=normalise(obj.get('name',''))[0]
            if len(name)<3 or name in GENERIC_NAMES:continue
            groups[(map_name,name)].append(dict(id=str(oid),type=obj.get('object_type','Unknown')))
    repeated={k:v for k,v in groups.items() if len(v)>1}
    kinds=Counter();examples=[]
    for (map_name,name),items in repeated.items():
        ts={v['type'] for v in items};kind='road_only' if ts <= {'TrafficRoad','Footpath','Bridge'} else 'building_only' if ts=={'Building'} else 'other_or_mixed'
        kinds[kind]+=1
        examples.append(dict(map_name=map_name,name=name,count=len(items),types=sorted(ts),ids=[v['id'] for v in items]))
    impact=Counter();used=Counter();row_groups=Counter()
    for row in rows:
        refs=defaultdict(set)
        types=defaultdict(set)
        for lm in row['landmarks']:
            name=normalise(lm['name'])[0];refs[name].add(str(lm['id']));types[name].add(lm['object_type'])
        multi={n:ids for n,ids in refs.items() if len(ids)>1}
        impact['all_queries']+=1;impact['with_anchor']+=bool(refs);impact['same_name_multi_polygon']+=bool(multi)
        road=any(types[n]<={'TrafficRoad','Footpath','Bridge'} for n in multi)
        building=any(types[n]=={'Building'} for n in multi)
        impact['with_road_multi']+=road;impact['with_building_multi']+=building
        impact['with_other_or_mixed_multi']+=any(not(types[n]<={'TrafficRoad','Footpath','Bridge'} or types[n]=={'Building'}) for n in multi)
        for n in multi:used[(row['map_name'],n)]+=1
        row_groups[row['map_name'].split('_block_')[0]]+=bool(multi)
    return dict(public_named_groups=len(groups),within_map_repeated_name_groups=len(repeated),fraction_repeated=len(repeated)/len(groups),group_categories=dict(kinds),TRAIN_impact=dict(impact),TRAIN_city_affected=dict(row_groups),most_frequent_bound_groups=[dict(map_name=k[0],name=k[1],queries=v,public_polygon_count=len(groups.get(k,[]))) for k,v in used.most_common(12)],largest_public_groups=sorted(examples,key=lambda x:-x['count'])[:15],caveat='Multiple object IDs under the same name do not establish distinct physical instances or contiguous road segments. Classification uses public types only. Over-capacity empty-anchor tasks are not counted as bound multiplicity.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--rows',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(json.loads(a.catalog.read_text()),[json.loads(x) for x in a.rows.read_text().splitlines()]);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
