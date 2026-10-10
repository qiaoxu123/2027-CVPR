"""Train's region and identity partition checks; original CityNav splits are retained."""
import argparse,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from experiments.visual_grounding.data.build_candidate_sets import partition

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='runs/visual_grounding_benchmark/data_audit/full/train_seen');a=p.parse_args();root=Path(a.data);ids=defaultdict(set);cells=defaultdict(set);counts=Counter();byid={};source_frames=defaultdict(set)
 for line in (root/'regions.jsonl').open():
  r=json.loads(line);assert r['split']=='train_seen';part=partition(r);obj=(r['map_name'],r['object_id']);xy=r['object_center_world'];cell=(r['map_name'],int(xy[0]//200),int(xy[1]//200));ids[part].add(obj);cells[part].add(cell);counts[part]+=1;byid[r['sample_id']]=part;source_frames[part].add(r['roi_source'])
 assert ids['train'].isdisjoint(ids['internal_dev']);assert cells['train'].isdisjoint(cells['internal_dev'])
 n=0
 for line in (root/'candidates.jsonl').open():
  q=json.loads(line)
  for sid in q['candidate_sample_ids']:assert byid[sid]==q['internal_partition']
  n+=1
 result={'object_counts':{k:len(v) for k,v in ids.items()},'region_counts':{k:len(v) for k,v in cells.items()},'observation_rows':dict(counts),'object_overlap':0,'region_overlap':0,'candidate_queries_checked':n,'every_candidate_partition_matches_query':True,'shared_source_frames_across_partitions':len(source_frames['train']&source_frames['internal_dev']),'source_frame_families_disjoint':not bool(source_frames['train']&source_frames['internal_dev']),'no_adapter_training_performed':True,'human_verified_attribute_labels':0};(root/'internal_split_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
