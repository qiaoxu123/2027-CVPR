"""Coverage denominators and attribute instance counts, independent of model predictions."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='runs/visual_grounding_benchmark/data_audit/full');p.add_argument('--splits',nargs='+',default=['train_seen','val_seen','val_unseen']);a=p.parse_args();root=Path(a.data);stats=json.loads((root/'stats.json').read_text());splits={}
 for split in a.splits:
  counts=Counter();types=Counter();native=Counter();materialized=Counter();attribute=Counter();instances=defaultdict(set);observations=Counter();targets=set();targetepisodes=set();visible_frames=set();quality=Counter()
  for line in (root/'regions.jsonl').open():
   r=json.loads(line)
   if r['split']!=split:continue
   assert r['observed_step']<=r['observation_cutoff_step']
   key=(r['map_name'],r['object_id']);types[r['object_type']]+=1;observations[key]+=1;counts['valid_geometric_rois']+=1;visible_frames.add((r['episode_id'],r['observed_step']))
   if r['roi_path']:materialized[r['object_type']]+=1;counts['materialized_rois']+=1
   if r['roi_quality']['attribute_resolution_eligible']:
    native[r['object_type']]+=1
    for kind,item in r['attributes'].items():attribute[kind]+=1;instances[kind].add(key);quality[item['confidence']]+=1
   if r['is_episode_target']:targets.add(key);targetepisodes.add(r['episode_id']);counts['target_rois']+=1
  querycounts=Counter();queryinstances=defaultdict(set)
  for line in (root/'manifest.jsonl').open():
   r=json.loads(line)
   if r['split']!=split:continue
   if not r.get('query_attributes'):querycounts['unsupported_or_uncertain_visual_target_phrase']+=1
   for kind in r.get('query_attributes',{}):querycounts[kind]+=1;queryinstances[kind].add((r['map_name'],r['object_id']))
  candidates=Counter();sizes=Counter();candidateobjects=set();multis=0
  for line in (root/'candidates.jsonl').open():
   r=json.loads(line)
   if r['split']!=split:continue
   candidates[r['attribute_type']]+=1;sizes[len(r['candidate_sample_ids'])]+=1;multis+=len(r['positive_sample_ids'])>1
   candidateobjects.update((r['map_name'],oid) for oid in r['candidate_object_ids'])
  source=stats['splits'][split];splits[split]=dict(counts=dict(counts),roi_types=dict(types),native_side_at_least24=dict(native),materialized_types=dict(materialized),weak_attribute_rois=dict(attribute),weak_attribute_unique_instances={k:len(v) for k,v in instances.items()},query_attribute_rows=dict(querycounts),query_attribute_instances={k:len(v) for k,v in queryinstances.items()},verified_attribute_samples=0,visible_target_objects=len(targets),visible_target_episodes=len(targetepisodes),target_episode_visibility_rate=len(targetepisodes)/source['episodes_total'],target_object_visibility_rate=len(targets)/source['unique_target_objects'],mean_observations_per_visible_object=sum(observations.values())/max(1,len(observations)),candidate_queries_by_attribute=dict(candidates),candidate_size_distribution=dict(sizes),candidate_unique_objects=len(candidateobjects),multi_positive_queries=multis,source=source)
 result={'splits':splits,'supervision_status':'language weak labels only; no independent image attribute annotation','unsupported_attribute_categories':['building_size','surrounding_visual_context'],'inference_vs_supervision':'object IDs, annotation geometry and target positions remain offline linkage; encoders take source images and query only','observation_limit':'fixed human-prefix offline, not policy trajectories; current/past images only'};(root/'detailed_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
