"""Public eligibility first; TRAIN GT only in a separate post-hoc agreement audit."""
import argparse,json,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from graphnav.unified_direct_bearing_loss import DirectBearingDataset
from graphnav.checkpointing import atomic_json,file_sha256


def main():
 p=argparse.ArgumentParser()
 for n in ('source','original-v2','public-objects','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();s=torch.load(a.original_v2,map_location='cpu',weights_only=True);c=s['config']['geometry_args']
 ds=DirectBearingDataset(a.source,public_objects=a.public_objects,deduplicate_queries=False,grid_size=28,max_landmarks=c['max_landmarks'],max_roads=c['max_roads'],max_relations=c['max_relations'],prior_sigma_m=c['prior_sigma_m'],rule_sigma_m=c['rule_sigma_m'])
 assert len(ds)==21878
 stats=Counter();per=Counter();rows=[];wordbin={1:0,2:4,3:2,4:6};wordvector={1:(1,0),2:(-1,0),3:(0,1),4:(0,-1)}
 for i,r in enumerate(ds.records):
  ids,index=ds._predicates[i];stats['all_TRAIN']+=1;stats['no_public_anchor']+=int(not r['landmarks']);stats['multiple_public_polygons']+=int(len(r['landmarks'])>1)
  if index<0:continue
  pid=int(ids[index]);stats['public_text_eligible']+=1;per[str(pid)]+=1
  center=np.asarray(r['landmarks'][index]['contour']).mean(0);delta=(np.asarray(r['target_xy'])-center)*np.asarray(r['physical_map_sizes']);rad=np.linalg.norm(delta);valid=bool(rad>1)
  theta=float(np.arctan2(delta[1],delta[0]));k=int(np.round((theta%(2*math.pi))*8/(2*math.pi)))%8
  half=float(delta@np.asarray(wordvector[pid]))>0;exact=k==wordbin[pid]
  stats['GT_defined_angular_labels']+=valid;stats['engineering_axis_halfplane_matches']+=int(valid and half);stats['engineering_axis_8bin_matches']+=int(valid and exact)
  rows.append(dict(index=i,map_name=r['map_name'],instruction=r['instruction'],public_anchor=r['landmarks'][index]['name'],public_id=r['landmarks'][index]['id'],predicate_id=pid,GT_angle_radians_OFFLINE=theta,GT_angle_bin_OFFLINE=k,valid_label=valid,engineering_halfplane_match=half,engineering_8bin_match=exact,used_for_gate=False))
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'eligible_TRAIN_audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));n=stats['GT_defined_angular_labels']
 result=dict(counts=dict(stats),per_predicate=dict(per),public_eligibility_fraction=stats['public_text_eligible']/len(ds),engineering_halfplane_agreement=stats['engineering_axis_halfplane_matches']/max(n,1),engineering_8bin_agreement=stats['engineering_axis_8bin_matches']/max(n,1),geographic_CRS_verified=False,GT_agreement_never_used_for_selection=True,source_sha256=file_sha256(a.source),public_catalog_sha256=file_sha256(a.public_objects))
 atomic_json(result,a.output/'data_audit.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
