"""Create public pose-based TRAIN sampling mask, preserve original caches."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
from graphnav.deployment_observation_filter import observation_in_envelope
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous filter')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.mapdata import GROUND_LEVEL
    path=a.observations/'public_frames.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()];keys=[];counts=Counter()
    for row in rows:
        height=row['agent_pose'][2]-GROUND_LEVEL[row['map_name']]
        keep=observation_in_envelope(row,GROUND_LEVEL[row['map_name']]);counts['total']+=1;counts['kept' if keep else 'excluded']+=1
        if keep:keys.append(row['frame_key'])
        if not keep:counts['below_action_min' if height<10 else 'above_action_max_noninitial']+=1
    result=dict(frame_keys=keys,counts=dict(counts),source_public_frames_sha256=file_sha256(path),min_action_height_m=10.,max_action_height_m=100.,initial_pose_exempt=True,
                warning='Pose-only sampling control, not visibility or collision validation. Original observations are not altered.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result['counts']))


if __name__=='__main__':main()
