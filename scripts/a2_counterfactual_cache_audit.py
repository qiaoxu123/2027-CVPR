"""TRAIN-only audit for genuine cached same-observation/different-target queries."""
from collections import defaultdict
import json
import numpy as np
from experiments.a2_waldo_landmark_belief.paths import DATA,ROOT,bootstrap,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.revision_data import RevisionViews
bootstrap()
from sbfnav.dataset import CityNavDataset


def main():
    views=RevisionViews(BASE/'collected_snapshot','train');dataset=CityNavDataset(DATA,'train_seen');groups=defaultdict(list)
    for i,entry in enumerate(views.entries):
        record=dataset[entry['episode']];pose=np.asarray(record.trajectory[entry['step']])
        key=(record.map_name,tuple(np.round(pose,5)))
        groups[key].append((i,record.object_id,np.asarray(record.target_position[:2]),record.instruction))
    pairs=[]
    for rows in groups.values():
        for i,left in enumerate(rows):
            for right in rows[i+1:]:
                if left[1]!=right[1] and left[3]!=right[3] and np.linalg.norm(left[2]-right[2])>40:
                    pairs.append([left[0],right[0]])
    report=dict(scope='TRAIN cached teacher views only; offline audit, no model or GPU',views=len(views),
        same_pose_groups=sum(len(v)>1 for v in groups.values()),distinct_target_pair_candidates=len(pairs),pairs=pairs,
        source_annotation_sha256=sha(dataset.annotation_path),cache_manifest_sha256=sha(BASE/'collected_snapshot/manifest.json'),
        criteria='same map+trajectory4D pose rounded5 decimals, distinct genuine source instruction and target object, target separation>40m; positive pairs would additionally require cached sensor/box equality before inference',
        conclusion='No valid cached pair found; corresponding-target counterfactual diagnostic remains pending' if not pairs else 'Candidate pairs require actual cached public observation equality verification before counterfactual inference')
    save(report,ROOT/'reports/a2_execution_20261027/REVISION_COUNTERFACTUAL_AUDIT.json');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
