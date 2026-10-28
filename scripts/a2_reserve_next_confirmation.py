"""Reserve new untouched TRAIN-source targets without collecting observations."""
from collections import defaultdict,Counter
import hashlib,json,re
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.revision_confirmation import immutable_json


def main():
    source=json.loads((BASE/'source_manifest.json').read_text());by_index={e['index']:e for e in source['entries']}
    excluded={(e['map_name'],e['object_id']) for e in json.loads((BASE/'revision_confirmation_reserve.json').read_text())['targets']}
    previous=len(excluded);files=0;unrecognized=[]
    for directory in BASE.glob('cache*'):
        if not directory.is_dir() or directory.name=='cache_query':continue
        for path in directory.rglob('*.pt'):
            match=re.fullmatch(r'(train|dev|calibration)_(\d+)_(\d+)\.pt',path.name)
            if match is None:unrecognized.append(str(path));continue
            e=by_index[int(match[2])];excluded.add((e['map_name'],e['object_id']));files+=1
    if unrecognized:raise ValueError('Unknown old observed-view filename; inspect before reserving')
    groups=defaultdict(list)
    for entry in source['entries']:
        key=(entry['map_name'],entry['object_id'])
        if entry['split']!='train' and key not in excluded:groups[key].append(entry['index'])
    chosen=sorted(groups,key=lambda g:hashlib.sha256(f'a2-next-confirmation:0:{g[0]}:{g[1]}'.encode()).digest())[:256]
    if len(chosen)!=256:raise ValueError('Insufficient untouched targets')
    report=dict(status='reserved_before_next_revision_labels_or_training_no_observations',seed=0,
        source_manifest_sha256=sha(BASE/'source_manifest.json'),previous_reserve_sha256=sha(BASE/'revision_confirmation_reserve.json'),
        old_view_files_scanned=files,previous_reserved_targets=previous,available_unused_targets=len(groups),
        selection='Hash first256 originaldev/calibration TRAIN-source groups excluding all prior observed views and v1reserved256; no performance/coverage selection',
        targets=[dict(map_name=m,object_id=o,episode_indices=groups[(m,o)]) for m,o in chosen],no_observation_collection_started=True)
    immutable_json(report,BASE/'revision_v2_confirmation_reserve.json')
    save(report,ROOT/'reports/a2_execution_20261027/REVISION_V2_CONFIRMATION_RESERVE.json')
    print(json.dumps({k:v for k,v in report.items() if k!='targets'}),flush=True)


if __name__=='__main__':main()
