"""Full TRAIN/dev label audit; never decode a calibration source row."""
from collections import Counter
import json,time
import torch
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.relation_v3_data import RelationV3Views


def main():
    torch.set_num_threads(2);start=time.perf_counter();results={};roles_sha=sha(BASE/'revision_v3/roles.json')
    code={n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in ('relation_v3_data.py','relation_v3_fusion.py','core.py')}
    for role in ('train','dev'):
        dataset=RelationV3Views(role);counts=Counter();sources={}
        if dataset.roles_sha256!=roles_sha:raise ValueError('Roles changed between TRAIN/dev audit')
        for i in range(len(dataset)):
            row=dataset[i];labels=row['labels'];x=row['public']['network'];spatial=row['public']['spatial'];entry=dataset.entries[i]
            if labels['identity_label_mask'].any():raise ValueError('Unknown/heldout identity became supervision')
            if row['target_instance']!=[entry['map_name'],entry['object_id']]:raise ValueError('Label target changed')
            if x['box_rgb'].shape[-1]!=1024 or x['instruction_tokens'].shape[-1]!=768 or spatial['grid_world'].shape!=(784,2):raise ValueError('Frozen feature/grid configuration changed')
            if any(not torch.isfinite(v).all() for v in labels.values()):raise ValueError('Nonfinite private label')
            positive=labels['localization_utility']&labels['utility_valid'];negative=~labels['localization_utility']&labels['utility_valid']
            source=entry['source_kind'];s=sources.setdefault(source,Counter())
            c=dict(views=1,valid_candidates=int(labels['utility_valid'].sum()),positive_candidates=int(positive.sum()),negative_candidates=int(negative.sum()),
                positive_views=int(positive.any()),contrast_views=int(positive.any() and negative.any()),
                same_category_negative_candidates=int((negative&(x['box_kind']==labels['target_kind'])).sum()),
                geometric_positive_candidates=int(labels['geometric_utility'].sum()),unknown_target_kind_views=int(labels['target_kind']<0),
                valid_reference_labels=int(labels['angle_valid'].sum()),identity_labels_used=0)
            counts.update(c);s.update(c)
            if (i+1)%2000==0:print(json.dumps(dict(role=role,completed=i+1,total=len(dataset),seconds=time.perf_counter()-start)),flush=True)
        if counts['views']!=len(dataset):raise ValueError('Partial audit cannot be complete')
        results[role]=dict(counts=dict(counts),by_source={k:dict(v) for k,v in sources.items()},targets=len(dataset.allowed),objects_sha256=dataset.objects_sha256)
    if any(sha(ROOT/'experiments/a2_waldo_landmark_belief'/n)!=h for n,h in code.items()):raise ValueError('Label code changed during audit')
    if sha(BASE/'revision_v3/roles.json')!=roles_sha:raise ValueError('Roles changed during full audit')
    report=dict(status='all_v3_TRAIN_dev_rows_sha_and_labels_audited',roles_sha256=roles_sha,label_code_sha256=code,
        counts=results,calibration_tensor_deserializations=0,identity_negative_labels_used=False,
        supervision='Spatialkernelpeak<=20m plus targetcategory compatibility; TRAIN/dev owncoordinates only; NOT objectidentity assertion',seconds=time.perf_counter()-start)
    save(report,BASE/'revision_v3/data_audit.json');save(report,ROOT/'reports/a2_execution_20261027/RELATION_V3_DATA_AUDIT.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
