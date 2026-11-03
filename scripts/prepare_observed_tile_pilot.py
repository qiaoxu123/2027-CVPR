"""Freeze 24 fit-only queries, separating public image inputs and private scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.retarget_training_data import select_training_query
from scripts.analyze_instruction_landmark_association import PublicNames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hard-cars',action='store_true',help='12 A0-error-selected fit diagnostics; never a representative benchmark')
    for key in ('root','catalog','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve selected pilot')
    r=a.root;obs=r/'fixed_yaw_train400_s1';cache=r/'chain_train400_features'
    manifest=json.loads((cache/'manifest.json').read_text())
    allowed=set(json.loads((r/'public_deployment_filter.json').read_text())['frame_keys'])
    truth={x['frame_key']:x for x in map(json.loads,(obs/'labels_fit.jsonl').read_text().splitlines()) if x['status']=='captured'}
    frames={x['frame_key']:x for x in map(json.loads,(obs/'public_frames.jsonl').read_text().splitlines())}
    review_path=Path('reports/landmark_chain_visual/retarget_training/strict_review12/ai_review.json')
    review=json.loads(review_path.read_text())['rows']
    proposals=json.loads((r/'strict_retarget_support_v1/links_private.json').read_text())
    links={(x['frame_key'],str(x['private_training_instance_id'])):x for x in proposals}
    query_inputs=torch.load(r/'retarget_public_inputs_v1/public_inputs.pt',map_location='cpu',weights_only=True)
    retarget_tasks={x['task_key']:x for x in map(json.loads,(r/'retarget_training_data_v1/public_tasks.jsonl').read_text().splitlines())}
    private_retarget=torch.load(r/'retarget_training_data_v1/loss_labels_private.pt',map_location='cpu',weights_only=True)
    selected=[];private={};seen=set()
    for entry in review:
        frame=entry['frame_key'];link=links[(frame,str(entry['private_training_instance_id']))]
        query=f"retarget_{select_training_query(link['available_train_query_indices'],entry['map_name'],str(entry['private_training_instance_id'])):05d}"
        case=frame+'__'+query
        selected.append(dict(case_key=case,source_frame_key=frame,task_key=query,map_name=entry['map_name'],instruction=retarget_tasks[query]['instruction']))
        private[case]=dict(category='Car',target_world=private_retarget[case]['target_world'],physical_group=private_retarget[case]['physical_group'],
            label_quality='AI-supported geometry proxy' if entry['candidate_same_instance_support']=='supported' else 'uncertain geometry proxy',human_pixel_gt=False)
        seen.add(frame)
    for category in ('Building','Parking','Ground'):
        candidates=[x for k,x in truth.items() if k in allowed and k not in seen and x['category']==category]
        candidates.sort(key=lambda x:hashlib.sha256(('tile-fit24-seed0:'+x['frame_key']).encode()).hexdigest())
        groups=set();count=0
        for label in candidates:
            group=tuple(label['physical_group'])
            if group in groups:continue
            groups.add(group);frame=label['frame_key'];f=frames[frame];case=frame+'__'+label['task_key']
            selected.append(dict(case_key=case,source_frame_key=frame,task_key=label['task_key'],map_name=f['map_name'],instruction=f['instruction']))
            private[case]=dict(category=category,target_world=label['target_world'][:2],physical_group=label['physical_group'],label_quality='world-position evaluation only',human_pixel_gt=False)
            seen.add(frame);count+=1
            if count==4:break
        if count!=4:raise ValueError('not enough distinct fit targets')
    if a.hard_cars:
        excluded_groups={tuple(v['physical_group']) for v in private.values()};excluded_frames=set(seen)
        selected=[];private={};seen=set();chosen_groups=set()
        candidates=json.loads((r/'retarget_training_data_v1/public_links.json').read_text())
        candidates.sort(key=lambda x:hashlib.sha256(('tile-hard12-seed0:'+x['sample_key']).encode()).hexdigest())
        for link in candidates:
            target=private_retarget[link['sample_key']];group=tuple(target['physical_group']);q=query_inputs[link['task_key']]
            bounds=torch.tensor(q['world_bounds']);choice=int(q['prior_logits'].argmax())
            xy=bounds[:2]+torch.tensor([(choice%28+.5)/28,(choice//28+.5)/28])*(bounds[2:]-bounds[:2])
            error=float((xy-torch.tensor(target['target_world'])).norm())
            frame=link['source_frame_key']
            if error<=20 or group in excluded_groups or group in chosen_groups or frame in excluded_frames or frame in seen:continue
            case=link['sample_key'];task=retarget_tasks[link['task_key']]
            selected.append(dict(case_key=case,source_frame_key=frame,task_key=link['task_key'],map_name=task['map_name'],instruction=task['instruction']))
            private[case]=dict(category='Car',target_world=target['target_world'],physical_group=list(group),
                label_quality='unreviewed geometry proxy; A0-error-selected fit diagnostic',human_pixel_gt=False,selection_a0_ne_m=error)
            chosen_groups.add(group);seen.add(frame)
            if len(selected)==12:break
    expected=12 if a.hard_cars else 24
    assert len(selected)==expected and len(seen)==expected
    if any(manifest['map_assignment'][x['map_name']]!='fit' for x in selected):raise ValueError('fit-only pilot')
    source={}
    for chunk in manifest['chunks']:
        path=cache/chunk['file']
        if file_sha256(path)!=chunk['sha256']:raise ValueError('feature cache changed')
        for row in torch.load(path,map_location='cpu',weights_only=True)['records']:
            if row['frame_key'] in seen:source[row['frame_key']]=row
    original_queries={(x['map_name'],x['instruction']):x for x in map(json.loads,(cache/'public_queries.jsonl').read_text().splitlines())}
    roles=torch.load(r/'reference_vectors_v2.pt',map_location='cpu',weights_only=True)
    catalog=json.loads(a.catalog.read_text());names=PublicNames(catalog);inputs={}
    for task in selected:
        row=source[task['source_frame_key']];frame=frames[task['source_frame_key']]
        if task['task_key'].startswith('retarget_'):
            query=query_inputs[task['task_key']]
            fields={k:query[k] for k in ('prior_logits','target_text','context_text','world_bounds','public_entities')}
        else:
            query=original_queries[(task['map_name'],task['instruction'])]
            ids=list(dict.fromkeys(oid for m in names.match(task['map_name'],task['instruction']) for oid in m['public_ids']))[:16]
            fields=dict(prior_logits=row['prior_logits'],target_text=row['target_text'],context_text=roles[task['task_key']]['reference_vector'],world_bounds=query['world_bounds'],
                        public_entities=[dict(kind='region',geometry_m=catalog[task['map_name']][oid]['contour']) for oid in ids])
        inputs[task['case_key']]=dict(**fields,global_world=row['candidate_world'][:784],cached_image_features=row['image_features'],cached_world=row['candidate_world'][784:],
            cached_local_valid=row['local_valid'],cached_quality=row['quality'],frame=frame)
    a.output.mkdir(parents=True)
    atomic_json(selected,a.output/'public_selection.json');atomic_save(inputs,a.output/'public_inputs.pt');atomic_json(private,a.output/'private_scoring.json')
    atomic_json(dict(cases=len(selected),categories=dict(Counter(x['category'] for x in private.values())),all_fit=True,camera_yaw=0,
        selection=('12 new physical Car targets and source frames, hash selected among A0-error>20m fit proposals; explicitly outcome-conditioned diagnostic' if a.hard_cars else '12 previously reviewed fit Car proposals, including uncertain; 4 other targets/category by hash; no new outcome filtering'),
        warning='Biased exploratory fit pilot, no human pixel GT; no generalization or SR claim',
        public_selection_sha256=file_sha256(a.output/'public_selection.json'),public_inputs_sha256=file_sha256(a.output/'public_inputs.pt'),
        private_scoring_sha256=file_sha256(a.output/'private_scoring.json'),review_sha256=file_sha256(review_path),feature_manifest_sha256=file_sha256(cache/'manifest.json'),
        producer_sha256=file_sha256(__file__)),a.output/'manifest.json')
    print(json.dumps(dict(cases=len(selected),public_input_fields=list(next(iter(inputs.values()))))))


if __name__=='__main__':main()
