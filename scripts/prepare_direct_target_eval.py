"""Frozen text-only Val-Seen eligibility; GT written only to private scoring file."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
import spacy
from graphnav.structured_navigation import parse_instruction
from graphnav.direct_target_language import select_direct_relations
from scripts.analyze_instruction_landmark_association import PublicNames


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Frozen selection cannot be overwritten')
    a.output.mkdir(parents=True)
    path=a.data_root/'processed_citynav/citynav_val_seen.json'
    records=json.loads(path.read_text());catalog=json.loads((a.data_root/'cityrefer/objects.json').read_text())
    public=PublicNames(catalog);nlp=spacy.load('en_core_web_sm',disable=['ner'])
    texts=[(i,j,r['area']+'_block_'+str(r['block']),text) for i,r in enumerate(records) for j,text in enumerate(r['descriptions'])]
    tasks=[];labels=[];ledger=[]
    for (i,j,map_name,text),doc in zip(texts,nlp.pipe((x[-1] for x in texts),batch_size=128)):
        parsed=parse_instruction(text,map_name,lambda _:doc,public)
        spec=select_direct_relations(parsed,catalog[map_name])
        if not spec['relations']:
            ledger.append(dict(episode=i,instruction_index=j,eligible=False,reasons=spec['rejected']));continue
        with rasterio.open(a.data_root/'rgbd'/f'{map_name}.tif') as ds:
            b=ds.bounds;bounds=[b.left,b.bottom,b.right,b.top]
        anchors=[dict(**q,contours=[catalog[map_name][str(oid)]['contour'] for oid in q['landmark']['public_ids']])
                 for q in spec['relations']]
        raw=records[i];pose=raw['trajectory'][0]
        yaw=float(np.arctan2(pose[4],pose[3])) if len(pose)==6 else float(pose[3])
        index=len(tasks)
        tasks.append(dict(index=index,episode=i,instruction_index=j,map_name=map_name,instruction=text,
            structured=spec,anchors=anchors,world_bounds=bounds,start_pose=[*pose[:3],yaw],target_GT_used=False))
        # Private side channel only; never supplied to the runtime constructor.
        labels.append(dict(index=index,target=raw['target_positions'][j][:2],reference=raw['trajectory']))
        ledger.append(dict(episode=i,instruction_index=j,eligible=True,index=index))
    order=sorted(range(len(tasks)),key=lambda i:hashlib.sha256(('42'+tasks[i]['map_name']+tasks[i]['instruction']).encode()).hexdigest())
    summary=dict(source='revised Val-Seen',source_records=len(records),source_instructions=len(texts),eligible=len(tasks),
        pilot_indices=order[:32],selection_GT_used=False,parser='existing spaCy dependencies + strict scope guard',
        relation_counts=dict(Counter(q['predicate'] for t in tasks for q in t['anchors'])),
        original_text_retained=True,normal_target_GT_inputs=False,
        selection_code_sha256=hashlib.sha256(Path('graphnav/direct_target_language.py').read_bytes()).hexdigest())
    for name,content in [('tasks_public.json',tasks),('labels_private.json',labels),('selection_ledger.json',ledger),('selection_manifest.json',summary)]:
        (a.output/name).write_text(json.dumps(content,ensure_ascii=False,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
