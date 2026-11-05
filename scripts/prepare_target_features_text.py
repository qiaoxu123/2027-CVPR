"""Reuse verified public-only bindings, change only the instruction string."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.target_features_instruction import content_instruction,VERSION


def prepare(source,original,output,expected=21878):
    if output.exists(): raise FileExistsError(output)
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    native=[json.loads(x) for x in original.read_text().splitlines() if x.strip()]
    if len(rows)!=expected or len(native)!=expected: raise ValueError('Require complete aligned TRAIN')
    converted=[];audit=[];counts=Counter()
    for i,(row,old) in enumerate(zip(rows,native)):
        if (row['instruction'],row['target_xy'],row['world_bounds'],row['map_name']) != (old['instruction'],old['target_xy'],old['world_bounds'],old['map_name']):
            raise ValueError(f'Misaligned source {i}')
        if row['dataset_version']!='revised' or row['match_mode']!='fuzzy': raise ValueError('TRAIN protocol mismatch')
        if any(k in row for k in ('teacher_future','uav_pose')): raise ValueError('Static TRAIN required')
        text=content_instruction(row['instruction']);new=copy.deepcopy(row);new['instruction']=text
        converted.append(new);counts['prefix_removed' if text!=row['instruction'] else 'unchanged']+=1
        counts['no_public_anchor']+=int(not row['landmarks'])
        audit.append(dict(index=i,original=row['instruction'],model_instruction=text,content_preserved=True,GT_used=False))
        assert {k:v for k,v in row.items() if k!='instruction'}=={k:v for k,v in new.items() if k!='instruction'}
    output.mkdir(parents=True)
    dest=output/'target_features_landmark_relation.jsonl'
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in converted))
    (output/'audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audit))
    manifest=dict(rows=len(rows),counts=dict(counts),version=VERSION,source_sha256=file_sha256(source),original_source_sha256=file_sha256(original),output_sha256=file_sha256(dest),GT_used_for_text=False,public_binding='unchanged audited PR23 full_text bindings',text='all original semantic content; remove initial navigation imperative only')
    atomic_json(manifest,output/'manifest.json');return manifest


def main():
    p=argparse.ArgumentParser()
    for n in ('source','original','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.original,a.output),indent=2))
if __name__=='__main__':main()
