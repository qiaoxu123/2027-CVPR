"""TRAIN public inputs rebuilt under the frozen scoped name resolver; no private GT reads."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from graphnav.a3_runtime import public_static_record
from graphnav.public_landmark_mentions import PublicLandmarkMentions,add_mention_aliases
from graphnav.checkpointing import atomic_json,file_sha256


def main():
 p=argparse.ArgumentParser()
 for k in ['source-data','catalog','parser-freeze','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'manifest.json').exists():raise FileExistsError('Preserve prepared data')
 freeze=json.load(open(a.parser_freeze));sha=file_sha256(Path('graphnav/public_landmark_mentions.py'));assert freeze['parser_sha256']==sha
 objects=json.load(open(a.catalog));assert file_sha256(a.catalog)==freeze['catalog_sha256'];matcher=PublicLandmarkMentions(objects);catalog=SimpleNamespace(objects=objects);source=[json.loads(s) for s in (a.source_data/'public_train.jsonl').read_text().splitlines()];assert len(source)==21878;counts=Counter();rows=[]
 for i,q in enumerate(source):
  mentions=matcher.match(q['map_name'],q['instruction']);refs=tuple(dict.fromkeys(n for m in mentions for n in m['public_names']));b=q['world_bounds'];t=SimpleNamespace(x_min=b[0],y_min=b[1],x_max=b[2],y_max=b[3]);episode=SimpleNamespace(map_name=q['map_name'],instruction=q['instruction'],reference_names=refs,references=());r=add_mention_aliases(public_static_record(episode,t,catalog,max_landmarks=16),mentions);assert r['target_xy']==[0.,0.];rows.append(r);counts['no_anchor']+=not r['landmarks'];counts['name_set_changed']+={x['name'] for x in q['landmarks']}!={x['name'] for x in r['landmarks']}
  if (i+1)%2000==0:print('public prepared',i+1,flush=True)
 (a.output/'public_train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));shutil.copy(a.source_data/'labels_private.json',a.output/'labels_private.json');old=json.load(open(a.source_data/'manifest.json'));atomic_json(dict(old,counts_scoped=dict(counts),parser_sha256=sha,public_sha256=file_sha256(a.output/'public_train.jsonl'),source_data_manifest_sha256=file_sha256(a.source_data/'manifest.json'),private_GT_not_read_during_binding=True),a.output/'manifest.json')
if __name__=='__main__':main()
