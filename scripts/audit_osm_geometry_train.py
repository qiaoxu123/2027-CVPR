"""TRAIN whitelist determined only by original text and public map names."""
import argparse,json,hashlib
from pathlib import Path
from scripts.analyze_instruction_landmark_association import PublicNames
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();path=a.data/'processed_citynav/citynav_train_seen.json';rows=json.loads(path.read_text());assert len(rows)==21878
objects=json.loads((a.data/'cityrefer_original/objects.json').read_text());matcher=PublicNames(objects);kept=[];excluded=[]
for i,r in enumerate(rows):
 m=f"{r['area']}_block_{r['block']}";text=r['descriptions'][0];matches=matcher.match(m,text);entry=dict(index=i,map_name=m,instruction=text,public_mentions=matches)
 if matches:kept.append(entry)
 else:entry['reason']='no_public_named_landmark_text_match';excluded.append(entry)
a.output.mkdir(parents=True,exist_ok=True)
for name,data in [('kept',kept),('excluded',excluded)]:
 (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
summary=dict(seed=42,train_n=len(rows),kept_n=len(kept),excluded_n=len(excluded),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selection='original text + public names only; no GT binding',refined_hett_identity='local Revised count matches; HETT release checksum not yet established',verified_osm_scenes=0,training_started=False)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
