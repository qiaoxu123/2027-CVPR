"""Export private TRAIN supervision after public predictions were sealed.

Do not run confirmation export until the development decision is frozen.
This tool is not imported by any inference module.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
from graphnav.checkpointing import file_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--structured',type=Path,required=True);p.add_argument('--subset',choices=['fit_dev','confirm'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.directory;seal=json.loads((o/'prediction_seal.json').read_text());assert file_sha256(o/'predictions.npz')==seal['prediction_sha256'];assert not a.output.exists()
 if a.subset=='confirm':
  frozen=json.loads((o/'development_decision.json').read_text());assert frozen['final_strategy_frozen'] is True
 raw=json.loads(a.annotations.read_text())
 with gzip.open(a.structured,'rt') as f:parsed=json.load(f)
 assert len(raw)==len(parsed)==22002
 assert all(row['descriptions'][0]==pr['original_text'] and f"{row['area']}_block_{row['block']}"==pr['map_name'] and pr['episode']==i for i,(row,pr) in enumerate(zip(raw,parsed)))
 (o/'original_text_verification.json').write_text(json.dumps(dict(original_instructions=22002,exact_text_and_map_matches=22002,annotations_sha256=file_sha256(a.annotations),structured_sha256=file_sha256(a.structured),scope='All original instruction strings, not a 154-row subset. Destination coordinates not involved in this text comparison.'),indent=2))
 rows=json.loads((o/'public.json').read_text());selected=[r for r in rows if (r['subset']=='confirm')==(a.subset=='confirm')];labels=[]
 for r in selected:
  assert r['instruction']==raw[r['episode']]['descriptions'][0]
  labels.append(dict(index=r['index'],episode=r['episode'],subset=r['subset'],target_world=raw[r['episode']]['target_positions'][-1],target_ids=[str(x) for x in raw[r['episode']]['object_ids']]))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(labels,indent=2));print('Exported private',a.subset,len(labels),'labels; never used by inference.')
if __name__=='__main__':main()
