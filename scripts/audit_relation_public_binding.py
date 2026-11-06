"""Recompute every cached reference from public text; no target fields accessed."""
import argparse,json
from pathlib import Path
from scripts.analyze_instruction_landmark_association import PublicNames
from graphnav.unified_text_ablation import public_text_record
from graphnav.checkpointing import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--data-manifest',type=Path,required=True);p.add_argument('--public-objects',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=json.loads(a.data_manifest.read_text())['records'];catalog=json.loads(a.public_objects.read_text());matcher=PublicNames(catalog);bad=[]
 for i,row in enumerate(rows):
  ref,_=public_text_record(row['map_name'],row['instruction'],row['world_bounds'],catalog,matcher,mode='full_text',max_landmarks=16)
  if ref['landmarks']!=row['landmarks']:bad.append(i)
 result=dict(N=len(rows),public_binding_mismatches=bad,target_coordinates_or_ID_used=False,source_SHA=file_sha256(a.data_manifest),catalog_SHA=file_sha256(a.public_objects));a.output.write_text(json.dumps(result,indent=2));assert not bad,'Public binding mismatch'
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
