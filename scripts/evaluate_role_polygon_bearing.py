"""Frozen confirmation evaluation; no model/epoch/parameter selection here."""
import argparse,json
from pathlib import Path
import torch
from scripts.train_role_polygon_bearing import PublicBatch,evaluate
from graphnav.polygon_neighborhood_bearing import NeighborhoodBearingHead
from graphnav.relation_bearing_network import RolePolygonBearing
from graphnav.checkpointing import file_sha256,atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 lock=a.run/'FROZEN_BEFORE_CONFIRM.json'
 if not lock.exists():raise ValueError('Freeze every model before confirmation')
 frozen=json.loads(lock.read_text())
 for path,sha in frozen['code'].items():assert file_sha256(path)==sha
 for spec in frozen['models'].values():assert file_sha256(spec['path'])==spec['SHA256']
 output=a.run/'confirmation'
 if output.exists():raise FileExistsError('Confirmation already accessed: preserve outputs, do not rerun for tuning')
 output.mkdir();atomic_json(dict(first_confirmation=True,frozen_manifest_SHA=file_sha256(lock)),output/'OPENED.json')
 ids=json.loads((a.run/'data/split.json').read_text())['indices']['confirm'];labels=torch.load(a.run/'PRIVATE_confirm_labels.pt',weights_only=True,map_location='cpu');cache=PublicBatch(a.run);results={}
 for name,spec in frozen['models'].items():
  state=torch.load(spec['path'],weights_only=True,map_location='cpu');arm=spec['model_arm'];model=(NeighborhoodBearingHead() if arm=='old' else RolePolygonBearing()).cuda().eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True)
  report,rows=evaluate(model,cache,ids,labels,arm,16);results[name]=report
  with (output/(name+'_rows.jsonl')).open('w') as f:
   for q in rows:f.write(json.dumps(q)+'\n')
  print(name,json.dumps(report),flush=True);del model;torch.cuda.empty_cache()
 atomic_json(dict(frozen=frozen,results=results),output/'summary.json')
if __name__=='__main__':main()
