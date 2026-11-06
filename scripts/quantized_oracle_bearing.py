"""Supplemental offline8-bin sensitivity, no model/weight/closedloop changes."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from scripts.oracle_bearing_geometry import rerank


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 rows=json.loads((a.run/'offline_rows.json').read_text());fields=torch.load(a.run/'offline_fields.pt',weights_only=False);groups=defaultdict(lambda:defaultdict(list))
 for row in rows:
  f=fields[str(row['index'])];angles=np.asarray(f['angles']);quantized=np.round(angles/(np.pi/4))*(np.pi/4)
  for kind,t in [('continuous',angles),('quantized8',quantized)]:
   z=rerank(torch.tensor(f['scores']['A']),f['grid'],f['anchors'] if len(t) else [],t,'D',1.,float(np.linalg.norm((f['bounds'][2:]-f['bounds'][:2])/28)))
   ne=float(np.linalg.norm(f['grid'][int(z.argmax())]-f['target']))
   for g in ['all',row['group']]:groups[g][kind].append(ne)
 summary={g:{k:dict(N=len(v),R20=float(np.mean(np.asarray(v)<=20)),NE=float(np.mean(v))) for k,v in methods.items()} for g,methods in groups.items()}
 (a.run/'quantized8_diagnostic.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
