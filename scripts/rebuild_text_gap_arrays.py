"""Reconstruct aggregates only from CRC-verified original GPU batch archives."""
import argparse,json,zipfile,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);meta=json.load(open(root/'run.json'));n=meta['N'];arms=meta['arms'];paths=sorted((root/'batches').glob('*.npz'));assert len(paths)==(n+15)//16
 out=root/'checked_arrays';out.mkdir(exist_ok=True);arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype=np.float32,shape=(n,784)) for k in arms}
 for bi,p in enumerate(paths):
  assert p.stem==f'{bi:04d}'
  with zipfile.ZipFile(p) as z:assert z.testzip() is None
  batch=np.load(p);assert set(batch.files)==set(arms)
  for k in arms:
   v=batch[k];assert v.shape==(min(16,n-bi*16),784);assert np.isfinite(v).all();arrays[k][bi*16:min(n,bi*16+16)]=v
 checks={}
 for k,v in arrays.items():
  v.flush();p=root/(k+'.npz')
  try:
   with zipfile.ZipFile(p) as z:good=z.testzip() is None
  except zipfile.BadZipFile:good=False
  if good:assert np.array_equal(np.load(p)['logits'],v)
  checks[k]=dict(old_aggregate_crc_valid=good,reconstructed_sha256=hashlib.sha256((out/(k+'.npy')).read_bytes()).hexdigest(),shape=list(v.shape))
 result=dict(N=n,batches=len(paths),all_batch_crc_valid=True,old_aggregates_corrupt=sum(not c['old_aggregate_crc_valid'] for c in checks.values()),all_readable_old_aggregates_exactly_match=True,arrays=checks)
 (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
