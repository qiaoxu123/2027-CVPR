"""Bounded official annotation-archive inspection, never installs simulator."""
import gzip,json,time,zipfile
from pathlib import Path
import requests

def main():
 root=Path('runs/multiview_rgb_grounding/resource_audit');url='https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln';trials=[];start=time.time();archive=root/'aerialvln_annotations.zip';downloaded=False
 for direct in [True,False]:
  s=requests.Session();s.trust_env=not direct
  try:
   with s.get(url,stream=True,timeout=(15,30)) as r:
    trials.append({'direct':direct,'status':r.status_code,'content_type':r.headers.get('Content-Type')})
    if not r.ok:continue
    size=0
    with archive.open('wb') as f:
     for chunk in r.iter_content(1024*1024):
      size+=len(chunk)
      if size>100*1024*1024:raise ValueError('annotation-only size budget exceeded')
      f.write(chunk)
    downloaded=zipfile.is_zipfile(archive)
    if downloaded:break
  except (requests.RequestException,ValueError) as e:trials.append({'direct':direct,'error':type(e).__name__})
 result={'url':url,'trials':trials,'actual_zip_downloaded':downloaded,'seconds':time.time()-start}
 if downloaded:
  with zipfile.ZipFile(archive) as z:
   infos=z.infolist();result['files']=[{'path':i.filename,'bytes':i.file_size} for i in infos];result['archive_bytes']=archive.stat().st_size
   candidates=[i for i in infos if i.filename.endswith(('.json','.json.gz')) and i.file_size<20*1024*1024]
   if candidates:
    i=min(candidates,key=lambda v:v.file_size);content=z.read(i);content=gzip.decompress(content) if i.filename.endswith('.gz') else content;data=json.loads(content)
    (root/'aerialvln_actual_sample.json').write_text(json.dumps(data,indent=2)+'\n')
    result['sample_path']=i.filename;result['sample_schema']=list(data) if isinstance(data,dict) else None
    eps=data.get('episodes',[]) if isinstance(data,dict) else data;result['sample_episode_count']=len(eps) if isinstance(eps,list) else None
    if isinstance(eps,list) and eps and isinstance(eps[0],dict):result['episode_schema']=list(eps[0])
 (root/'aerialvln_archive_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
