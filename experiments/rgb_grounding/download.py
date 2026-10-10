"""Bounded, SHA-recorded downloads. Try direct, then configured proxy."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests


def download(spec, root):
    destination = root/spec['name']; destination.parent.mkdir(parents=True, exist_ok=True)
    record = dict(spec, attempts=[])
    if destination.exists():
        raise ValueError('refusing unverified overwrite: '+str(destination))
    partial = destination.with_suffix(destination.suffix+'.partial')
    for proxy in (False, True):
        started=time.time()
        try:
            session=requests.Session();session.trust_env=proxy
            headers={}
            if spec.get('authenticated'):
                from urllib.parse import urlparse
                from huggingface_hub import get_token
                if urlparse(spec['url']).hostname!='huggingface.co':
                    raise ValueError('authenticated download must use official HF origin')
                credential=get_token()
                if not credential:raise ValueError('HF login required')
                headers['Authorization']='Bearer '+credential
            response=session.get(spec['url'],headers=headers,stream=True,timeout=(8,30))
            response.raise_for_status()
            limit=int(spec.get('limit_gb',8)*1024**3)
            if int(response.headers.get('Content-Length',0))>limit:
                raise ValueError('download budget exceeded')
            sha=hashlib.sha256();size=0;last=time.time()
            with partial.open('wb') as handle:
                for chunk in response.iter_content(4*1024**2):
                    if not chunk:continue
                    size+=len(chunk)
                    if size>limit:raise ValueError('download budget exceeded')
                    handle.write(chunk);sha.update(chunk)
                    if time.time()-last>20:
                        print(json.dumps(dict(name=spec['name'],proxy=proxy,bytes=size)),flush=True);last=time.time()
            partial.rename(destination)
            record.update(success=True,bytes=size,sha256=sha.hexdigest(),seconds=time.time()-started,
                          actual_proxy=proxy,path=str(destination))
            break
        except (requests.RequestException,ValueError) as error:
            record['attempts'].append(dict(proxy=proxy,error=type(error).__name__,seconds=time.time()-started))
            print(json.dumps(dict(name=spec['name'],proxy=proxy,error=type(error).__name__)),flush=True)
    else:record['success']=False
    (destination.with_suffix(destination.suffix+'.provenance.json')).write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);parser.add_argument('--root',required=True);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    specs=json.loads(Path(args.spec).read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(lambda spec:download(spec,root),specs))
    batches=root/'download_batches';batches.mkdir(exist_ok=True)
    identity=hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()[:16]
    (batches/(identity+'.json')).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
