"""Fetch official metadata directly first; retain raw responses locally."""
import base64, concurrent.futures, hashlib, json, pathlib, subprocess, time
import requests

REPOS = ['xuefanfu/3DG-VLN','wangkunyu241/SkyFind','UAV-AVL/Benchmark',
         'AirVLN/AirVLN','prince687028/TravelUAV','QwenLM/Qwen3-VL',
         'IDEA-Research/Grounded-SAM-2','QingyongHu/SensatUrban','water-cookie/citynav']

def fetch(url):
    trials=[]
    for direct in [True,False]:
        s=requests.Session();s.trust_env=not direct
        try:
            r=s.get(url,timeout=20);trials.append({'direct':direct,'status':r.status_code})
            if r.ok:return r,trials
        except requests.RequestException as e:trials.append({'direct':direct,'error':type(e).__name__})
    return None,trials

def main():
    root=pathlib.Path('runs/multiview_rgb_grounding/resource_audit');root.mkdir(parents=True,exist_ok=True)
    start=time.time()
    def audit(repo):
        key=repo.replace('/','__');result={'repo':repo,'url':'https://github.com/'+repo}
        for endpoint in ['', '/commits?per_page=1','/readme','/license']:
            p=subprocess.run(['gh','api','repos/'+repo+endpoint],capture_output=True)
            if p.returncode:result[endpoint or 'metadata']={'available':False};continue
            raw=p.stdout;(root/(key+'__'+(endpoint.split('?')[0].replace('/','') or 'metadata')+'.json')).write_bytes(raw)
            data=json.loads(raw)
            if endpoint=='':result.update(default_branch=data['default_branch'],code_license=(data.get('license') or {}).get('spdx_id'))
            elif endpoint.startswith('/commits'):result['commit']=data[0]['sha']
            else:
                content=base64.b64decode(data.get('content',''));(root/(key+endpoint.replace('/','__')+'.txt')).write_bytes(content)
                result[endpoint[1:]]={'path':data.get('path'),'sha256':hashlib.sha256(content).hexdigest(),'available':True}
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:repos=list(pool.map(audit,REPOS))
    endpoints={
        'skyfind_metadata':'https://huggingface.co/api/datasets/kunyu241/SkyFind',
        'skyfind_sample':'https://datasets-server.huggingface.co/first-rows?dataset=kunyu241/SkyFind&config=default&split=train',
        'qwen4b_metadata':'https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct',
        'siglip_metadata':'https://huggingface.co/api/models/google/siglip2-large-patch16-384'}
    checks={}
    for name,url in endpoints.items():
        r,trials=fetch(url);checks[name]={'url':url,'trials':trials,'available':r is not None}
        if r is not None:
            (root/(name+'.json')).write_bytes(r.content);checks[name]['sha256']=hashlib.sha256(r.content).hexdigest()
            data=r.json()
            if name=='skyfind_sample':
                rows=data.get('rows',[]);checks[name]['rows']=len(rows)
                if rows:
                    row=rows[0]['row'];checks[name]['columns']=list(row)
                    checks[name]['annotation_sample']={k:v for k,v in row.items() if k!='image'}
                    im=row.get('image',{});im=im[0] if isinstance(im,list) and im else im
                    if isinstance(im,dict) and im.get('src'):
                        image,attempts=fetch(im['src']);checks[name]['image_download_trials']=attempts
                        if image is not None:
                            (root/'skyfind_sample_image').write_bytes(image.content)
                            checks[name]['image_bytes']=len(image.content)
            else:checks[name].update(revision=data.get('sha'),gated=data.get('gated'),card_license=data.get('cardData',{}).get('license'))
    (root/'audit.json').write_text(json.dumps({'repositories':repos,'http_checks':checks,'seconds':time.time()-start},indent=2)+'\n')
    print(json.dumps({'repositories':len(repos),'http':{k:v['available'] for k,v in checks.items()},'seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
