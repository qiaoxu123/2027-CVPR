"""Read official manifests with direct-first networking; no credential output."""
import concurrent.futures
import json
from pathlib import Path
import time
import requests

SOURCES = {
    'refsegrs': 'https://huggingface.co/api/datasets/JessicaYuan/RefSegRS',
    'refdrone': 'https://huggingface.co/api/datasets/sunzc-sunny/RefDrone',
    'aerialvg': 'https://huggingface.co/api/datasets/IPEC-COMMUNITY/AerialVG',
    'georsclip': 'https://huggingface.co/api/models/Zilun/GeoRSCLIP',
    'dior_mirror': 'https://huggingface.co/api/datasets/LittleCollections/DIOR-RSVG',
    'dior_official': 'https://raw.githubusercontent.com/ZhanYang-nwpu/RSVG-pytorch/main/README.md',
    'rrsisd_official': 'https://raw.githubusercontent.com/Lsan2401/RMSIN/main/README.md',
    'skyclip_official': 'https://raw.githubusercontent.com/wangzhecheng/SkyScript/main/README.md',
    'skyclip_license': 'https://raw.githubusercontent.com/wangzhecheng/SkyScript/main/LICENSE',
}


def fetch(item):
    name, url = item
    attempts = []
    for proxy in (False, True):
        started = time.time()
        try:
            session = requests.Session(); session.trust_env = proxy
            response = session.get(url, timeout=(8, 25))
            attempts.append(dict(proxy=proxy, status=response.status_code,
                                 seconds=time.time()-started))
            response.raise_for_status()
            try: payload = response.json()
            except ValueError: payload = response.text
            return name, dict(url=url, attempts=attempts, payload=payload, success=True)
        except requests.RequestException as error:
            attempts.append(dict(proxy=proxy, error=type(error).__name__,
                                 seconds=time.time()-started))
    return name, dict(url=url, attempts=attempts, success=False)


def main():
    root = Path('runs/rgb_language_grounding/source_audit'); root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for name, result in pool.map(fetch, SOURCES.items()):
            (root/(name+'.json')).write_text(json.dumps(result, indent=2))
            payload=result.get('payload',{})
            files=[x['rfilename'] for x in payload.get('siblings',[])] if isinstance(payload,dict) else []
            print(json.dumps(dict(source=name,success=result['success'],files=files[:35],
                sha=payload.get('sha') if isinstance(payload,dict) else None)),flush=True)


if __name__ == '__main__': main()
