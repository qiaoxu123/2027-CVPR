"""Retrieve authoritative baseline papers and licenses; store hashes and table evidence."""
from common import *
import concurrent.futures,requests,fitz

URLS={
 'citynav_paper':'https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.pdf',
 'htnav_paper':'https://openaccess.thecvf.com/content/CVPR2026/papers/Fan_HTNav_A_Hybrid_Navigation_Framework_with_Tiered_Structure_for_Urban_CVPR_2026_paper.pdf',
 'hett_paper':'https://ojs.aaai.org/index.php/AAAI/article/download/38885/42847',
 'flightgpt_paper':'https://aclanthology.org/2025.emnlp-main.338.pdf',
 'citynav_license':'https://raw.githubusercontent.com/water-cookie/citynav/main/README.md',
 'sensaturban_readme':'https://raw.githubusercontent.com/QingyongHu/SensatUrban/master/README.md',
 'sensaturban_license':'https://raw.githubusercontent.com/QingyongHu/SensatUrban/master/LICENSE',
}

def fetch(item):
    key,url=item
    dest=OUT/'local_only/source_downloads'/key
    dest.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not dest.exists():
            r=requests.get(url,timeout=60);r.raise_for_status();dest.write_bytes(r.content)
        data=dest.read_bytes();result=dict(key=key,url=url,status='Retrieved',sha256=sha(dest),bytes=len(data),local_path=str(dest))
        if data.startswith(b'%PDF'):
            doc=fitz.open(stream=data,filetype='pdf')
            pages=[(i+1,p.get_text()) for i,p in enumerate(doc)]
            # Full PDFs and full text remain local; publish numeric transcriptions separately.
            (dest.with_suffix('.txt')).write_text('\n'.join(f'PAGE {i}\n{t}' for i,t in pages))
            result['pages']=len(pages)
            result['table_pages']=[i for i,t in pages if ('Table 1:' in t or 'Table 2:' in t)]
        return result
    except Exception as e:return dict(key=key,url=url,status='Blocked',reason=str(e))

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,URLS.items()))
    dump(OUT/'sources/primary_source_manifest.json',rows)
    for r in rows:print(r['key'],r['status'],r.get('table_pages',[]))

if __name__=='__main__':main()
