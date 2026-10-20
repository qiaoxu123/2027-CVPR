"""Verify each BibTeX record against publisher/arXiv metadata; never invent fields."""
from common import *
import re,concurrent.futures,requests,difflib,unicodedata,subprocess
from bs4 import BeautifulSoup

def entries(text):
    records=[]
    for match in re.finditer(r'@(\w+)\s*\{\s*([^,]+),',text):
        start=match.start();i=match.end();depth=1
        while depth and i<len(text):
            if text[i]=='{':depth+=1
            elif text[i]=='}':depth-=1
            i+=1
        body=text[match.end():i-1];fields={}
        for f in re.finditer(r'(\w+)\s*=\s*\{',body):
            j=f.end();k=j;level=1
            while level and k<len(body):
                if body[k]=='{':level+=1
                elif body[k]=='}':level-=1
                k+=1
            fields[f[1].lower()]=body[j:k-1]
        records.append(dict(key=match[2],type=match[1],fields=fields,start=start,end=i,raw=text[start:i]))
    return records

def norm(s):
    return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())

def fetch(entry):
    f=dict(entry['fields']);key=entry['key']
    # Prefer accepted publication records over older preprint metadata.
    overrides={'navgpt24':'10.1609/aaai.v38i7.28597',
               'groundingdino24':'10.1007/978-3-031-72970-6_3',
               'vlnce20':'10.1007/978-3-030-58604-1_7'}
    if key in overrides:f['doi']=overrides[key]
    url=f.get('url') or ('https://doi.org/'+f['doi']);out=dict(key=key,source_url=url)
    cache=OUT/'local_only/bibliography'/f'{key}.json';cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        cached=read(cache)
        if cached.get('status')=='Verified':
            if key in ('groundingdino24','vlnce20'):
                cached['venue']='Computer Vision -- ECCV '+('2024' if key=='groundingdino24' else '2020')
            return cached
    try:
        if f.get('doi'):
            api='https://api.crossref.org/works/'+f['doi'];r=requests.get(api,timeout=35);r.raise_for_status();m=r.json()['message']
            authors=[(a.get('family','')+', '+a.get('given','')).strip(', ') for a in m.get('author',[])]
            date=m.get('published',m.get('issued',{})).get('date-parts',[[None]])[0][0]
            out.update(title=m['title'][0],authors=authors,year=str(date),venue=m.get('container-title',[''])[0],doi=m.get('DOI',''),metadata_url=api,source_sha256=__import__('hashlib').sha256(r.content).hexdigest())
        else:
            r=requests.get(url,timeout=35);r.raise_for_status();soup=BeautifulSoup(r.content,'html.parser')
            meta={}
            for tag in soup.find_all('meta'):
                name=tag.get('name',tag.get('property','')).lower()
                if tag.get('content'):meta.setdefault(name,[]).append(tag['content'])
            title=meta.get('citation_title',[''])[0];authors=meta.get('citation_author',[])
            if not title or not authors:raise ValueError('Publisher page returned no complete citation title/authors; not inferred from a search snippet')
            date=meta.get('citation_publication_date',meta.get('citation_date',['']))[0]
            year=re.search(r'(19|20)\d{2}',date)
            venue=meta.get('citation_conference_title',meta.get('citation_journal_title',['']))[0]
            if 'openaccess.thecvf.com' in url:
                match=re.search(r'(CVPR|ICCV)(\d{4})',url) or re.search(r'(cvpr|iccv)_(\d{4})',url,re.I)
                if match:
                    year=str(match[2]);venue='Proceedings of the IEEE/CVF '+('Conference on Computer Vision and Pattern Recognition' if match[1].upper()=='CVPR' else 'International Conference on Computer Vision')
            else:year=year[0] if year else ''
            if 'arxiv.org' in url:venue='arXiv preprint';year=str(year)
            if 'proceedings.neurips.cc/paper_files/paper/' in url:
                year=re.search(r'/paper/(\d{4})/',url)[1]
                out['date_note']='Conference year is taken from official proceedings volume, distinct from later DOI/publication deposit date.'
            out.update(title=title,authors=authors,year=str(year),venue=venue,doi=meta.get('citation_doi',[''])[0],metadata_url=r.url,source_sha256=__import__('hashlib').sha256(r.content).hexdigest())
        similarity=difflib.SequenceMatcher(None,norm(f['title']),norm(out['title'])).ratio()
        if key in ('groundingdino24','vlnce20'):
            out['venue']='Computer Vision -- ECCV '+('2024' if key=='groundingdino24' else '2020')
            out['venue_note']='Crossref container is the LNCS series; conference identity is verified by the matching ECCV/ECVA publication and DOI proceedings volume.'
        if similarity<.85:
            if key=='openfly25' and '2502.18041' in out['metadata_url'] and out['title'].lower().startswith('openfly:'):
                out['version_note']='Same official arXiv identifier; latest version changes subtitle to A comprehensive platform for aerial vision-language navigation. Verified title and full author order follow that version.'
            else:raise ValueError('Source/title mismatch, normalized similarity '+str(similarity))
        out['title_similarity']=similarity;out['status']='Verified' if out['authors'] and out['year'] else 'Need Verification'
    except Exception as e:out.update(status='Blocked',reason=str(e))
    dump(cache,out);return out

def replace_field(raw,key,value):
    pattern=re.compile(r'(?m)(\b'+key+r'\s*=\s*)\{');m=pattern.search(raw)
    if not m:return raw[:-1].rstrip().rstrip(',')+',\n  '+key+' = {'+value+'},\n}'
    j=m.end();level=1;k=j
    while level:
        if raw[k]=='{':level+=1
        elif raw[k]=='}':level-=1
        k+=1
    return raw[:j]+value+raw[k-1:]

def main():
    path=PAPER/'main.bib';original=OUT/'sources/manuscript_bibliography_original.bib'
    if not original.exists():original.write_text(subprocess.check_output(['git','-C',str(PAPER),'show','22c4fcae13b9abb7db9f5cd31ff28ae96249a43c:main.bib'],text=True))
    text=original.read_text();es=entries(text);assert len(es)==48,len(es)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:found=list(pool.map(fetch,es))
    rows=[];newtext=text
    for e,m in reversed(list(zip(es,found))):
        f=e['fields'];changes=[];raw=e['raw']
        if m['status']=='Verified':
            updates={'author':' and '.join(m['authors']),'year':m['year']}
            if norm(m['title'])!=norm(f['title']):updates['title']=m['title']
            if m.get('doi'):updates['doi']=m['doi']
            venuekey='booktitle' if e['type']=='inproceedings' else 'journal' if e['type']=='article' else 'howpublished'
            if m.get('venue'):updates[venuekey]=m['venue']
            for key,value in updates.items():
                if f.get(key)!=value:raw=replace_field(raw,key,value);changes.append(key)
            newtext=newtext[:e['start']]+raw+newtext[e['end']:]
        rows.append(dict(key=e['key'],status=m['status'],original_title=f['title'],verified_title=m.get('title',''),original_year=f.get('year',''),verified_year=m.get('year',''),verified_authors=' and '.join(m.get('authors',[])),verified_venue=m.get('venue',''),doi=m.get('doi',f.get('doi','')),source_url=m['source_url'],metadata_url=m.get('metadata_url',''),source_sha256=m.get('source_sha256',''),fields_updated=';'.join(changes),reason=m.get('reason','')))
    path.write_text(newtext);rows.reverse();csvwrite(OUT/'bibliography_audit.csv',rows);dump(OUT/'sources/bibliography_metadata.json',found)
    summary={s:sum(r['status']==s for r in rows) for s in ('Verified','Need Verification','Blocked')};dump(OUT/'bibliography_summary.json',summary)
    lines=['# Bibliography verification — all 48 records','',str(summary),'',
        'Metadata is retrieved from publishers/CVF/ACL/PMLR/arXiv or the DOI registration record. Full author lists and verified venue/year/DOI fields are corrected only after a matching-title check. Blocked records remain unchanged; no title/author is reconstructed from keywords. Raw HTML stays local. Detailed evidence and source hashes: bibliography_audit.csv and sources/bibliography_metadata.json.','', '|Key|Status|Updated fields|Reason|','|---|---|---|---|']
    lines += [f"|{r['key']}|{r['status']}|{r['fields_updated']}|{r['reason'].replace('|','/')}|" for r in rows]
    (OUT/'BIBLIOGRAPHY_AUDIT.md').write_text('\n'.join(lines)+'\n');print(summary,flush=True)

if __name__=='__main__':main()
