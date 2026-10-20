"""Record primary licensing sources, manuscript/source identities and asset gaps.

Only public license/readme metadata is requested. No dataset/weight download,
form submission, credentials, or model evaluation.
"""
import base64
import re
import subprocess
from html.parser import HTMLParser
import requests
from common import *


class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=False
    def handle_starttag(self,t,a):
        if t in ['script','style']:self.skip=True
    def handle_endtag(self,t):
        if t in ['script','style']:self.skip=False
    def handle_data(self,d):
        if not self.skip:self.parts.append(d)


def main():
    p=FINAL/'local_only/license_sources';p.mkdir(parents=True,exist_ok=True)
    urls={
        'citynav':'https://github.com/water-cookie/citynav',
        'sensaturban_code':'https://github.com/QingyongHu/SensatUrban',
        'sensaturban_terms':'https://competitions.codalab.org/competitions/31519',
        'sensaturban_download_form':'https://docs.google.com/forms/d/e/1FAIpQLSfUJVpZ4aBTXwSVhHa5xWaMQ9jeB9EGvo2TsnFovEAykz2eEQ/viewform',
        'CC_BY_4':'https://creativecommons.org/licenses/by/4.0/',
        'CC_BY_NC_SA_4':'https://creativecommons.org/licenses/by-nc-sa/4.0/'}
    sources=[]
    for key,url in urls.items():
        r=requests.get(url,timeout=30);r.raise_for_status();file=p/(key+'.html');file.write_bytes(r.content)
        parser=Text();parser.feed(r.text);plain=' '.join(parser.parts)
        # CodaLab embeds Terms HTML as unicode-escaped content in JavaScript.
        decoded=re.sub(r'\\u([0-9a-fA-F]{4})',lambda m:chr(int(m[1],16)),r.text)
        checks={
            'citynav_CC_BY_4_notice':bool(re.search(r'CityNav Dataset\s*:\s*CC BY 4.0',plain)),
            'sensaturban_NC_SA_notice':'Creative Commons Attribution-NonCommercial-ShareAlike' in decoded,
            'form_nonredistribution_condition':'I will not redistribute the dataset in any way and in any format.' in plain,
            'form_research_only_condition':'non-commercial research and academic purposes' in plain}
        sources.append({'key':key,'url':url,'resolved_url':r.url,'retrieved_date':'2026-10-20',
            'status':r.status_code,'response_sha256':sha(file),'checks':checks,
            'raw_response_retained_local_only':str(file.relative_to(FINAL))})
        print('license',key,checks,flush=True)
    dump(FINAL/'LICENSE_SOURCE_AUDIT.json',{'sources':sources,
        'decision':'RGB kept local_only; no unrestricted dataset/screenshot redistribution or CVF publication-rights clearance asserted.',
        'reason':'CityNav annotation CC BY4 does not erase underlying SensatUrban terms. Dataset terms indicate NC-SA; current download form adds research/nonredistribution conditions. Synthetic-view screenshots are derived from the same data and are not automatically exempt.',
        'conditional_scope':'Academic/noncommercial adaptation is described by CC BY-NC-SA, with attribution/change notice/share-alike duties; the download-form condition and publisher-specific display/redistribution rights need clarification. No consent form submitted.'})
    paper=Path('/home/ubuntu5/Workspace/2027-Qiaoxu-CVPR')
    dump(FINAL/'MANUSCRIPT_SNAPSHOT.json',{'repository':'https://github.com/JLU-MCNS-MEC/2027-Qiaoxu-CVPR',
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=paper,text=True).strip(),
        'sections':{f'sec/{name}':sha(paper/'sec'/name) for name in ['3_problem.tex','4_method.tex']},
        'working_tree_modified_by_this_task':False})
    upstream=[]
    for repo in ['crotonyl/HETT','tyrofanfan/HTNav']:
        content=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/contents/README.md'],text=True))
        text=base64.b64decode(content['content']).decode()
        releases=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/releases'],text=True))
        upstream.append({'repo':repo,'readme_blob_sha':content['sha'],
            'readme_content_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'upload_in_progress_notice':bool(re.search(r'upload.*in progress',text,re.I)),
            'releases':[{'tag':r['tag_name'],'assets':[{'name':a['name'],'bytes':a['size'],'url':a['browser_download_url']} for a in r['assets']]} for r in releases],
            'weight_readiness_inference':'Readme/release metadata only; no native inference or weight compatibility assertion'})
    base=read(ASSETS/'baseline_preflight.json')
    local=Path('/home/ubuntu5/Workspace/HETT/weights/best.pt')
    candidates=[]
    if local.exists():candidates.append({'path':str(local),'bytes':local.stat().st_size,'sha256':sha(local),
        'equals_archived_official_HETT_sha':sha(local)==base['HETT']['sha256'],
        'note':'Filename alone is not evidence of a HETT navigation checkpoint; this weights directory also contains yolo_v3.cfg.'})
    dump(FINAL/'BASELINE_READINESS.json',{'upstream_metadata':upstream,
        'archived_HETT_asset':base['HETT'],'archived_HTNav_status':base['HTNav'],
        'archived_HETT_path_exists_on_this_host':Path(base['HETT']['checkpoint']).exists(),
        'local_candidate_files':candidates,
        'matched_official_HETT_runtime_and_evaluator_verified':False,'verified_official_HTNav_checkpoint_found':False,
        'no_forced_reproduction':True,'new_training':False})


if __name__=='__main__':main()
