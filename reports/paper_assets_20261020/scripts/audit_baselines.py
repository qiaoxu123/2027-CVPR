"""Keep published, official-weight, independent and substituted results separate."""
from common import *
import re,requests,subprocess
from bs4 import BeautifulSoup

SPLITS=('val_seen','val_unseen','test_unseen')
POLICIES={
 'AerialVLN+GSM':dict(inputs='language + RGB/depth + GSM; CityNav adaptation',action_space='native discrete actions; local dual-GRU adaptation separate',max_steps='native/local protocol recorded per result; not assumed equal to 20 bounded waypoints',stop='native/local progress/action logic',limitation='CityNav published adaptation and independent local reimplementation are separate; no verified author CityNav-adaptation checkpoint is substituted. Local training assumptions/controls remain explicit.'),
 'MGP':dict(inputs='language + GSM/map; goal and progress controller',action_space='native macro + repeated primitive moves; historical variants retained',max_steps='20 macro; move_iteration=5 in archived faithful wrapper',stop='predicted progress (historical threshold .75)',limitation='Official checkpoint recovered; historical result checkpoint path is absent. Native cache/goal-readout variants and OSR differ. No new full evaluation claimed.'),
 'FlightGPT':dict(inputs='language + current RGB + global annotated orthophoto/map',action_space='VLM action sequence',max_steps='2 VLM rounds x up to 75 primitive actions in released eval.py',stop='VLM-generated actions/native environment',limitation='Public 16.60GB GRPO shards not locally complete; vLLM environment and fixed-ID evaluator not validated. No local navigation score.'),
 'HETT':dict(inputs='language + map + RGB/depth + history',action_space='coarse goal and fine actions',max_steps='native budget must be verified in checkpoint-compatible wrapper',stop='native learned progress/action logic; not GraphNav .3',limitation='Official best_val_unseen checkpoint present. Research checkout cannot stand in for upstream model; no verified complete native inference/export environment. Existing documentation-only scores are historical evidence.'),
 'HTNav':dict(inputs='language + RGB/depth + map; hybrid IL/RL',action_space='tiered macro goal + micro actions',max_steps='native hierarchical budget differs from 20 bounded waypoints',stop='native value/progress and tiered control',limitation='No verified released trained checkpoint. Local independently trained repaired implementation is a distinct variant. Native SPL teacher-path denominator differs.'),
 'GeoNav':dict(inputs='GPT-4o + multiscale geospatial/context maps + current observations',action_space='high-level API plan and native movement',max_steps='native API pipeline; not assumed equal to 20',stop='phase/target localization API logic',limitation='Local original Test5311 used DeepSeek Flash, not official GPT-4o. No revised three-split original-model run; preserve API/input/3D-versus-XY scoring differences.'),
 'GeomBelief':dict(inputs='language + named public map polygons + causal rendered RGB/depth support/history',action_space='bounded waypoint: horizontal <=50m, vertical <=10m',max_steps='20',stop='probability .3 with altitude/arrival guard',limitation='Frozen C epoch1, Seen128 SR/SPL then Train Dev selection. Other ablation weights and 384 grid never substituted into main row.'),
 'Seq2Seq+GSM':dict(inputs='language + RGB/depth + GSM',action_space='native discrete actions',max_steps='native wrapper; see source registry',stop='native progress/action',limitation='Local independent training; missing historical checkpoint links, GSM cache/gdino variants differ.'),
 'CMA+GSM':dict(inputs='language + RGB/depth + GSM',action_space='native discrete actions',max_steps='native wrapper; see source registry',stop='native progress/action',limitation='Local independent training; missing historical checkpoint links, GSM cache/gdino variants differ.'),
}

def main():
    primary=read(OUT/'sources/primary_source_manifest.json');pm={r['key']:r for r in primary}
    htnav=(OUT/'local_only/source_downloads/htnav_paper.txt').read_text()
    table=htnav.split('Table 1. Performance comparison',1)[1].split('Furthermore,',1)[0]
    rows=[]
    revised={
      'MGP':[54.1,10.96,40.82,10.42,65.6,8.33,26.75,7.64,82.6,9.70,31.46,8.22],
      'FlightGPT':[62.7,19.95,33.33,18.29,62.4,16.25,33.87,14.33,61.4,24.47,42.17,21.29],
      'HTNav':[37.5,31.05,51.82,27.48,49.8,17.69,37.82,15.44,40.3,25.49,48.82,21.56]}
    for method,values in revised.items():
        start=table.index(method+'*\n')+len(method)+2
        found=[float(s) for s in table[start:].splitlines()[:12]]
        assert found==values,(method,found)
        for i,split in enumerate(SPLITS):
            rows.append(dict(method=method,variant='author-reported revised row',evidence_layer='published_table',dataset='revised_paper_no_file_hash',split=split,
                **dict(zip(('NE','SR','OSR','SPL'),values[4*i:4*i+4])),source_url=pm['htnav_paper']['url'],source_sha256=pm['htnav_paper']['sha256'],source_locator='HTNav CVPR2026 Table 1, PDF p6 (printed 10981)',status='Verified publisher table',strictly_comparable=False))
    # The publisher endpoint is blocked; use the same authors' archived arXiv table.
    hp=OLD/'sources/hett.html';soup=BeautifulSoup(hp.read_text(),'html.parser')
    ht=next(t for t in soup.find_all('table') if '31.09' in t.get_text())
    tr=next(t for t in ht.find_all('tr') if 'HETT(Ours)*' in t.get_text(' ',strip=True).replace(' ',''))
    cells=tr.find_all(['td','th']);values=[float(c.get_text(strip=True)) for c in cells[1:]]
    assert values==[37.2,31.09,51.86,25.76,51.3,19.10,34.78,16.70,40.4,28.90,49.56,23.79]
    for i,split in enumerate(SPLITS):
        rows.append(dict(method='HETT',variant='author-reported refined row',evidence_layer='published_table',dataset='revised_paper_no_file_hash',split=split,
            **dict(zip(('NE','SR','OSR','SPL'),values[4*i:4*i+4])),source_url='https://arxiv.org/html/2512.14222',source_sha256=sha(hp),source_locator='HETT authors Table 2; AAAI PDF retrieval blocked; archived author HTML verified',status='Verified author table; publisher endpoint blocked',strictly_comparable=False))
    # Original benchmark aggregates remain reference rows; source retained per row.
    for r in csv.DictReader((OLD/'table1_paper_reported.csv').open()):
        if r['method'] not in POLICIES or r['dataset']!='original_CityNav':continue
        entry=dict(method=r['method'],variant='published original benchmark reference',evidence_layer='published_table',dataset='original_paper_no_file_hash',split=r['split'],
              **{k:r[k] for k in ('NE','SR','OSR','SPL')},source_url=r['source_url'],source_sha256=sha(OLD/'table1_paper_reported.csv'),source_locator='Archived table1_paper_reported.csv; secondary transcription retained',status='Archived published reference; secondary provenance',strictly_comparable=False)
        if r['method'] in ('MGP','HTNav','FlightGPT','Seq2Seq+GSM','CMA+GSM','AerialVLN+GSM'):
            start=table.index(r['method']+'\n')+len(r['method'])+1
            found=[float(x) for x in table[start:].splitlines()[:12]];i=SPLITS.index(r['split'])
            assert found[4*i:4*i+4]==[float(r[k]) for k in ('NE','SR','OSR','SPL')]
            entry.update(source_url=pm['htnav_paper']['url'],source_sha256=pm['htnav_paper']['sha256'],source_locator='HTNav Table 1 original row, PDF p6',status='Verified publisher comparison table')
        if r['method']=='HETT':
            entry.update(source_url='https://arxiv.org/html/2512.14222',source_sha256=sha(hp),source_locator='HETT author Table 2 original row',status='Verified archived author table')
        rows.append(entry)
    # Existing local records are rehashed, never relabeled as paper/official outputs.
    for name in ('table1_ours_unified.csv','table1_local_other_protocols.csv'):
        for r in csv.DictReader((OLD/name).open()):
            method='GeomBelief' if r['method']=='PolyBelief' else r['method']
            if method not in POLICIES:continue
            if method=='GeomBelief' and r['checkpoint_sha256']!='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a':continue
            p=current(r['result_path']);cp=current(r['checkpoint']) if r['checkpoint'] else None
            pred=current(r['predictions']) if r.get('predictions') else None
            layer='local_frozen_main' if method=='GeomBelief' else 'local_independent_reproduction'
            if method=='MGP':layer='local_official_weight_claim_unverified_historical_hash'
            if method=='GeoNav':layer='local_substituted_model'
            if method=='HETT':layer='historical_document_only'
            actual_hash=sha(p) if p.is_file() else ''
            asserted=r.get('result_sha256','')
            assert not asserted or not actual_hash or asserted==actual_hash,(str(p),asserted,actual_hash)
            entry={k:r[k] for k in ('variant','dataset','split','episodes','NE','SR','OSR','SPL','checkpoint_sha256','source_commit','protocol','data_sha256','command') if k in r}
            entry.update(method=method,evidence_layer=layer,source_file=str(p),source_sha256=actual_hash,
                checkpoint=str(cp) if cp else '',checkpoint_present=bool(cp and cp.is_file()),checkpoint_actual_sha256=sha(cp) if cp and cp.is_file() else '',
                predictions=str(pred) if pred else '',predictions_present=bool(pred and pred.is_file()),
                source_locator=f'{name}; '+r.get('experiment_id',''),status='Existing verified file' if p.is_file() else 'Missing source file',
                metric_scope='common XY recomputation' if name=='table1_ours_unified.csv' else 'historical/native protocol',strictly_comparable=False)
            rows.append(entry)
    recovered=read(OLD/'mgp_recovered_asset.json');hett=read(OLD/'hett_asset_preflight.json')
    availability={
      'MGP':(Path(recovered['asset']),recovered['sha256'],'Recovered official checkpoint; not proven identical to absent historical checkpoint'),
      'HETT':(Path(hett['checkpoint']),hett['sha256'],'Official best_val_unseen; clean native complete inference/export not validated'),
      'FlightGPT':(BASELINES/'FlightGPT/model_weight/Flightgpt/model-00001-of-00004.safetensors','','Locked revision 9eb62538741af0e4239cb7394cb9cf9b6dc6c703; incomplete local shards'),
      'HTNav':(BASELINES/'HTNav/official_trained_checkpoint.pt','','No verified official trained checkpoint'),
      'GeoNav':(BASELINES/'GeoNav/official_gpt4o_checkpoint','','Original uses API GPT-4o, not a downloadable local checkpoint; local substituted API run separate')}
    for method,(p,expected,note) in availability.items():
        actual=sha(p) if p.is_file() else '';assert not expected or actual==expected,(method,actual)
        for split in SPLITS:
            rows.append(dict(method=method,variant='official asset readiness, not an evaluation',evidence_layer='official_released_asset',dataset='revised_target_protocol',split=split,checkpoint=str(p),checkpoint_present=p.is_file(),checkpoint_sha256=actual,
                             status='Blocked: no validated complete inference protocol/environment',limitation=note,strictly_comparable=False))
    for r in rows:
        meta=POLICIES[r['method']]
        r.update({k:v for k,v in meta.items() if k!='limitation'})
        r['limitation']=r.get('limitation','')+('; ' if r.get('limitation') else '')+meta['limitation']
        r['selection_policy']='Seen128 SR/SPL then Train Dev; never Unseen/Test' if r['method']=='GeomBelief' else ('best_val_unseen filename on official release; not selected anew in this audit' if r['method']=='HETT' else 'Historical/source selection retained; no new Unseen/Test selection')
        if r['method']=='MGP' and (OUT/'baseline_preflight.json').exists():r['native_preflight']=read(OUT/'baseline_preflight.json')['MGP']['result']
    csvwrite(OUT/'baseline_results.csv',rows)
    # Compare exact current manuscript revised rows with the independently parsed sources.
    tex=(PAPER/'sec/5_experiments.tex').read_text();checks=[]
    for method in revised.keys()|{'HETT','GeomBelief'}:
        line=next(l for l in tex.splitlines() if l.startswith(method+('$' if method!='GeomBelief' else ' (local)')))
        numbers=[float(v) for v in re.findall(r'(?<![A-Za-z])\d+\.\d+',line)]
        refs=[]
        for split in SPLITS:
            ref=next(r for r in rows if r['method']==method and r['split']==split and (r['evidence_layer']=='local_frozen_main' if method=='GeomBelief' else r['evidence_layer']=='published_table' and 'revised' in r['dataset']))
            refs.extend(float(ref[k]) for k in ('NE','SR','OSR','SPL'))
        assert len(numbers)==12 and all(abs(a-b)<=.0051 for a,b in zip(numbers,refs)),(method,numbers,refs)
        checks.append(dict(method=method,checked_metrics=12,all_match_at_paper_precision=True))
    dump(OUT/'baseline_table_checks.json',checks)
    original_checks=[]
    original_tex=tex.split('\\label{tab:baselines_original}',1)[1].split('\\end{table}',1)[0]
    for method in ('Seq2Seq+GSM','CMA+GSM','AerialVLN+GSM','MGP','FlightGPT','HETT','HTNav','GeoNav'):
        line=next(l for l in original_tex.splitlines() if l.startswith(method+'~'))
        numbers=[float(v) for v in re.findall(r'(?<![A-Za-z])\d+\.\d+',line)];refs=[]
        for split in SPLITS:
            ref=next(r for r in rows if r['method']==method and r['split']==split and r['evidence_layer']=='published_table' and r['dataset']=='original_paper_no_file_hash')
            refs.extend(float(ref[k]) for k in ('SR','SPL'))
        assert len(numbers)==6 and all(abs(a-b)<.0051 for a,b in zip(numbers,refs)),(method,numbers,refs)
        original_checks.append(dict(method=method,checked_metrics=6,all_match_at_paper_precision=True))
    dump(OUT/'baseline_original_table_checks.json',original_checks)
    # Current upstream README/commit probes; raw third-party text stays local.
    probes=[]
    for method,repo in [('MGP','water-cookie/citynav'),('HETT','crotonyl/HETT'),('HTNav','tyrofanfan/HTNav'),('FlightGPT','Uavln/FlightGPT'),('GeoNav','Xhtshr/geonav-official')]:
        try:
            meta=requests.get('https://api.github.com/repos/'+repo,timeout=20);meta.raise_for_status();branch=meta.json()['default_branch']
            c=requests.get(f'https://api.github.com/repos/{repo}/commits/{branch}',timeout=20);c.raise_for_status();commit=c.json()['sha']
            rr=requests.get(f'https://raw.githubusercontent.com/{repo}/{commit}/README.md',timeout=20);rr.raise_for_status()
            target=OUT/'local_only/source_downloads'/f'{method}_README.md';target.write_bytes(rr.content)
            probes.append(dict(method=method,repo=repo,commit=commit,readme_sha256=sha(target),url=rr.url,status='Retrieved',mentions_upload_in_progress=bool(re.search('upload.*progress|uploading',rr.text,re.I))))
        except Exception as e:probes.append(dict(method=method,repo=repo,status='Blocked',reason=str(e)))
    dump(OUT/'sources/upstream_probe.json',probes)
    lines=['# Baseline verification and protocol comparability','',
      'Audit date: 2026-10-20. NE is in meters; SR/OSR/SPL are percentages. Empty cells mean not measured/verified; they never receive copied paper values.',
      '', '## Evidence layers','',
      '`published_table`: author/publication aggregates; `official_released_asset`: checkpoint readiness with no score; `local_independent_reproduction`: independent/repaired implementations; `local_substituted_model`: changed API model; `historical_document_only`: no verified episode-level source. MGP historical official-weight claims retain unverified checkpoint linkage. GeomBelief is frozen GraphNav-V1 C epoch1 under the manuscript name.',
      '', '## Current manuscript table','',
      'All 60 revised-table numeric cells match the independently parsed source/archived formal values at manuscript precision. MGP*, FlightGPT*, HTNav* come from [HTNav Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Fan_HTNav_A_Hybrid_Navigation_Framework_with_Tiered_Structure_for_Urban_CVPR_2026_paper.pdf). HETT* matches the [authors Table 2](https://arxiv.org/html/2512.14222); AAAI PDF access failed through the current proxy, so publisher-byte validation is blocked. Original CityNav rows are separately audited; GeoNav original numbers retain the existing secondary-table provenance rather than claiming its paywalled original table was fetched.',
      '', '## Method readiness and restrictions','', '|Method|Existing asset / remaining restriction|','|---|---|']
    lines += [f'|{m}|{v[2]}|' for m,v in availability.items()]
    lines += ['', '## Exact data and evaluator scope','',
      'Local Original counts are 2498/2826/5311; Revised counts 2470/2697/5281. Full file SHA and removed IDs are indexed in the preserved dataset_version_audit.json. Counts alone do not establish author dataset byte identity. Retained episode language/trajectory/target are unchanged in that audit; revised reference annotations differ separately.',
      '', 'Common offline metrics: NE = final XY Euclidean distance; SR = NE <=20m; OSR = min over actually visited XY <=20m; SPL = SR × straight-line start-to-GT distance / max(actual XY path length, straight-line distance). This harmonizes metric definitions only. Input privileges, geometry, controller, step/action budget, stopping and checkpoint-selection protocols remain different. HTNav native teacher-path SPL and GeoNav native 3D evaluator branches must not be conflated with common XY metrics.',
      '', 'There is no newly completed strict official-weight matched evaluation in this bundle. MGP historical refined three-split outputs already exist and are retained; recovered weight identity cannot be retroactively asserted. HETT requires a clean checkpoint-compatible runtime and isolated ID-preserving exporter; FlightGPT lacks complete shards/runtime; HTNav lacks official trained weights; original GeoNav requires the original API protocol. No training is necessary for asset organization and none was started.',
      '', 'No Val-Unseen/Test-Unseen checkpoint selection occurred. Existing release and historical selection provenance is retained, including HETT best_val_unseen. No cross-protocol ranking, significance or speedup claim is made.',
      '', 'Machine-readable evidence: baseline_results.csv, baseline_table_checks.json, sources/upstream_probe.json. Every local source is rehashed; missing weights and predictions remain explicit.']
    lines += ['', '## Original-table coverage', '', 'All 48 SR/SPL cells in the current original-data manuscript table also match the indexed published references, including AerialVLN+GSM. Together with 60 revised-table cells, this checks 108 manuscript numeric cells. Full NE/SR/OSR/SPL are retained in the CSV wherever available; unavailable original-source details remain explicit.']
    if (OUT/'baseline_preflight.json').exists():
        lines += ['', '## Fresh native readiness probe', '', 'MGP native `main_goal_predictor.py --help` fails in the declared reproduction environment with `ModuleNotFoundError: No module named groundingdino`; see baseline_preflight.json and sources/mgp_native_preflight.log for command, exit status and SHA. The native entrypoint unconditionally loops all splits and applies terminal goal selection, so it is not an already-validated isolated fixed-ID common-protocol evaluator. The actual FlightGPT snapshot contains metadata/tokenizer files and zero of four complete weight shards. No native training or Test inference was launched.']
    (OUT/'BASELINE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print('baseline rows',len(rows),'manuscript revised cells checked',sum(r['checked_metrics'] for r in checks))

if __name__=='__main__':main()
