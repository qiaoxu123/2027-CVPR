"""Finalize reproducibility checks/artifact hashes; optionally link the pushed commit."""
from common import *
import argparse,re,ast,subprocess,shutil

def paper_notes(commit):
    url=f'https://github.com/qiaoxu123/2027-CVPR/tree/{commit}/reports/paper_assets_20261020'
    build=read(OUT/'latex_build.json');pages=build['pages'];reference_page=build['references_start_page']
    text=f'''# Real CityNav visual evidence manifest — 2026-10-20

Experiment evidence commit: `{commit}` on `research/visual-residual-20261013`.

[Complete source bundle]({url}); [asset inventory]({url}/ASSET_MANIFEST.md); [report and limitations]({url}/REPORT.md); [per-figure provenance]({url}/figures).

The formal manuscript identity GeomBelief is frozen GraphNav-V1 C epoch1, SHA `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`. Figure 2 uses independent fixed-seed matched geometry heads; Figure 3 uses the independently calibrated residual. These are separate configurations, never a combined new main result.

|Figure|Real evidence|Files|
|---|---|---|
|1 Overview|Archived formal episode `birmingham_block_5:8:4`; actual language, named map, B0/Bt, residual, footprints and path/Stop|figures/citynav/figure1_overview.*|
|2 Geometry|Query `birmingham_block_5:8:5:0`; Name/Type, Shuffled, Correct; errors 358.29/22.29/12.02m; three separately trained fixed heads|figures/citynav/figure2_geometry.*|
|3 Visual|Corrected `8:5`, reinforced `8:4`, degraded `33:0` on birmingham_block_5; actual fields with matched-pose Real/Zero/Shuffle/No-residual controls|figures/citynav/figure3_visual_updates.*|
|4 Navigation|Two successes, two wrong-goal false Stops and two visited-but-failed cases, formal checkpoint|figures/citynav/figure4_navigation.*|

Nine episode--model cases cover eight distinct validation episodes, 80 observed-frame instances and 77 distinct map/pose views. Dense arrays and traces in `figures/citynav/source/` reproduce all public plots using `python figures/citynav/reproduce.py` without a checkpoint or raw dataset. All four PNG previews match independent bundled-source redraws byte for byte. Every figure has exact source/weight/data SHA, and the full bundle adds 310 L1/KL/Top-1-change rows, 10,448 offline-rescored evaluation episodes and split/subgroup statistics.

The uploaded drawings contain real measured map/prediction/trajectory/observation-footprint data, no synthetic city scenes or RGB pixels. Actual local RGB-rich PDFs/PNGs exist at `/home/rental/20260922_1/Workspace/2027-CVPR/GeomBelief-assets/reports/paper_assets_20261020/local_only/figures/`. Crop indices, original rasters' SHA/affine and renderer commands are in the bundled observation_index.json. CityNav declares CC BY 4.0, but the underlying SensatUrban raster redistribution terms have not been established merely from the code MIT license; RGB remains local pending that clarification. Map attribution: CityNav / OpenStreetMap contributors, https://www.openstreetmap.org/copyright.

Original conceptual artwork is retained as `figures/overview_conceptual.tex` and `figures/visual_evidence_conceptual.tex`, outside the live manuscript inputs. Fixed source-order, outcome-selected examples illustrate behavior; they are not unbiased frequency/causal estimates and were not used to choose weights. GT enters scoring/plotting only.

Validation and local LaTeX build: `notes/latex_build.json`. The working draft compiles to {pages} pages (references from page {reference_page}); page-budget editing remains outside this asset audit. No undefined references/citations or overfull boxes remain. Existing package/font warnings are recorded. The three official full-split results, weights and original logs are unchanged. Concurrent remote layout work through c8bdaba was integrated without force-pushing it away; its architecture and measured profiling graphics remain live and all illustrative source files are preserved. Standalone figure filenames identify the four real-data deliverables; extra architecture/profiling figures have their own manuscript counters.
'''
    (PAPER/'notes/visual_evidence_manifest.md').write_text(text)
    for name in ('baseline_verification.md','bibliography_verification.md'):
        p=PAPER/'notes'/name;s=p.read_text()
        s=s.split('## Immutable evidence commit')[0].rstrip()+f'\n\n## Immutable evidence commit\n\n[{commit}]({url}), including source tables, hashes, CSVs, exact protocol restrictions and reproduction scripts.\n'
        p.write_text(s)
    dump(PAPER/'notes/experiment_asset_commit.json',dict(repo='qiaoxu123/2027-CVPR',branch='research/visual-residual-20261013',commit=commit,bundle='reports/paper_assets_20261020',url=url))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paper-notes-commit');args=parser.parse_args()
    if args.paper_notes_commit:
        assert re.fullmatch('[0-9a-f]{40}',args.paper_notes_commit)
        paper_notes(args.paper_notes_commit);return
    proc=subprocess.run([__import__('sys').executable,str(OUT/'scripts/validate.py')],capture_output=True,text=True)
    (OUT/'verification.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    import pymupdf
    pdf=PAPER/'build/main.pdf';doc=pymupdf.open(pdf);log=(PAPER/'build/main.log').read_text();old=read(OUT/'latex_build.json')
    old.update(pdf_sha256=sha(pdf),pages=len(doc),undefined_citations=bool(re.search(r'Citation .* undefined',log)),undefined_references=bool(re.search(r'Reference .* undefined',log)),overfull_hbox='Overfull \\hbox' in log)
    old['references_start_page']=next(i+1 for i,page in enumerate(doc) if '\nReferences\n' in page.get_text())
    old['notes']=re.sub(r'Current \d+-page',f'Current {len(doc)}-page',old['notes'])
    old['notes']=re.sub(r'references from page \d+',f'references from page {old["references_start_page"]}',old['notes'])
    assert not any(old[k] for k in ('undefined_citations','undefined_references','overfull_hbox'))
    for dest in (OUT/'latex_build.json',PAPER/'notes/latex_build.json'):dump(dest,old)
    shutil.copyfile(PAPER/'build/compile_stdout.log',OUT/'latex_compile.log')
    report=OUT/'REPORT.md';s=report.read_text();s=re.sub(r'compiles to \d+ pages, references beginning on page \d+',f'compiles to {len(doc)} pages, references beginning on page {old["references_start_page"]}',s);report.write_text(s)
    checks=read(OUT/'verification.json');checks['latex_build']=old;checks['profiling_common_ids']=read(OUT/'profiling_scope.json')['common_episode_ids'];dump(OUT/'verification.json',checks)
    secret=re.compile(rb'\b(?:github_pat_|ghp_|hf_|sk-)[A-Za-z0-9_\-]{18,}');rows=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or any(k in p.relative_to(OUT).parts for k in ('local_only','__pycache__')) or p.name=='PRODUCT_MANIFEST.csv':continue
        assert not secret.search(p.read_bytes()),'Potential credential: '+str(p.relative_to(OUT))
        assert p.stat().st_size<20*1024*1024,'Unexpected large public artifact: '+str(p)
        if p.suffix=='.py':ast.parse(p.read_text())
        rows.append(dict(path=str(p.relative_to(OUT)),status='Existing',bytes=p.stat().st_size,sha256=sha(p)))
    csvwrite(OUT/'PRODUCT_MANIFEST.csv',rows)
    print('Final public manifest:',len(rows),'files;',sum(r['bytes'] for r in rows),'bytes')

if __name__=='__main__':main()
