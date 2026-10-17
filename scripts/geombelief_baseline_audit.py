"""Read-only baseline protocol refresh, pinned official sources and local evidence."""
import json,urllib.request,hashlib,subprocess,csv
from pathlib import Path
from scripts.geombelief_map_experiments import OUT,HERE
from graphnav.checkpointing import file_sha256,atomic_json

def main():
    dest=OUT/'baseline_sources';dest.mkdir(exist_ok=True)
    repos={'MGP':'water-cookie/citynav','HETT':'crotonyl/HETT','HTNav':'tyrofanfan/HTNav','GeoNav':'Xhtshr/geonav-official','FlightGPT':'Uavln/FlightGPT'}
    rows=[]
    for method,repo in repos.items():
        try:
            info=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/commits/HEAD'],text=True))
            commit=info['sha']
            for path in ['README.md','gsamllavanav/evaluate.py','htnav/evaluate.py','vlnce/evaluate.py','eval.py']:
                url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
                try:
                    content=urllib.request.urlopen(url,timeout=30).read()
                    target=dest/(method+'__'+path.replace('/','__'));target.write_bytes(content)
                    rows.append(dict(method=method,repo=repo,commit=commit,url=url,path=str(target),sha256=file_sha256(target),status='Retrieved'))
                except Exception as e:rows.append(dict(method=method,repo=repo,commit=commit,url=url,status='Unavailable',reason=str(e)))
        except Exception as e:rows.append(dict(method=method,repo=repo,status='Blocked',reason=str(e)))
    atomic_json(rows,OUT/'baseline_source_manifest.json')
    parent=HERE/'reports/paper_experiments'
    for name in ['table1_paper_reported.csv','table1_ours_unified.csv','table1_local_other_protocols.csv']:
        source=list(csv.DictReader((parent/name).open()))
        selected=[r for r in source if r.get('method') in repos]
        if selected:
            with (OUT/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    (OUT/'BASELINE_PROTOCOL_UPDATE.md').write_text('''# GeomBelief baseline protocol update

Official source snapshots, commit SHAs and fetch failures are in baseline_source_manifest.json. Original results and baselines were not retrained. Existing source evidence remains in ../baseline_protocol_audit.md and ../dataset_version_audit.json.

|Method|Evidence category|Remaining comparability issue|
|---|---|---|
|MGP|Official code and recovered weight; local variants separately indexed|Goal/progress postprocessing and cache protocols differ; no relabeling as strict reproduction|
|HETT|Official repository and released checkpoint evidence|Local research checkout contains modifications; best_val_unseen weight-selection provenance remains explicit|
|HTNav|Official source; local independent training results|Official README still says code upload in progress; no verified official trained weight was found in that README. Local model is not an official-weight reproduction|
|GeoNav (DeepSeek Flash)|Model-substituted API experiment only|Not GPT-4o, single local run versus paper multiple runs; original Test 5311 must not be merged with revised Test5281|
|FlightGPT|Official code and public model reference|Full local official-weight evaluation not complete; no number invented|

Three copied CSVs keep paper-reported, local common-evaluator and other/API protocols separate. They are evidence views, not newly reproduced results.

All local GeomBelief metrics use XY final distance ≤20m; OSR uses any actual visited XY; SPL = success × straight-line start-to-goal distance / max(actual XY path length, straight-line start-to-goal distance). GT enters scoring only. Stop probability, predicted-goal arrival gate and evaluator success are different concepts.

HTNav local htnav/evaluate.py uses XY distances but its SPL optimal length is the generated shortest teacher trajectory, not necessarily Euclidean start-to-goal distance. GeoNav vlnce/evaluate.py uses a 3D final-distance branch and teacher path length, while scenegraphnav/evaluate.py uses XY; the module actually used must be recorded rather than assigning a repository-wide metric. Native '<20' versus common '<=20' is also retained in audit. FlightGPT includes adapted evaluator modules; do not infer formula from metric names.

Current revised episode counts are 2470/2697/5281, versus original2498/2826/5311. Prior file-level audit found311 total removed episodes; this is not interchangeable with every paper's stated revision count. No official dataset-byte identity is asserted without matching source SHA. Inputs differ across methods: language/name+map geometry, observed RGB-D/history, global map images and API model prior knowledge are reported separately.

Official links: [MGP/CityNav](https://github.com/water-cookie/citynav), [HETT](https://github.com/crotonyl/HETT), [HTNav](https://github.com/tyrofanfan/HTNav), [GeoNav](https://github.com/Xhtshr/geonav-official), [FlightGPT](https://github.com/Uavln/FlightGPT). No excluded historical research method was added to the comparison.
''')
if __name__=='__main__':main()
