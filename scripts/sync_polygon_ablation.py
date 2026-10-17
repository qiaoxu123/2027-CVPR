"""Back up completed Polygon research outputs, preserving prior remote assets."""
import json
from pathlib import Path
import subprocess
import tempfile
import traceback
import time

from scripts.polygon_ablation_plan import HERE,OUT,WORKSPACE
from graphnav.checkpointing import file_sha256,atomic_json

REPO='qiaoxu123/2027-CVPR'
GIT=['git','-c','url.https://github.com/qiaoxu123/2027-CVPR.git.insteadOf=https://github.com/qiaoxu123/2027-CVPR-Dev.git','-c','credential.helper=!gh auth git-credential']


def sync_completed_experiment():
    if not (OUT/'workflow_complete.json').exists():return
    if (OUT/'remote_sync.json').exists() and json.loads((OUT/'remote_sync.json').read_text()).get('verified'):
        return
    try:
        _sync()
    except Exception:
        atomic_json(dict(status='Blocked',error=traceback.format_exc(),time=time.time()),OUT/'remote_sync_failure.json')


def _sync():
    allowed=['.gitignore','graphnav/polygon_ablation.py',
             *[str(p.relative_to(HERE)) for p in (HERE/'scripts').glob('*polygon*.py')],
             'tests/test_polygon_ablation.py','tests/test_polygon_statistics.py','docs/PAPER_EXPERIMENT_TODO.md',
             'reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md','reports/paper_experiments/experiment_registry.csv',
             'reports/paper_experiments/todo_status.csv','reports/paper_experiments/polygon_ablation']
    existing=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=HERE,text=True).splitlines()
    if any(not any(p==a or p.startswith(a+'/') for a in allowed) for p in existing):
        raise ValueError('Unrelated staged files exist; refuse to include them in automatic experiment commit')
    subprocess.run(['git','add',*allowed],cwd=HERE,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=HERE).returncode:
        subprocess.run(['git','commit','-m','Complete paired Polygon representation experiments and map-level analysis'],cwd=HERE,check=True)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=HERE,text=True).strip()
    subprocess.run([*GIT,'push','origin',branch],cwd=HERE,check=True)
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    plan_sha=file_sha256(OUT/'frozen_plan.json')
    tag='polybelief-polygon-ablation-20261017-'+plan_sha[:8]
    scratch=Path(tempfile.mkdtemp(prefix='polybelief-polygon-assets-'))
    # Cache tensors are reproducible and excluded; all actual trained weights,
    # raw episode/query predictions and stable logs are retained.
    files=[p for p in OUT.rglob('*') if p.is_file() and '.cache' not in p.parts and
           'public_inputs' not in p.parts and p.name not in ('run.lock','remote_sync.json','remote_sync_failure.json')]
    manifest=dict(source_commit=source_commit,restore_root='Workspace',excluded='Shared DATA and reproducible feature caches',
                  files=[dict(path=str(p.relative_to(WORKSPACE)),bytes=p.stat().st_size,sha256=file_sha256(p)) for p in sorted(files)])
    mp=scratch/'POLYGON_ASSET_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    listing=scratch/'files.txt';listing.write_text(''.join(str(p.relative_to(WORKSPACE))+'\n' for p in sorted(files)))
    archive=scratch/'polybelief-polygon-ablation.tar.zst'
    subprocess.run(['tar','--zstd','-cf',str(archive),'-C',str(WORKSPACE),'-T',str(listing)],check=True)
    archives=[archive]
    # GitHub release assets have a per-file size limit. Chunk the byte stream
    # without changing or omitting any experiment file when needed.
    if archive.stat().st_size>1500*1024**2:
        archives=[]
        with archive.open('rb') as source:
            part=0
            while source.tell()<archive.stat().st_size:
                target=scratch/f'{archive.name}.part{part:03d}'
                with target.open('wb') as dest:
                    remaining=1500*1024**2
                    while remaining:
                        block=source.read(min(8*1024**2,remaining))
                        if not block:break
                        dest.write(block);remaining-=len(block)
                archives.append(target);part+=1
    assets=[*archives,mp]
    sums=scratch/'POLYGON_SHA256SUMS';sums.write_text(''.join(f'{file_sha256(p)}  {p.name}\n' for p in assets));assets.append(sums)
    probe=subprocess.run(['gh','release','view',tag,'--repo',REPO],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if probe.returncode==0:
        raise ValueError('Release tag already exists but no verified local receipt; preserve remote asset, investigate before overwrite')
    notes=scratch/'notes.md';notes.write_text('Completed pre-registered Polygon geometry contribution experiments. Frozen interventions and independent three-seed retraining, Descriptor Only and End-to-End, are reported separately. Positive and negative results preserved. See the committed POLYGON_ABLATION_REPORT.md. No original model or prior result overwritten. Restore from Workspace after checking SHA256SUMS and per-file manifest.\n')
    notes.write_text(notes.read_text()+'If the archive is chunked, concatenate .part000, .part001, ... in lexical order to restore polybelief-polygon-ablation.tar.zst before extraction.\n')
    subprocess.run(['gh','release','create',tag,*map(str,assets),'--repo',REPO,'--target',source_commit,'--title','PolyBelief Polygon contribution experiment assets','--notes-file',str(notes),'--prerelease'],check=True)
    remote=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{tag}'],text=True))
    for f in assets:
        a=next(a for a in remote['assets'] if a['name']==f.name)
        if a['digest']!='sha256:'+file_sha256(f) or a['size']!=f.stat().st_size:
            raise ValueError('Remote digest/size differs: '+f.name)
    atomic_json(dict(status='Done',verified=True,source_commit=source_commit,branch=branch,release_url=remote['html_url'],
                     archive_bytes=archive.stat().st_size,archive_sha256=file_sha256(archive),files=len(files),
                     assets=[dict(name=p.name,bytes=p.stat().st_size,sha256=file_sha256(p)) for p in assets],
                     remote_server_digest_matches=True,time=time.time()),OUT/'remote_sync.json')
    (OUT/'POLYGON_ASSET_MANIFEST.json').write_bytes(mp.read_bytes())
    (OUT/'POLYGON_SHA256SUMS').write_bytes(sums.read_bytes())
    indexes=[HERE/'docs/PAPER_EXPERIMENT_TODO.md',HERE/'reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md']
    for path in indexes:
        path.write_text(path.read_text().replace('|原始结果和权重远程备份|Running|','|原始结果和权重远程备份|Done|'))
    report=OUT/'POLYGON_ABLATION_REPORT.md'
    report.write_text(report.read_text()+f"\n远程同步已校验：[完整权重、逐样本结果和日志归档]({remote['html_url']})。服务器资产SHA及逐文件清单见remote_sync.json与POLYGON_ASSET_MANIFEST.json。\n")
    subprocess.run(['git','add',str(OUT.relative_to(HERE)),*[str(p.relative_to(HERE)) for p in indexes]],cwd=HERE,check=True)
    subprocess.run(['git','commit','-m','Record verified remote Polygon experiment asset backup'],cwd=HERE,check=True)
    subprocess.run([*GIT,'push','origin',branch],cwd=HERE,check=True)


if __name__=='__main__':sync_completed_experiment()
