"""Commit reviewed GeomBelief artifacts and verify immutable remote asset backup."""
import json,subprocess,tempfile
from pathlib import Path
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.workspace_paths import WORKSPACE
from scripts.geombelief_map_experiments import OUT,OLD,HERE
from scripts.sync_polygon_ablation import GIT,REPO


def run(args,**kw):
    with (OUT/'remote_transport.log').open('a') as f:return subprocess.run(args,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True,**kw)

def commit_push(message):
    allowed=['.gitignore','docs/PAPER_EXPERIMENT_TODO.md','reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md','reports/paper_experiments/geombelief_remaining']
    allowed += [str(p.relative_to(HERE)) for pattern in ('graphnav/geombelief*.py','scripts/geombelief*.py','tests/test_geombelief*.py') for p in HERE.glob(pattern)]
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=HERE,text=True).splitlines()
    assert all(any(x==a or x.startswith(a+'/') for a in allowed) for x in staged),'Unrelated staged work'
    run(['git','add',*allowed])
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=HERE).returncode:run(['git','commit','-m',message])
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=HERE,text=True).strip();run([*GIT,'push','origin',branch])
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),branch

def main():
    assert json.loads((OUT/'acceptance.json').read_text())['completed_core']
    receipt=OUT/'remote_sync.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('verified'):return
    commit,branch=commit_push('Complete GeomBelief map geometry and frozen controller attribution')
    references=[]
    # Reverify both old assets that hold reused correct-geometry training and
    # navigation outputs, including original C/head source weights.
    for name in ('remote_sync.json','training_static_remote_sync.json'):
        r=json.loads((OLD/name).read_text());tag=r['release_url'].rsplit('/',1)[-1]
        remote=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{tag}'],text=True))
        archive=next(a for a in remote['assets'] if a.get('digest')=='sha256:'+r['archive_sha256'])
        assert archive['size']==r['archive_bytes'];references.append(dict(release_url=r['release_url'],archive_sha256=r['archive_sha256'],bytes=r['archive_bytes'],reverified=True))
    excluded={'map_run.lock','control_run.lock','visual_run.lock','remote_sync.json','remote_transport.log','GEOMBELIEF_ASSET_MANIFEST.json','GEOMBELIEF_SHA256SUMS'}
    exclude_dirs={'.cache','public_inputs','controller_inputs','controller_observations','visual_inputs','visual_observations','rendered'}
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in excluded and not set(p.relative_to(OUT).parts)&exclude_dirs)
    manifest=dict(source_commit=commit,branch=branch,restore_root='Workspace',reused_verified_releases=references,
        excludes='Shared read-only DATA and reproducible scratch/cache only; all new training weights, logs and sample traces included',
        files=[dict(path=str(p.relative_to(WORKSPACE)),bytes=p.stat().st_size,sha256=file_sha256(p)) for p in files])
    mp=OUT/'GEOMBELIEF_ASSET_MANIFEST.json';atomic_json(manifest,mp)
    scratch=Path(tempfile.mkdtemp(prefix='geombelief-assets-'));listing=scratch/'files.txt';listing.write_text(''.join(str(p.relative_to(WORKSPACE))+'\n' for p in files))
    archive=scratch/'geombelief-core-experiments.tar.zst'
    run(['tar','--zstd','--mtime=@0','--owner=0','--group=0','--numeric-owner','-cf',str(archive),'-C',str(WORKSPACE),'-T',str(listing)])
    archives=[archive]
    if archive.stat().st_size>1500*1024**2:
        archives=[]
        with archive.open('rb') as f:
            while f.tell()<archive.stat().st_size:
                p=scratch/f'{archive.name}.part{len(archives):03d}'
                with p.open('wb') as out:
                    remaining=1500*1024**2
                    while remaining:
                        block=f.read(min(8*1024**2,remaining))
                        if not block:break
                        out.write(block);remaining-=len(block)
                archives.append(p)
    checksum=OUT/'GEOMBELIEF_SHA256SUMS';checksum.write_text(''.join(file_sha256(p)+'  '+p.name+'\n' for p in [*archives,mp]))
    tag='geombelief-core-20261017-'+file_sha256(OUT/'frozen_plan.json')[:8]
    notes=scratch/'notes.md';notes.write_text('GeomBelief controlled geometry (15 new 3-seed training jobs, 3 archived correct-geometry weights), frozen formal-model controller attribution and frozen visual-input interventions. Full raw results, all checkpoints and logs. Fair visual-input retraining remains Pending and is not included as completed. Restore old verified Polygon static/final assets first; see manifest. Source commit '+commit+'.')
    exists=subprocess.run(['gh','release','view',tag,'--repo',REPO],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    if not exists:run(['gh','release','create',tag,'--repo',REPO,'--target',commit,'--title','GeomBelief core contribution experiments','--notes-file',str(notes)])
    for p in [*archives,mp,checksum]:
        remote=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{tag}'],text=True));existing=next((a for a in remote['assets'] if a['name']==p.name),None)
        if existing:assert existing.get('digest')=='sha256:'+file_sha256(p) and existing['size']==p.stat().st_size,'Existing asset differs; refusing overwrite'
        else:run(['gh','release','upload',tag,str(p),'--repo',REPO])
    remote=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{tag}'],text=True));assets=[]
    for p in [*archives,mp,checksum]:
        a=next(a for a in remote['assets'] if a['name']==p.name);assert a.get('digest')=='sha256:'+file_sha256(p) and a['size']==p.stat().st_size
        assets.append(dict(name=p.name,bytes=p.stat().st_size,sha256=file_sha256(p)))
    atomic_json(dict(status='Done',verified=True,source_commit=commit,branch=branch,release_url=remote['html_url'],assets=assets,archive_sha256=file_sha256(archive),archive_bytes=archive.stat().st_size,files=len(files),reused_verified_releases=references),receipt)
    commit_push('Record verified remote GeomBelief experiment backup')

if __name__=='__main__':main()
