"""Back up finished map training/static results while navigation continues."""
import json,subprocess,tempfile
from pathlib import Path
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.workspace_paths import WORKSPACE
from scripts.geombelief_map_experiments import OUT
from scripts.geombelief_sync import commit_push,REPO,run

def main():
    assert (OUT/'all_static_complete.json').exists()
    receipt=OUT/'training_static_remote_sync.json'
    if receipt.exists():return
    commit,branch=commit_push('Archive complete GeomBelief static geometry evidence and reproducible figures')
    files=[]
    for sub in ('training','localization'):
        files.extend(p for p in (OUT/sub).rglob('*') if p.is_file())
    for name in ('frozen_plan.json','all_training_frozen.json','all_static_complete.json','train_dev_manifest.json','train_seen_query_manifest.json','val_seen_query_manifest.json','val_unseen_query_manifest.json','train_seen_shuffle_manifest.json','val_seen_shuffle_manifest.json','val_unseen_shuffle_manifest.json','shuffle_effectiveness.json','MAP_GEOMETRY_INPUT_AUDIT.md'):
        files.append(OUT/name)
    files=sorted(files)
    manifest=OUT/'GEOMBELIEF_STATIC_ASSET_MANIFEST.json'
    atomic_json(dict(source_commit=commit,scope='15 completed trainings and 30 static split evaluations; Correct Geometry weights reused from Polygon static release',files=[dict(path=str(p.relative_to(WORKSPACE)),bytes=p.stat().st_size,sha256=file_sha256(p)) for p in files]),manifest)
    tmp=Path(tempfile.mkdtemp(prefix='geombelief-static-'));listing=tmp/'files.txt';listing.write_text(''.join(str(p.relative_to(WORKSPACE))+'\n' for p in files))
    archive=tmp/'geombelief-training-static.tar.zst'
    run(['tar','--zstd','--mtime=@0','--owner=0','--group=0','--numeric-owner','-cf',str(archive),'-C',str(WORKSPACE),'-T',str(listing)])
    assert archive.stat().st_size<2*1024**3
    tag='geombelief-static-20261017-'+file_sha256(OUT/'frozen_plan.json')[:8]
    notes=tmp/'notes.md';notes.write_text('Complete, frozen three-seed map input experiments: 15 new training jobs and 30 complete static evaluations. Original Correct Geometry three seeds reused, not retrained. Navigation remains running and is not claimed complete in this snapshot. Source '+commit+'.')
    if subprocess.run(['gh','release','view',tag,'--repo',REPO],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        run(['gh','release','create',tag,'--repo',REPO,'--target',commit,'--title','GeomBelief completed map training and static validation','--notes-file',str(notes)])
    for p in (archive,manifest):
        remote=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{tag}'],text=True));a=next((a for a in remote['assets'] if a['name']==p.name),None)
        if a:assert a.get('digest')=='sha256:'+file_sha256(p) and a['size']==p.stat().st_size
        else:run(['gh','release','upload',tag,str(p),'--repo',REPO])
    remote=json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/releases/tags/{tag}'],text=True))
    for p in (archive,manifest):
        a=next(a for a in remote['assets'] if a['name']==p.name);assert a.get('digest')=='sha256:'+file_sha256(p) and a['size']==p.stat().st_size
    atomic_json(dict(status='Done',verified=True,source_commit=commit,branch=branch,release_url=remote['html_url'],archive_sha256=file_sha256(archive),archive_bytes=archive.stat().st_size,files=len(files),navigation_status='Running; excluded from completed static snapshot'),receipt)
    commit_push('Record verified remote backup of all GeomBelief map checkpoints')
if __name__=='__main__':main()
