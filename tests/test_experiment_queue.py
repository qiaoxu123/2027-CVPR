import json
from pathlib import Path
import subprocess
import sys
import time


def test_running_queue_imports_are_isolated_from_later_source_edits(tmp_path):
    (tmp_path/'graphnav').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'graphnav/__init__.py').write_text('')
    marker=tmp_path/'graphnav/marker.py';marker.write_text('VALUE=1\n')
    (tmp_path/'graphnav/entry.py').write_text("from pathlib import Path\nimport time\nPath('ready').touch()\ntime.sleep(.3)\nfrom graphnav.marker import VALUE\nPath('done').write_text(str(VALUE))\n")
    subprocess.run(['git','init','-q'],cwd=tmp_path,check=True)
    subprocess.run(['git','add','.'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=tmp_path,check=True)
    manifest=tmp_path/'queue.json';manifest.write_text(json.dumps(dict(jobs=[dict(name='late_import',
        command=[sys.executable,'-m','graphnav.entry'],log='job.log',complete_artifact='done')])))
    runner=Path(__file__).resolve().parents[1]/'scripts/run_experiment_queue.py'
    process=subprocess.Popen([sys.executable,str(runner),'--manifest','queue.json'],cwd=tmp_path)
    deadline=time.monotonic()+5
    while not (tmp_path/'ready').exists() and time.monotonic()<deadline and process.poll() is None:time.sleep(.01)
    assert (tmp_path/'ready').exists()
    marker.write_text('VALUE=2\n')
    assert process.wait(timeout=5)==0 and (tmp_path/'done').read_text()=='1'
    status=json.loads((tmp_path/'queue.status.json').read_text())
    assert status['status']=='complete' and Path(status['jobs'][0]['source_snapshot']).is_dir()


def test_evaluations_overlap_but_training_is_a_barrier(tmp_path):
    (tmp_path/'graphnav').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'graphnav/__init__.py').write_text('')
    (tmp_path/'graphnav/entry.py').write_text("import sys,time\nfrom pathlib import Path\nname=sys.argv[1]\nPath(name+'.started').write_text(str(time.time()))\ntime.sleep(.4)\nPath(name+'.done').write_text(str(time.time()))\n")
    subprocess.run(['git','init','-q'],cwd=tmp_path,check=True)
    subprocess.run(['git','add','.'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=tmp_path,check=True)
    jobs=[dict(name=name,kind='evaluation' if name!='train' else 'train',
        command=[sys.executable,'-m','graphnav.entry',name],log=name+'.log',complete_artifact=name+'.done')
        for name in ('eval0','eval1','train','eval2')]
    (tmp_path/'queue.json').write_text(json.dumps(dict(evaluation_workers=2,jobs=jobs)))
    runner=Path(__file__).resolve().parents[1]/'scripts/run_experiment_queue.py'
    subprocess.run([sys.executable,str(runner),'--manifest','queue.json'],cwd=tmp_path,check=True,timeout=10)
    started={name:float((tmp_path/(name+'.started')).read_text()) for name in ('eval0','eval1','train','eval2')}
    done={name:float((tmp_path/(name+'.done')).read_text()) for name in started}
    assert started['eval0']<done['eval1'] and started['eval1']<done['eval0']
    assert started['train']>=max(done['eval0'],done['eval1'])
    assert started['eval2']>=done['train']
    status=json.loads((tmp_path/'queue.status.json').read_text())
    assert status['status']=='complete' and all(j['returncode']==0 for j in status['jobs'])
