"""Export sampled GPU telemetry for completed jobs and timestamped epochs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeline',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('refusing telemetry report overwrite')
    raw=Path(args.timeline).read_bytes()
    # Snapshot a running monitor safely; an incomplete final record is ignored.
    if not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
    records=list(csv.DictReader(io.StringIO(raw.decode())))
    t=np.array([float(r['unix']) for r in records]);util=np.array([float(r['gpu_util_percent']) for r in records])
    memory=np.array([float(r['memory_mib']) for r in records]);power=np.array([float(r['power_w']) for r in records])
    def interval(name,kind,start,end):
        select=(t>=start)&(t<=end);size=int(select.sum())
        return dict(name=name,kind=kind,started_unix=start,finished_unix=end,seconds=end-start,
            telemetry_samples=size,gpu_util_mean_percent=float(util[select].mean()) if size else None,
            gpu_util_p95_percent=float(np.quantile(util[select],.95)) if size else None,
            sampled_memory_peak_mib=float(memory[select].max()) if size else None,
            power_mean_w=float(power[select].mean()) if size else None)
    stages=[]
    for path in sorted(Path('runs').glob('*queue.status.json')):
        state=json.loads(path.read_text())
        for job in state['jobs']:
            if job['status']=='complete':stages.append(interval(job['name'],'job_includes_startup_and_validation',job['started'],job['started']+job['seconds']))
    for folder in sorted(Path('runs').glob('*')):
        path=folder/'history.json'
        if not path.is_file():continue
        rows=json.loads(path.read_text())
        if not isinstance(rows,list):continue
        for row in rows:
            for split in ('train','dev','seen'):
                result=row.get(split,{})
                if isinstance(result,dict) and 'started_unix' in result and 'finished_unix' in result:
                    stages.append(interval(f'{folder.name}/epoch{row["epoch"]}/{split}','timestamped_epoch',result['started_unix'],result['finished_unix']))
    output.mkdir(parents=True)
    with (output/'stages.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(stages[0]));writer.writeheader();writer.writerows(stages)
    (output/'summary.json').write_text(json.dumps(dict(timeline_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        samples=len(records),first_unix=float(t[0]),last_unix=float(t[-1]),stages=len(stages),
        note='nvidia-smi sampled approximately every2seconds; short epochs may have0samples. Sampled memory includes CUDA process reservations and differs from torch peak allocated. Job windows include preparation/validation. Earlier P0/P1 source training predates this monitor and cannot be reconstructed. Running/incomplete jobs excluded.'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True,layout='constrained');hours=(t-t[0])/3600
    axes[0].plot(hours,util,lw=.6);axes[0].set_ylabel('GPU utilization (%)');axes[0].set_ylim(0,100)
    axes[1].plot(hours,memory/1024,lw=.7);axes[1].set_ylabel('Device memory (GiB)');axes[1].set_xlabel('Hours since telemetry start')
    fig.savefig(output/'timeline.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(output=str(output),stages=len(stages),samples=len(records))))


if __name__=='__main__':main()
