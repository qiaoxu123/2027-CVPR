"""Local GPU telemetry; stopped only by its own explicit sentinel file."""
import argparse
import csv
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--stop-file',required=True)
    p.add_argument('--interval',type=float,default=2.);args=p.parse_args()
    path=Path(args.output)
    if path.exists():raise ValueError('refusing to overwrite telemetry')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(['utc','unix','gpu_util_percent','memory_mib','power_w'])
        while not Path(args.stop_file).exists():
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
            w.writerow([datetime.now(timezone.utc).isoformat(),time.time(),*raw.strip().split(',')]);f.flush()
            time.sleep(args.interval)


if __name__=='__main__':main()
