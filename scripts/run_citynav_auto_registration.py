import argparse,json,os
from pathlib import Path
from tools.osm_map.auto_registration import run_job
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--controls',required=True);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=json.load(open(a.controls));r=run_job(a.data,d['city'],a.scene,d,a.output,os.environ.get('OSM_HTTP_PROXY'));print(json.dumps({k:r.get(k) for k in ['map_name','structure_cost_initial','structure_cost_final','manual_validation','validation','verified']},ensure_ascii=False))
