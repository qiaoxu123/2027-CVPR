"""Safe native import/help and weight-file readiness; no navigation/training."""
from common import *
import subprocess,os

def main():
    native=BASELINES/'CityNav-GSM/external/citynav'
    python=BASELINES/'.reproduction-env/bin/python'
    command=[str(python),'main_goal_predictor.py','--help']
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.run(command,cwd=native,env=env,capture_output=True,text=True,timeout=30)
    log=OUT/'sources/mgp_native_preflight.log';log.write_text(proc.stdout+proc.stderr)
    model=BASELINES/'FlightGPT/model_weight/Flightgpt';index=read(model/'model.safetensors.index.json');shards=sorted(set(index['weight_map'].values()))
    evidence=dict(MGP=dict(command=command,cwd=str(native),returncode=proc.returncode,log=str(log),log_sha256=sha(log),
        result='Blocked: native helper cannot import groundingdino in the declared reproduction environment' if 'No module named \'groundingdino\'' in proc.stderr else 'Native help only; not a full inference validation',
        wrapper_restriction='Upstream main_goal_predictor.py unconditionally iterates Seen/Unseen/Test, performs terminal goal selection and writes native outputs. A split/ID/isolated-output/common-metric wrapper is not validated for recovered weight.'),
        FlightGPT=dict(snapshot=str(model),shards=[dict(file=n,present=(model/n).is_file()) for n in shards],complete=all((model/n).is_file() for n in shards)),
        HETT=read(OLD/'hett_asset_preflight.json'),HTNav=dict(official_weight_found=False,upstream_probe='sources/upstream_probe.json',local_independent_models_are_not_official=True),
        GeoNav=dict(local_model='DeepSeek Flash substitution',official_model='GPT-4o API',same_method_identity=False))
    dump(OUT/'baseline_preflight.json',evidence)
    rows=list(csv.DictReader((OUT/'baseline_results.csv').open()))
    for row in rows:
        if row['method']=='MGP':row['native_preflight']=evidence['MGP']['result']
    csvwrite(OUT/'baseline_results.csv',rows)
    print('MGP native help returncode',proc.returncode,'FlightGPT shards present',sum(r['present'] for r in evidence['FlightGPT']['shards']),'/',len(shards))

if __name__=='__main__':main()
