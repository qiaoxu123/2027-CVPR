"""Apply the predeclared verified Train supervision gate after zero-shot evaluation."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from experiments.visual_grounding.freeze_dataset import read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    verified_path=data/'verified_attributes.json';verified=json.loads(verified_path.read_text()) if verified_path.exists() else []
    verified=[r for r in verified if r.get('reviewer_kind')=='independent_human' and r.get('split')=='train_seen']
    status=dict(status='stopped_before_training',verified_train_attribute_samples=len(verified),adapter_epochs_run=0,lora_epochs_run=0,trainable_parameters_trained=0,training_time_seconds=0,zero_shot_adapter_lora_comparison=None,reason='No independently image-verified Train attribute labels. Conservative CityRefer target binding supplies language weak supervision only; unknown negatives and mixed-instance ROIs cannot be made reliable by training. The predeclared gate requires identity/region-disjoint Train/Dev with >=2 distinct observed instances per attribute class.',data_source=str(data.resolve()),protocol_sha256=sha('experiments/visual_grounding/protocol.json'),recommendation='Retain native SigLIP2; annotate pure visible roof/vehicle instances and negatives before a supervised semantic adapter pilot. Existing P2 identity head is evaluated separately, not a new semantic pilot.')
    if verified:raise ValueError('verified labels present: manual gate assessment required, refusing automatically stop')
    (out/'gate.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
if __name__=='__main__':main()
