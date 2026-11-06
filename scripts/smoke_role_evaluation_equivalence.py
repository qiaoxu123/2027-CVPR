"""Real dev-only numeric equivalence check of efficient evaluation; no new fitting."""
import argparse,json
from pathlib import Path
import torch
from scripts.train_role_polygon_bearing import PublicBatch,evaluate
from scripts.role_bearing_evaluation import evaluate_cpu_metrics
from graphnav.relation_bearing_network import RolePolygonBearing


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 cache=PublicBatch(a.run);labels=torch.load(a.run/'PRIVATE_dev_labels.pt',weights_only=True);ids=json.load(open(a.run/'data/split.json'))['indices']['dev'][:16]
 state=torch.load(a.run/'angle_only/relations/best.pt',weights_only=True,map_location='cpu');model=RolePolygonBearing().cuda().eval().requires_grad_(False);model.load_state_dict(state['model'])
 old,orows=evaluate(model,cache,ids,labels,'relations',16);new,nrows=evaluate_cpu_metrics(model,cache,ids,labels,'relations',16)
 assert [(v['pred'],v['top1'],v['top2']) for v in orows]==[(v['pred'],v['top1'],v['top2']) for v in nrows]
 delta=max(abs(x['NLL']-y['NLL']) for x,y in zip(orows,nrows));error=max(abs(x['error']-y['error']) for x,y in zip(orows,nrows));assert delta<1e-5 and error<1e-3
 out=dict(N=len(orows),discrete_predictions_bitwise_equal=True,max_NLL_delta=delta,max_error_degree_delta=error,source='real TRAIN development, frozen current checkpoint');(a.run/'evaluation_equivalence.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
