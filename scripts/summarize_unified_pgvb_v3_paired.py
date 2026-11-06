"""Posthoc full-2470 paired statistics; no model calls, model selection or partial scores."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256

def cluster_ci(delta,groups):
    rng=np.random.default_rng(42);units=list(groups.values());values=[]
    for _ in range(2000):
        ids=[i for j in rng.integers(0,len(units),len(units)) for i in units[j]]
        values.append(float(delta[ids].mean()))
    return np.quantile(values,[.025,.975]).tolist()

def pair(a,b,groups):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool);d=b.astype(float)-a.astype(float)
    return dict(n=len(a),fixes=int((~a&b).sum()),breaks=int((a&~b).sum()),net=int(d.sum()),delta=float(d.mean()),physical_target_group_95CI=cluster_ci(d,groups))

def main():
    p=argparse.ArgumentParser()
    for k in ['run','objects','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();folder=a.run/'val_seen_initial_full';s=json.loads((folder/'summary.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    assert s['coverage']=={'completed':2470,'expected':2470} and s['errors']==0,'Require complete, corrected full evaluation'
    assert file_sha256(a.objects)==manifest['objects_sha256'];objects=json.loads(a.objects.read_text())
    rows=[json.loads((folder/'rows'/f'{i:05d}.json').read_text()) for i in range(2470)];groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[tuple(r['episode_id'][:2])].append(i);r['target_category']=objects.get(r['episode_id'][0],{}).get(str(r['episode_id'][1]),{}).get('object_type','Unknown')
    result=dict(N=2470,training_seed=42,physical_target_groups=len(groups),bootstrap_replicates=2000,bootstrap_seed=42,
                uncertainty_scope='Episode/physical-target sampling only; does not measure training-seed variance',
                initial_pair=pair([r['geometry']['hits']['1'] for r in rows],[r['fused']['hits']['1'] for r in rows],groups),by_category={},by_map={},navigation=None)
    for key,output in [('target_category','by_category'),('map_name','by_map')]:
        for group in sorted({r[key] for r in rows}):
            subset=[r for r in rows if r[key]==group]
            result[output][group]=dict(N=len(subset),**{b:dict(Top1=float(np.mean([r[b]['hits']['1'] for r in subset])),NE=float(np.mean([r[b]['NE'] for r in subset]))) for b in ['geometry','fused']},fixes=sum(not r['geometry']['hits']['1'] and r['fused']['hits']['1'] for r in subset),breaks=sum(r['geometry']['hits']['1'] and not r['fused']['hits']['1'] for r in subset))
    nav={}
    nav_manifests={}
    for method in ['geometry','fused']:
        f=a.run/f'navigation_{method}'/'metrics.json'
        if f.exists():
            m=json.loads(f.read_text());assert m['coverage']=={'completed':2470,'expected':2470};assert m['manifest']['checkpoint_sha256']==manifest['checkpoint_sha256']
            nav_manifests[method]=m['manifest']
            nav[method]=[json.loads((f.parent/'rows'/f'{i:05d}.json').read_text()) for i in range(2470)]
            for i,r in enumerate(nav[method]):
                assert r['episode_id']==rows[i]['episode_id'];assert np.allclose(r['poses'][0],rows[i]['start_pose'],atol=1e-6,rtol=0)
    if len(nav)==2:
        for key in ['protocol','source_v2_sha256','stage1_sha256','config_sha256','annotations_sha256']:
            assert nav_manifests['geometry'][key]==nav_manifests['fused'][key], 'Paired navigation protocol mismatch: '+key
        result['navigation']=pair([r['metrics']['SR'] for r in nav['geometry']],[r['metrics']['SR'] for r in nav['fused']],groups)
    a.output.mkdir(parents=True,exist_ok=True);atomic_json(result,a.output/'paired_summary.json')
    with (a.output/'paired2470.csv').open('w') as f:
        fields=['index','episode_id','map_name','category','geometry_initial_hit','fused_initial_hit','geometry_initial_NE','fused_initial_NE','geometry_SR','fused_SR'];writer=csv.DictWriter(f,fields);writer.writeheader()
        for i,r in enumerate(rows):writer.writerow(dict(index=i,episode_id=json.dumps(r['episode_id']),map_name=r['map_name'],category=r['target_category'],geometry_initial_hit=r['geometry']['hits']['1'],fused_initial_hit=r['fused']['hits']['1'],geometry_initial_NE=r['geometry']['NE'],fused_initial_NE=r['fused']['NE'],geometry_SR=nav['geometry'][i]['metrics']['SR'] if len(nav)==2 else '',fused_SR=nav['fused'][i]['metrics']['SR'] if len(nav)==2 else ''))
    print(json.dumps(result['initial_pair'],indent=2))
if __name__=='__main__':main()
