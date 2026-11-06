"""Independently check archived report invariants without the original server."""
import argparse,csv,gzip,hashlib,json,math
from pathlib import Path
import numpy as np


def verify(root):
    summary=json.loads((root/'summary.json').read_text())
    rows=list(csv.DictReader((root/'per_episode.csv').open(encoding='utf-8-sig')))
    with gzip.open(root/'verified_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        evidence=[json.loads(s) for s in f]
    assert len(rows)==len(evidence)==summary['count']==2470
    by_id={tuple(r['episode_id']):r for r in evidence}
    assert len(by_id)==2470
    assert len({tuple(json.loads(r['episode_id'])) for r in rows})==2470
    for r in rows:
        e=by_id[tuple(json.loads(r['episode_id']))]
        assert e['index']==int(r['index'])
        assert e['public_record']['instruction']==r['instruction']
        assert e['public_record']['map_name']==r['map_name']
        assert [x['id'] for x in e['public_record']['landmarks']]==json.loads(r['reference_ids'])
        assert len(e['public_record']['landmarks'])==int(r['bound_reference_count'])
        ne=math.dist(e['poses'][-1][:2],e['target_position_POSTHOC'][:2])
        assert math.isclose(ne,float(r['baseline_NE_m']),abs_tol=1e-6,rel_tol=0)
        assert e['metrics']['NE']==float(r['baseline_NE_m'])
        assert e['metrics']['SR']==int(r['baseline_SR'])==int(ne<=20)
        assert e['poses'][-1]==json.loads(r['final_pose'])
        assert e['poses'][0]==json.loads(r['start_pose'])
    assert sum(int(r['baseline_SR']) for r in rows)==1130
    assert sum(not int(r['bound_reference_count']) for r in rows)==126
    assert sum(not int(r['baseline_SR']) and not int(r['bound_reference_count']) for r in rows)==122
    for name,bound in [('all',None),('no_anchor',False),('with_anchor',True)]:
        selected=[r for r in rows if bound is None or bool(int(r['bound_reference_count']))==bound]
        section=summary['sections'][name]
        assert len(selected)==section['N']
        assert sum(1-int(r['baseline_SR']) for r in selected)==section['failures']
        values=np.array([float(r['baseline_NE_m']) for r in selected])
        q=section['NE_all']
        assert np.allclose(np.quantile(values,[.5,.75,.9,.95,.99]),[q[k] for k in ['median_m','p75_m','p90_m','p95_m','p99_m']])
        assert math.isclose(values.mean(),q['mean_m'])
    bins=list(csv.DictReader((root/'distance_bins.csv').open(encoding='utf-8-sig')))
    assert sum(int(r['all_tasks']) for r in bins)==2470
    assert sum(int(r['failures']) for r in bins)==1340
    assert sum(int(r['no_anchor_failures']) for r in bins)==122
    far=[r for r in rows if float(r['baseline_NE_m'])>=100 and not int(r['baseline_SR'])]
    assert len(far)==152 and sum(not int(r['bound_reference_count']) for r in far)==90
    cases=json.loads((root/'cases.json').read_text())
    assert len(cases)==8
    gallery=(root/'cases.html').read_text()
    assert gallery.count('<article ')==8 and gallery.count('<svg ')==8
    for c in cases:
        e=by_id[tuple(c['episode_id'])]
        assert c['poses']==e['poses'] and c['all_bound_polygons']==e['public_record']['landmarks']
        assert (root/f"case_{c['index']:04d}.png").is_file()
    for name in ['binding_status_sr','failure_distance_distribution']:
        for ext in ['png','svg']:assert (root/f'{name}.{ext}').is_file()
    return dict(episodes=2470,unique_episode_ids=2470,successes=1130,failures=1340,
        no_anchor=126,no_anchor_failures=122,far_failures=152,far_no_anchor=90,
        max_NE_m=max(float(r['baseline_NE_m']) for r in rows),case_count=8,
        check='PASS: independent archived evidence, per-episode outcomes, quantiles, denominators and gallery')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path)
    print(json.dumps(verify(p.parse_args().report),ensure_ascii=False,indent=2))
