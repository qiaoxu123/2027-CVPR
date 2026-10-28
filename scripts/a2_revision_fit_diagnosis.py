"""Frozen v1 FIT-only failure diagnosis; never reselect a deployed gate."""
import json
import numpy as np
import torch
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.revision_calibrate import load_packet
from experiments.a2_waldo_landmark_belief.revision_fusion import bounded_belief_mix


def auc(positive,negative):
    if not positive or not negative:return None
    a=np.asarray(positive)[:,None];b=np.asarray(negative)[None,:]
    return float(((a>b)+.5*(a==b)).mean())


def main():
    torch.set_num_threads(2)
    root=BASE/'revision_v1';directory=root/'calibration/packets';manifest=json.loads((directory/'manifest.json').read_text())
    raw=json.loads((root/'confirmation/manifest.json').read_text());sources={e['filename']:e for e in raw['entries']}
    signature=sha(directory/'signature.json');records=[];count={k:0 for k in ('views','available','a0_correct','q_correct','max_gate_corrected','max_gate_broken','oracle_recoverable','q_expected_helpful')}
    positive=[];negative=[];by_group={}
    for entry in manifest['entries']:
        source=sources[entry['filename']]
        if source['fold']!='fit':continue
        # Raw source hash is verified by the committed packet sidecar.
        path=directory/entry['filename'];bound=json.loads(path.with_suffix('.sha.json').read_text())
        if sha(path)!=entry['sha256'] or sha(path.with_suffix('.sha.json'))!=entry['sidecar_sha256'] or bound['source_row_sha256']!=source['sha256']:
            raise ValueError('Committed FIT packet provenance changed')
        packet=load_packet(path,signature,bound['source_row_sha256'])
        if packet['fold']!='fit':raise ValueError('Cannot diagnose verification packets')
        base,q,grid,target=(packet[k].float() for k in ('base','q','grid','target'))
        distance=(grid-target).norm(dim=-1);before=float(distance[base.argmax()]);probability=float(packet['benefit_logit'].sigmoid())
        count['views']+=1;count['a0_correct']+=int(before<=20)
        if not packet['available']:continue
        count['available']+=1;count['q_correct']+=int(distance[q.argmax()]<=20)
        after=float(distance[bounded_belief_mix(base,q,.25*probability).argmax()])
        gain=before>20 and after<=20;harm=before<=20 and after>20
        recoverable=before>20 and any(distance[bounded_belief_mix(base,q,g).argmax()]<=20 for g in (.025,.0625,.125,.25))
        expected=bool((q*distance).sum()+1<(base.softmax(-1)*distance).sum())
        count['max_gate_corrected']+=int(gain);count['max_gate_broken']+=int(harm);count['oracle_recoverable']+=int(recoverable);count['q_expected_helpful']+=int(expected)
        if gain:positive.append(probability)
        if harm:negative.append(probability)
        by_group.setdefault(tuple(packet['group']),dict(gain=False,harm=False,recoverable=False))
        group=by_group[tuple(packet['group'])];group['gain']|=gain;group['harm']|=harm;group['recoverable']|=bool(recoverable)
        records.append(dict(group=packet['group'],episode=packet['episode'],step=packet['step'],before_ne=before,after_ne=after,
                            benefit_probability=probability,corrected=gain,broken=harm,oracle_recoverable=bool(recoverable),expected_ne_helpful=expected))
    report=dict(scope='Opened v1 FIT127 targets only; frozen cached proposals; CPU diagnosis, no new gate or benchmark selection',counts=count,
                groups={k:sum(bool(g[k]) for g in by_group.values()) for k in ('gain','harm','recoverable')},
                benefit_gain_vs_harm_auc=auc(positive,negative),
                gain_probability_mean=float(np.mean(positive)) if positive else None,harm_probability_mean=float(np.mean(negative)) if negative else None,
                oracle_caveat='GT chooses best of4 fixed mixture weights; discrete recoverable opportunities, not a continuous-weight capacity bound or deployable gain',
                source_manifest_sha256=sha(directory/'manifest.json'),checkpoint_sha256=sha(root/'training/best.pt'),records=records)
    save(report,ROOT/'reports/a2_execution_20261027/REVISION_V1_FIT_DIAGNOSIS.json')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
