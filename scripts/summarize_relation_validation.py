"""Summarize paired full-Dev diagnostics, clustering four states per episode."""
import json
from pathlib import Path
import numpy as np
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--relation-dir',default='relation_dev_controls_20261101')
    parser.add_argument('--visual-dir',default='visual_dev_controls_20261101')
    parser.add_argument('--report',default='RELATION_VALIDATION_20261101.md')
    parser.add_argument('--summary',default='relation_validation_summary_20261101.json')
    args=parser.parse_args()
    root=Path('.')
    relation=Path(args.relation_dir)
    visual=Path(args.visual_dir)
    results={name:json.loads((folder/'controls.json').read_text()) for name,folder in [('relation',relation),('visual',visual)]}
    def read(folder,control):
        return json.loads((folder/f'predictions_{control}.json').read_text())
    def paired(first,second):
        assert first['dev_indices']==second['dev_indices']
        difference=(np.array(first['errors_m'])<=20).astype(float)-(np.array(second['errors_m'])<=20)
        groups=np.array(first['dev_indices'])//4
        _,inverse=np.unique(groups,return_inverse=True)
        sums=np.bincount(inverse,weights=difference);counts=np.bincount(inverse)
        rng=np.random.default_rng(0);draws=[]
        for _ in range(50):
            sample=rng.integers(0,len(sums),size=(100,len(sums)))
            draws.extend((100*sums[sample].sum(1)/counts[sample].sum(1)).tolist())
        return dict(difference_pp=100*difference.mean(),cluster_bootstrap_95ci_pp=np.quantile(draws,[.025,.975]).tolist(),helped=int((difference>0).sum()),harmed=int((difference<0).sum()),episodes=len(sums),states=len(difference))
    comparisons={
        'relation_vs_visual':paired(read(relation,'real'),read(visual,'real')),
        'relation_vs_zero':paired(read(relation,'real'),read(relation,'zero')),
        'relation_vs_no_relation':paired(read(relation,'real'),read(relation,'no_relation')),
        'relation_vs_shuffle':paired(read(relation,'real'),read(relation,'shuffle')),
    }
    Path(args.summary).write_text(json.dumps(dict(results=results,comparisons=comparisons),indent=2))
    report=Path(args.report)
    with report.open('a') as f:
        f.write('\n## 完整 Dev 结果\n\n|模型/干预|状态数|R@1 (20m)|平均误差 m|\n|---|---:|---:|---:|\n')
        for name,controls in results.items():
            for control,m in controls.items():
                f.write(f"|{name}/{control}|{m['samples']}|{100*m['corrected_R1']:.4f}%|{m['corrected_NE']:.4f}|\n")
        f.write('\n配对结果（按 episode 聚类、5,000 次 bootstrap；这些是重复前缀观测状态，非闭环 SR）：\n\n')
        for name,c in comparisons.items():
            f.write(f"- {name}: {c['difference_pp']:+.4f}pp，95%CI {c['cluster_bootstrap_95ci_pp']}，帮助 {c['helped']} / 损害 {c['harmed']} 状态。\n")
        f.write('\n## Real 条件的困难子集\n\n|模型/子集|样本数|R@1|平均误差m|\n|---|---:|---:|---:|\n')
        for name,folder in [('relation',relation),('visual',visual)]:
            for group,m in read(folder,'real').get('strata',{}).items():
                n=m['samples']
                if n:f.write(f"|{name}/{group}|{n}|{100*m['hits']/n:.4f}%|{m['error_sum']/n:.4f}|\n")
        f.write('\n以上为离线定位诊断，不能表述为闭环导航收益。关系模块的独立作用须同时对照同损失普通版与no_relation；关系词与静态分差仅为困难样本代理。\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
