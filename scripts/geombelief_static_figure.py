"""Paper figure from completed, matched three-seed static experiments."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.geombelief_results import map_paths,OUT

def main():
    assert (OUT/'all_static_complete.json').exists()
    arms=['name_type','shuffled','direction','distance','distance_direction','correct']
    labels=['Name / type','Shuffled geometry','Direction','Distance','Distance + direction','Correct geometry']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True);sources=[]
    for ax,split,title in zip(axes,['val_seen','val_unseen'],['Validation Seen (2,426 queries)','Validation Unseen (2,679 queries)']):
        values=[]
        for arm in arms:
            seeds=[]
            for seed in (0,1,2):
                metric,trace,cp=map_paths(arm,seed,split,'localization')
                seeds.append(json.loads(metric.read_text())['metrics']['R@1']*100)
                sources.append(dict(arm=arm,split=split,seed=seed,metric_path=str(metric),metric_sha256=file_sha256(metric),checkpoint_sha256=file_sha256(cp),R1_percent=seeds[-1]))
            values.append(seeds)
        values=np.asarray(values)
        ax.bar(np.arange(6),values.mean(1),yerr=values.std(1,ddof=1),capsize=3,color=['#89939d','#b1a0b9','#78a6d0','#4a85b7','#346a99','#db9045'])
        for k,v in enumerate(values.mean(1)):ax.text(k,v+values.std(1,ddof=1)[k]+.8,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_xticks(np.arange(6),labels,rotation=35,ha='right');ax.set_title(title);ax.set_ylim(0,48);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('R@1 within 20 m (%)')
    fig.suptitle('Language-grounded map geometry: matched training budget')
    fig.text(.5,.015,'Mean ± sample SD over 3 seeds. Error bars are not map-generalization confidence intervals.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.95]);dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(dest/f'map_geometry_static.{ext}',dpi=180,bbox_inches='tight')
    atomic_json(dict(seeds=[0,1,2],error_bars='sample standard deviation, not confidence interval',sources=sources),OUT/'static_figure_manifest.json')
if __name__=='__main__':main()
