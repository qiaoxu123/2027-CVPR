"""Scientific figures from saved TRAIN-only sensitivity results."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',required=True);p.add_argument('--algebra',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise ValueError('Do not overwrite figures')
    out.mkdir(parents=True)
    data=json.loads(Path(a.summary).read_text());alg=json.loads(Path(a.algebra).read_text())
    proxy=data['Oracle_hypothesis_eligible'];states=proxy['states'];stages=proxy['stages'];counts=[stages[k]['covered'] for k in ['raw900','valid900','nms','top15','top30','top60']]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].bar(['900','Valid','NMS','Top15','Top30','Top60'],counts,color='steelblue');axes[0].set_title(f'Registered-box coverage / {states} UNREVIEWED states');axes[0].set_ylim(0,states)
    for i,n in enumerate(counts):axes[0].text(i,n+3,str(n),ha='center')
    labels=['Native15','A+box','B+box','C bound','C ideal'];names=['Native15','A_box_exists','B_added_box_selected','C_target_bounded','C_target_unbounded']
    axes[1].bar(labels,[proxy['methods'][k]['R1']*100 for k in names],color=['gray','orange','gold','seagreen','lightgreen']);axes[1].set_ylim(0,105);axes[1].set_title('POSITION HYPOTHESIS R@1 (%) / not GT oracle')
    names=['original_conditional_clipped','conditional_unclipped','global_prior_clipped','global_prior_unclipped','ideal_density_observed_only'];labels=['Original','No clip','Global P0','Both','Ideal V']
    vals=[alg['summary']['A0_wrong'][k]['corrected_A0'] for k in names]
    axes[2].bar(labels,vals,color='seagreen');axes[2].set_ylim(0,105);axes[2].set_title('C: corrected A0 errors /99, ideal position supplied')
    for i,n in enumerate(vals):axes[2].text(i,n+1,str(n),ha='center')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('TRAIN automatic projection sensitivity — human reviewed=0; no model trained',fontsize=10);fig.tight_layout();fig.savefig(out/'diagnosis.png',dpi=150);fig.savefig(out/'diagnosis.pdf');plt.close(fig)


if __name__=='__main__':main()
