"""Read saved frozen predictions only; no inference or selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 o=Path('reports/relation_bearing');s=json.load(open(o/'results.json'));data={}
 with (o/'confirmation_predictions.csv').open() as f:
  import csv
  for x in csv.DictReader(f):data.setdefault(x['method'],[]).append(x)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,k in zip(axes,['old','matching_angle','relations_angle']):
  cm=np.zeros((8,8))
  for x in data[k]:cm[int(x['true_bin']),int(x['pred'])]+=1
  cm/=np.maximum(cm.sum(1,keepdims=True),1);im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');ax.set(title=k,xlabel='Predicted map bin',ylabel='True map bin')
 fig.colorbar(im,ax=axes.tolist(),shrink=.7);fig.savefig(o/'confusion_matrices.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));xs=np.arange(3);width=.24
 for j,k in enumerate(['old','matching_angle','relations_angle']):
  ax.bar(xs+(j-1)*width,[100*s['strata'][k][g]['macro_Top1'] for g in ['direct','indirect','unknown']],width,label=k)
 ax.set_xticks(xs,['Direct (proxy)','Indirect (proxy)','Unknown']);ax.set_ylabel('Query macro Top-1 (%)');ax.legend();fig.tight_layout();fig.savefig(o/'relation_strata.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
