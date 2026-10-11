"""Small scientific evidence artifacts; large model/features remain local."""
import argparse,csv,gzip,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .data import write


def run(a):
    root=Path(a.runs);out=Path(a.assets);out.mkdir(parents=True,exist_ok=True);evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    for p in sorted(root.rglob('*.json')):
        if any(v in p.parts for v in ['environment_private','renderer_private']):continue
        dest=evidence/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in sorted(root.rglob('*predictions.jsonl')):
        dest=evidence/p.relative_to(root);dest=dest.with_suffix('.jsonl.gz');dest.parent.mkdir(parents=True,exist_ok=True)
        compact=[]
        for line in p.read_text().splitlines():
            item=json.loads(line)
            if 'judgment_prob' in item:item['judgment_class']=[int(np.argmax(v)) for v in item.pop('judgment_prob')]
            for key in ['scores','score','match_prob','match_probability']:
                if key in item:item[key]=[round(v,6) for v in item[key]]
            compact.append(json.dumps(item,separators=(',',':'))+'\n')
        with gzip.GzipFile(filename=str(dest),mode='wb',mtime=0) as f:f.write(''.join(compact).encode())
    for p in sorted(root.glob('*.log')):(evidence/p.name).write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    if (root/'cases').exists():
        cases=out/'cases';cases.mkdir(exist_ok=True)
        for p in (root/'cases').glob('*.png'):shutil.copy2(p,cases/p.name)
        shutil.copy2(root/'cases/cases.json',cases/'cases.json')
    m=json.loads((root/'evaluation_seen/metrics.json').read_text());figs=out/'figures';figs.mkdir(exist_ok=True);rows=[]
    for name,v in m['models'].items():
        row=dict(model=name)
        for k in ['identity_r1','same_class_r1','same_color_r1','attribute_r1','goal_r1','goal_r4','goal_r8','ne','recovery','damage','net_recovery']:row[k]=v[k]['mean']
        rows.append(row)
    with (evidence/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    names=[n for n in m['models'] if n.endswith('_real')];labels=[n.removesuffix('_real') for n in names]
    for metric,title in [('identity_r1','Original target identity (weak)'),('goal_r1','Goal localization (XY20m)')]:
        values=[(m['models'][n][metric]['family_equal_weight']['mean'] if metric=='identity_r1' else m['models'][n][metric]['mean'])*100 for n in names];fig,ax=plt.subplots(figsize=(10,4));ax.bar(labels,values);ax.set(ylabel='R@1 (%)',title=title+(' — family weighted' if metric=='identity_r1' else ' — state weighted'));ax.tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(figs/(metric+'.png'),dpi=160);plt.close(fig)
    selected=m['selected'];ns=[selected+'_'+v for v in ['real','zero','shuffled']]+['geometry_only','identity_oracle'];fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axs,['identity_r1','goal_r1']):
        values=[m['models'][n][metric]['family_equal_weight']['mean'] if metric=='identity_r1' else m['models'][n][metric]['mean'] for n in ns]
        ax.bar(['Real','Black RGB','Shuffled','Geometry','Identity Oracle'],[np.nan if v is None else v*100 for v in values]);ax.set_title(metric+(' (family weighted)' if metric=='identity_r1' else ' (state weighted)'));ax.set_ylabel('R@1 (%)');ax.tick_params(axis='x',rotation=25)
        for j,v in enumerate(values):
            if v is None:ax.text(j,2,'N/A',ha='center')
    fig.tight_layout();fig.savefig(figs/'controls.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(names));ax.bar(x-.2,[m['models'][n]['recovery']['mean']*640 for n in names],.4,label='Recovery');ax.bar(x+.2,[m['models'][n]['damage']['mean']*640 for n in names],.4,label='Damage');ax.set_xticks(x,labels,rotation=30);ax.legend();fig.tight_layout();fig.savefig(figs/'safety.png',dpi=160);plt.close(fig)
    runtime=json.loads((root/'runtime/metrics.json').read_text())['models'];fig,axs=plt.subplots(1,2,figsize=(11,4));names=list(runtime)
    for ax,key,label in [(axs[0],'median_ms','Cached-head latency (ms)'),(axs[1],'peak_allocated_bytes','Peak allocated GPU (MiB)')]:
        ax.bar(names,[runtime[n][key]/(2**20 if key=='peak_allocated_bytes' else 1) for n in names]);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(figs/'runtime.png',dpi=160);plt.close(fig)
    write(out/'sha256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='sha256.json'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--assets',required=True);run(p.parse_args())
if __name__=='__main__':main()
