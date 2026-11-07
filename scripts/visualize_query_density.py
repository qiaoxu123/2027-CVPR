"""Posthoc TRAIN query-density cases, including corrections and damage."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('results','data','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows={n:json.loads((a.results/f'rows_{n}.json').read_text())['rows'] for n in (28,56)}
    fields={n:np.load(a.results/f'public_fields_{n}.npz')['logits'] for n in (28,56)}
    public=[json.loads(s) for s in (a.data/'public_train.jsonl').read_text().splitlines()]
    labels=json.loads((a.data/'labels_private.json').read_text())
    corrected=[];damaged=[]
    for k,(x,y) in enumerate(zip(rows[28],rows[56])):
        assert x['index']==y['index']
        if x['local']['NE']>20 and y['local']['NE']<=20:corrected.append(k)
        if x['local']['NE']<=20 and y['local']['NE']>20:damaged.append(k)
    fragments=[]
    for category,indices in [('correction',corrected[:4]),('damage',damaged[:4])]:
        for k in indices:
            i=rows[28][k]['index'];q=public[i];bounds=np.array(q['world_bounds']);target=labels[i]['target_world']
            fig,axes=plt.subplots(1,2,figsize=(9,4))
            for ax,n in zip(axes,(28,56)):
                z=fields[n][k].reshape(n,n);prob=np.exp(z-z.max());prob/=prob.sum()
                ax.imshow(prob,extent=[bounds[0],bounds[2],bounds[1],bounds[3]],origin='lower',cmap='magma')
                for landmark in q['landmarks']:
                    poly=np.asarray(landmark['contour'])*(bounds[2:]-bounds[:2])+bounds[:2];poly=np.vstack([poly,poly[0]])
                    ax.plot(poly[:,0],poly[:,1],color='cyan',linewidth=.7)
                ax.scatter(*target,color='lime',marker='*',label='GT (posthoc)')
                for decoder,color in [('grid','white'),('local','yellow')]:
                    ax.scatter(*rows[n][k][decoder]['xy'],color=color,marker='x',label=decoder)
                ax.set_title(f'{n}x{n}; local NE={rows[n][k]["local"]["NE"]:.1f}m');ax.legend(fontsize=6)
                ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]))
            fig.tight_layout();name=f'{category}_{i:05d}.png';fig.savefig(a.output/name,dpi=140);plt.close(fig)
            fragments.append(f'<h3>{category}: {i}</h3><p>{html.escape(q["instruction"])}</p><a href="{name}"><img width="900" src="{name}"></a>')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen query density</title><h1>TRAIN posthoc paired cases</h1><p>Public polygons: cyan; GT scoring marker: green; grid goal: white; local goal: yellow. No RGB or navigation experiment.</p>'+''.join(fragments))
    print(json.dumps(dict(corrections=len(corrected),damage=len(damaged),visualized=len(fragments))))
if __name__=='__main__':main()
