"""Every-step old E4 RGB and probability maps for four specified tasks; no inference."""
import argparse,html,json,shutil
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.landmark_projection import world_to_frame


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tasks={t['index']:t for t in json.loads(a.tasks.read_text())};rows=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
 for r in rows:
  if r['method']!='E4_PGVB' or r['index'] not in [1,13,34,39]:continue
  t=tasks[r['index']];folder=a.output/f'task_{r["index"]:02d}';folder.mkdir(parents=True,exist_ok=True);names=[]
  for k,d in enumerate(r['step_diagnostics']):
   source=Path(d['audit_artifact']);z=np.load(source);rgb=cv2.cvtColor(cv2.imread(str(source.with_suffix('.png'))),cv2.COLOR_BGR2RGB);shutil.copyfile(source.with_suffix('.png'),folder/f'step_{k:02d}_rgb.png');fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(rgb)
   for poly in t['public_polygons']:
    xy=world_to_frame(np.asarray(poly),z['footprint_world'],rgb.shape);axes[0].plot(xy[:,0],xy[:,1],'c-',lw=.7)
   axes[0].set_xlim(0,383);axes[0].set_ylim(383,0);axes[0].set_title('Current native384 RGB + public Polygon')
   b=z['bounds'];extent=[b[0],b[2],b[1],b[3]];logits=z['final_logits'];prob=np.exp(logits-logits.max());prob/=prob.sum()
   for ax,value,title in [(axes[1],z['pgvb_probability'] if 'pgvb_probability' in z else None,'PGVB conditional probability'),(axes[2],prob,'Actual fused controller Belief')]:
    if value is None:ax.text(.5,.5,'No valid PGVB evidence',transform=ax.transAxes,ha='center')
    else:ax.imshow(value.reshape(28,28),origin='lower',extent=extent,vmin=0,vmax=max(float(value.max()),1e-8),cmap='magma')
    for poly in t['public_polygons']:
     xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],'c-',lw=.7)
    fp=np.vstack((z['footprint_world'],z['footprint_world'][0]));ax.plot(fp[:,0],fp[:,1],ls='--',color='gray',lw=.7);ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3]);ax.set_title(title)
   axes[2].scatter(*d['selected_goal_xy'],c='red',s=18);fig.suptitle(f'Task {r["index"]} Step {k} | Goal NE {d["public_landmark_goal_distance_m"]:.2f}m | Stop p {d["stop_probability"]:.3f} | To Goal {d["predicted_goal_distance"]:.3f}m');fig.tight_layout();file=f'step_{k:02d}.png';fig.savefig(folder/file,dpi=105);plt.close(fig);names.append(file)
  page='<meta charset="utf-8"><title>逐步PGVB审查</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}img{max-width:100%}</style><h1>'+html.escape(t['landmark_name'])+'</h1><p>原封存E4轨迹，只读可视化；青色为公开Polygon，非人工像素Mask。各热图独立色阶，零起点。</p><select id="step">'+''.join(f'<option value="{k}">Step {k}</option>' for k in range(len(names)))+'</select><a id="raw">原生RGB</a><img id="frame"><script>const files='+json.dumps(names)+';const s=document.getElementById("step");function show(){const k=Number(s.value);document.getElementById("frame").src=files[k];document.getElementById("raw").href="step_"+String(k).padStart(2,"0")+"_rgb.png"}s.onchange=show;show()</script>'
  (folder/'index.html').write_text(page)
if __name__=='__main__':main()
