"""Package trained direction kernels and sealed real/synthetic evaluation artifacts."""
import argparse,html,json,shutil,textwrap
from pathlib import Path
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.direction_belief import DirectionBelief,WORDS
from graphnav.direction_belief_data import UNIT,point_distribution
from graphnav.checkpointing import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run;o=a.output;o.mkdir(parents=True,exist_ok=True)
 for folder in ['data_split','evaluation','pgvb_current']:
  target=o/folder;target.mkdir(exist_ok=True)
  for f in (r/folder).iterdir():
   if f.suffix in ['.json','.npz']:shutil.copyfile(f,target/f.name)
 for f in (r/'pilot_s0').glob('*.json'):shutil.copyfile(f,o/f.name)
 logs=o/'logs';logs.mkdir(exist_ok=True)
 for f in (r/'execution').iterdir():
  if f.suffix in ['.log','.sh','.exit']:shutil.copyfile(f,logs/f.name)
 metrics=json.loads((o/'evaluation/metrics.json').read_text());pub=json.loads((o/'evaluation/public_dev.json').read_text());labels={x['index']:x for x in json.loads((o/'data_split/private_dev_labels.json').read_text())};probs=np.load(o/'evaluation/real_probabilities.npz');models={};params={}
 for name in ['direction_synthetic','direction_real','nodirection_real']:
  payload=torch.load(r/'pilot_s0'/f'{name}.pt',map_location='cpu',weights_only=True);model=DirectionBelief(payload['ignore_direction']).eval().requires_grad_(False);model.load_state_dict(payload['model']);models[name]=model;weights=payload['model']['parameters_by_word.weight'];params[name]=dict(checkpoint_sha256=file_sha256(r/'pilot_s0'/f'{name}.pt'),raw_parameters=weights.tolist(),mean_offset_m=(100*weights[:,:2]).tolist(),std_m=(30*weights[:,2:].exp()).tolist())
 (o/'learned_parameters.json').write_text(json.dumps(params,indent=2))
 fig,axes=plt.subplots(2,5,figsize=(16,6));span=torch.tensor([[400.,400.]]);center=torch.tensor([[200.,200.]]);anchor=point_distribution(center,span,sigma=6.);counter={}
 for j,name in enumerate(['direction_synthetic','direction_real']):
  axes[j,0].imshow(anchor[0],origin='lower',extent=[0,400,0,400]);axes[j,0].set_title(name+' Anchor',fontsize=9)
  for k in range(4):
   with torch.no_grad():pr=models[name](anchor,torch.tensor([k]),span)[0].numpy()
   counter[name+'_'+WORDS[k]]=pr;ax=axes[j,k+1];ax.imshow(pr,origin='lower',extent=[0,400,0,400],vmin=0,vmax=float(pr.max()));ax.scatter(*(np.array([200,200])+60*UNIT[k]),c='red',marker='*',s=60);ax.set_title(WORDS[k]+' | synthetic reference')
 fig.suptitle('Same synthetic Anchor, change only word | red stars are synthetic 60m reference points');fig.tight_layout();fig.savefig(o/'four_world_directions.png',dpi=130);plt.close(fig)
 fig,axes=plt.subplots(2,5,figsize=(16,6))
 for row,angle in enumerate([0.,np.pi/2]):
  axes[row,0].imshow(anchor[0],origin='lower',extent=[0,400,0,400]);axes[row,0].set_title(f'Anchor, known heading {angle*180/np.pi:.0f}deg')
  for col,k in enumerate(range(4,8),1):
   with torch.no_grad():pr=models['direction_synthetic'](anchor,torch.tensor([k]),span,reference_heading=torch.tensor([angle]))[0].numpy()
   counter[f'{WORDS[k]}_theta{row}']=pr;ax=axes[row,col];ax.imshow(pr,origin='lower',extent=[0,400,0,400]);u=UNIT[k];rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);ax.scatter(*(np.array([200,200])+60*rot@u),c='red',marker='*');ax.set_title(WORDS[k])
 fig.suptitle('Relative directions only with an explicit synthetic reference heading; not real CityNav frame labels');fig.tight_layout();fig.savefig(o/'relative_known_heading.png',dpi=130);plt.close(fig);np.savez_compressed(o/'counterfactual_probabilities.npz',**counter)
 chosen=[]
 for word in ['north','south','west']:
  pos=next((j for j,p in enumerate(pub) if p['word']==word),None)
  if pos is not None and pos not in chosen:chosen.append(pos)
 delta=np.asarray(metrics['real']['direction_real']['all']['per_sample_NE'])-np.asarray(metrics['real']['nodirection_real']['all']['per_sample_NE'])
 for j in [int(delta.argmin()),int(delta.argmax())]:
  if j not in chosen:chosen.append(j)
 cases=[]
 for j in chosen:
  p=pub[j];target=np.asarray(labels[p['index']]['target_world'][:2]);b=p['bounds'];extent=[b[0],b[2],b[1],b[3]];fig,axes=plt.subplots(1,4,figsize=(15,4))
  for ax,name in zip(axes,['anchor','nodirection_real','direction_synthetic','direction_real']):
   values=probs[name][j];ax.imshow(values,origin='lower',extent=extent,vmin=0,vmax=max(float(values.max()),1e-8),cmap='magma')
   for poly in p['public_polygons']:
    xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],'c-',lw=.7)
   y,x=np.unravel_index(values.argmax(),(28,28));pred=np.array(b[:2])+(np.array([x,y])+.5)/28*np.array(p['span']);ax.scatter(*target,c='lime',marker='*',s=65);ax.scatter(*pred,c='red',s=20);ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3]);ax.set_title(f'{name}: NE={np.linalg.norm(pred-target):.1f}m',fontsize=9)
  fig.suptitle('\n'.join(textwrap.wrap(f'Episode {p["episode"]} | {p["word"]} of {p["landmark_name"]} | green = real TRAIN GT after scoring; cyan = public Anchor',110)),fontsize=9);fig.tight_layout();file=f'case_{p["episode"]}.png';fig.savefig(o/file,dpi=140);plt.close(fig);cases.append(dict(episode=p['episode'],index=p['index'],word=p['word'],instruction=p['instruction'],file=file))
 (o/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
 history=json.loads((o/'history.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
 for name in ['direction_synthetic','direction_real','nodirection_synthetic','nodirection_real']:
  data=[x for x in history if x['model']==name];axes[0].plot([x['step'] for x in data],[x['loss'] for x in data],label=name)
 axes[0].legend(fontsize=7);axes[0].set_title('TRAIN losses (no Dev model selection)')
 names=['nodirection_real','direction_synthetic','direction_real'];axes[1].bar(names,[metrics['real'][m]['all']['R20'] for m in names]);axes[1].set_title('Real TRAIN Dev R20, n=34');axes[1].tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(o/'summary.png',dpi=150);plt.close(fig)
 page='<meta charset="utf-8"><title>方向Belief最小实验</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}img{max-width:100%}</style><h1>32参数方向Belief实验</h1><p>合成机制验证与真实TRAIN评估分开。相对方向需要已知参考朝向；未验证真实地理北。</p><a href="REPORT_zh.md">中文报告</a><img src="summary.png"><h2>合成可控对照</h2><img src="four_world_directions.png"><img src="relative_known_heading.png"><h2>真实TRAIN目标</h2>'
 for c in cases:page+=f'<h3>{c["episode"]}: {c["word"]}</h3><p>{html.escape(c["instruction"])}</p><a href="{c["file"]}"><img src="{c["file"]}"></a>'
 (o/'index.html').write_text(page)
if __name__=='__main__':main()
