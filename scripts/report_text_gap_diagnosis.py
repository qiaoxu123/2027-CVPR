"""Per-query causal-path and token-block diagnosis of frozen text intervention."""
import json,argparse,csv,html,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KS=[1,5,10,20,50]
def measures(logits,dist):
 ids=np.argsort(-logits,axis=-1);err=np.take_along_axis(dist,ids,axis=-1);prob=np.exp(logits-logits.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
 hit={k:err[:,:k].min(-1)<=20 for k in KS};ranks=(err<=20).argmax(-1)+1
 return dict(hit=hit,ne=err[:,0],rank=ranks,mass=(prob*(dist<=20)).sum(-1))
def stats(d):return {**{f'R{k}':float(d['hit'][k].mean()) for k in KS},'NE':float(d['ne'].mean()),'mass20':float(d['mass'].mean())}
def pair(a,b,subset=None):
 sel=np.ones(len(a),bool) if subset is None else subset
 return dict(N=int(sel.sum()),fix=int((sel&~a&b).sum()),damage=int((sel&a&~b).sum()),net=int((sel&b).sum()-(sel&a).sum()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.previous);run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
 old=json.load(open(src/'summary.json'));audit=json.load(open(src/'parsing.json'));texts=json.load(open(run/'restore_v2/drop_corrected_arrays/text_variants.json'));n=len(audit);assert n==7619
 records=[json.loads(x) for x in open(src/'full.jsonl')];origins=np.array([r['world_bounds'][:2] for r in records]);full=np.load(src/'full_predictions.npz');mini=np.load(src/'target_type_relation_predictions.npz');grid=full['grid'];gt=full['gt'];dist=np.linalg.norm(grid-gt[:,None],axis=-1)
 all_logits={'full':full['logits'],'mini':mini['logits']}
 for f in (run/'factors').glob('*.npz'):all_logits[f.stem]=np.load(f)['logits']
 for f in (run/'restore_v2/checked_arrays').glob('*.npy'):all_logits[f.stem]=np.load(f)
 for f in (run/'restore_v2/drop_corrected_arrays').glob('*.npy'):all_logits[f.stem]=np.load(f)
 assert len(all_logits)==22,len(all_logits)
 pred={k:measures(v,dist) for k,v in all_logits.items()};summary={};rows=[];damaged=pred['full']['hit'][1]&~pred['mini']['hit'][1]
 for arm,d in pred.items():
  if '@' in arm:
   name,g=arm.split('@');base=('full' if g=='full' else 'full_text_mini_graph') if name.startswith('full_drop') else ('mini_text_full_graph' if g=='full' else 'mini')
  else:base='mini'
  summary[arm]=dict(**stats(d),paired_vs_reference=pair(pred[base]['hit'][1],d['hit'][1]),reference=base,recovered_original_loss=pair(pred['mini']['hit'][1],d['hit'][1],damaged))
  if '@' in arm:
   block=name.removeprefix('restore_').removeprefix('full_drop_');active=np.array([bool(x['tags'].get(block if block!='all' else 'attributes',[])) if block!='all' else bool(x['tags']['attributes'] or x['tags']['quantities'] or x['tags']['chain']) for x in texts]);
   if name=='reorder_minimal':active=np.array([x['variants'][name]!=x['minimal'] for x in texts])
   summary[arm]['active_queries']=int(active.sum())
 for i,r in enumerate(audit):
  row=dict(index=i,episode=r['episode'],map=r['map'],original=r['original'],minimal=r['minimal'],attributes=' '.join(t['text'] for t in texts[i]['tags']['attributes']),quantities=' '.join(t['text'] for t in texts[i]['tags']['quantities']),chain=' '.join(t['text'] for t in texts[i]['tags']['chain']),indirect_proxy=texts[i]['tags']['indirect_proxy'],direct_proxy=texts[i]['tags']['direct_proxy'])
  for arm,d in pred.items():row.update({arm+'_hit1':bool(d['hit'][1][i]),arm+'_hit20':bool(d['hit'][20][i]),arm+'_rank':int(d['rank'][i]),arm+'_NE':float(d['ne'][i])})
  rows.append(row)
 with open(out/'per_query.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'per_query.json').write_text(json.dumps(rows,ensure_ascii=False,separators=(',',':')))
 (out/'text_variants.json').write_text(json.dumps(texts,ensure_ascii=False,separators=(',',':')))
 summary_meta=dict(metrics=summary,N=n,full_correct_mini_wrong=int(damaged.sum()),of_original_loss_mini_top20_kept=int((damaged&pred['mini']['hit'][20]).sum()),of_original_loss_mini_top20_lost=int((damaged&~pred['mini']['hit'][20]).sum()),word_blocks={key:sum(bool(r['tags'][key]) for r in texts) for key in ['attributes','quantities','chain']})
 # Descriptive two-path Shapley accounting, not a semantic-generalization claim.
 v00=summary['mini']['R1'];v10=summary['full_text_mini_graph']['R1'];v01=summary['mini_text_full_graph']['R1'];v11=summary['full']['R1'];summary_meta['two_path_gap_accounting']=dict(BERT_text=.5*((v10-v00)+(v11-v01)),graph_rules=.5*((v01-v00)+(v11-v10)))
 (out/'summary.json').write_text(json.dumps(summary_meta,indent=2))
 shutil.copyfile(run/'words/word_tests.json',out/'word_tests.json');words=json.load(open(run/'words/word_tests.json'))
 wordstat={}
 for mode,key in [('drop_one_from_full','full_base'),('add_one_to_mini','mini_base')]:
  seq=[(r,e) for r in words for e in r['experiments'] if e['mode']==mode];wordstat[mode]=dict(tests=len(seq),R1_flips=sum(e['R1']!=r[key]['R1'] for r,e in seq),NE_changes_over20m=sum(abs(e['NE']-r[key]['NE'])>20 for r,e in seq))
 (out/'word_summary.json').write_text(json.dumps(wordstat,indent=2))
 fig,ax=plt.subplots(figsize=(10,5));arms=['mini','restore_attributes@mini','restore_quantities@mini','restore_chain@mini','restore_all@mini','reorder_minimal@mini','full'];ax.barh(arms,[summary[k]['R1']*100 for k in arms]);ax.set_xlabel('TRAIN Top1 position hit within 20m (%)');fig.tight_layout();fig.savefig(out/'restoration_r1.png',dpi=180);plt.close(fig)
 selected=[]
 # Include per-word failures and both-positive/counterexample cases.
 for w in words:
  if len(selected)<16:selected.append(w['index'])
 figures={}
 arms6=['full','mini','restore_attributes@mini','restore_quantities@mini','restore_chain@mini','restore_all@mini']
 for i in selected:
  fig,axs=plt.subplots(2,3,figsize=(12,8));world=grid[i]+origins[i];target=gt[i]+origins[i]
  maxp=max(np.exp(all_logits[k][i]-np.max(all_logits[k][i])).max()/np.exp(all_logits[k][i]-np.max(all_logits[k][i])).sum() for k in arms6)
  for ax,arm in zip(axs.flat,arms6):
   l=all_logits[arm][i];pr=np.exp(l-l.max());pr/=pr.sum();ax.scatter(world[:,0],world[:,1],c=pr,s=8,cmap='magma',vmin=0,vmax=maxp);top=np.argsort(-l)[:20];ax.scatter(world[top,0],world[top,1],s=12,facecolors='none',edgecolors='cyan');ax.scatter(*target,marker='*',c='lime',s=90)
   bounds=np.asarray(records[i]['world_bounds']);span=bounds[2:]-bounds[:2]
   for land in records[i]['landmarks']:
    q=np.asarray(land['contour'])*span+bounds[:2];ax.plot(*np.vstack([q,q[0]]).T,c='gray',lw=.6)
   ax.set_title(f"{arm}\nNE {pred[arm]['ne'][i]:.1f}m; rank {pred[arm]['rank'][i]}",fontsize=9);ax.set_aspect('equal')
  fig.suptitle(f'TRAIN {i}: polygons gray, GT green, Top20 cyan',fontsize=10);fig.tight_layout();name=f'case_{i}.png';fig.savefig(out/'cases'/name,dpi=130);plt.close(fig);figures[str(i)]='cases/'+name
 (out/'figures.json').write_text(json.dumps(figures,indent=2))
 # Interactive all-query review; static heatmaps for selected real cases.
 page=r'''<!doctype html><meta charset="utf-8"><title>原文与精简文本逐条诊断</title><style>body{font:16px sans-serif;margin:20px;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}pre{white-space:pre-wrap;background:#eee;padding:12px}input,button,select{padding:8px;margin:5px}.good{color:green}.bad{color:#b00}</style><h1>7619条TRAIN：原文与精简文本差距</h1><p>训练内拟合诊断；全部GT仅事后评价。切换固定图结构可比较BERT文本修复。完整词测试24条/1145次是事后典型案例，不是随机总体。</p><input id="idx" type="number" min="0" max="7618" value="0"><button id="prev">上一条</button><button id="next">下一条</button><button id="lost">下一条原文成功/精简失败</button><select id="graph"><option>mini</option><option>full</option></select><h2 id="title"></h2><b>原文</b><pre id="original"></pre><b>精简输入</b><pre id="minimal"></pre><b>被单独测试的信息块</b><pre id="blocks"></pre><div id="stats"></div><h3>本条逐词实验（若有）</h3><pre id="word"></pre><img id="fig" style="max-width:100%"><script>let rows,word,figures;const $=x=>document.getElementById(x);function show(){let i=Math.max(0,Math.min(7618,+$('idx').value));$('idx').value=i;let r=rows[i];$('title').textContent=i+' | '+r.episode;$('original').textContent=r.original;$('minimal').textContent=r.minimal;$('blocks').textContent='外观：'+r.attributes+String.fromCharCode(10)+'数量/序数/距离：'+r.quantities+String.fromCharCode(10)+'关系路径：'+r.chain;let g=$('graph').value;let arms=['full','mini','restore_attributes@'+g,'restore_quantities@'+g,'restore_chain@'+g,'restore_all@'+g,'reorder_minimal@'+g,'full_drop_attributes@'+g,'full_drop_quantities@'+g,'full_drop_chain@'+g];let table='<table><tr><th>输入干预</th><th>Top1</th><th>Top20</th><th>GT附近最好排名</th><th>NE</th></tr>';for(let a of arms)table+='<tr><td>'+a+'</td><td>'+r[a+'_hit1']+'</td><td>'+r[a+'_hit20']+'</td><td>'+r[a+'_rank']+'</td><td>'+r[a+'_NE'].toFixed(2)+'</td></tr>';$('stats').innerHTML=table+'</table>';let w=word.find(x=>x.index===i);$('word').textContent=w?JSON.stringify(w,null,1):'本条未做逐词实验；已有8组块级测试。';$('fig').style.display=figures[i]?'block':'none';if(figures[i])$('fig').src=figures[i];}async function load(){[rows,word,figures]=await Promise.all(['per_query.json','word_tests.json','figures.json'].map(x=>fetch(x).then(r=>r.json())));show()}$('idx').onchange=show;$('graph').onchange=show;$('prev').onclick=()=>{$('idx').value=Math.max(0,+$('idx').value-1);show()};$('next').onclick=()=>{$('idx').value=Math.min(7618,+$('idx').value+1);show()};$('lost').onclick=()=>{let start=+$('idx').value;for(let n=1;n<=rows.length;n++){let i=(start+n)%rows.length;if(rows[i].full_hit1&&!rows[i].mini_hit1){$('idx').value=i;show();break}}};load();</script>'''
 (out/'index.html').write_text(page)
 print(json.dumps(summary_meta,indent=2))
if __name__=='__main__':main()
