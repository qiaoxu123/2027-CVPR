"""Qualitative side-by-side actual parser graphs. Agent comments are NOT gold."""
import json,argparse,html,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOTES={13057:'原文目标为white car；A保留完整名词短语，B仅car会丢外观。C原始输出把Anatomy School当target。属于Agent语言观察，不是人工标准。',7434:'目标短语包含light grey car；只输出car会丢颜色。此处只检查目标角色，不评序数条件。',6822:'A未输出目标，B输出white car；不据此判断其他关系链正确或方向一定可修复。',10905:'B恢复了带颜色/屋顶的目标短语；只输出building不足以保留实例特征。',14114:'原文是以Grey roof building开头的片段；B恢复目标，A/C遗漏。',10469:'原文同时列car和parking lot，目标作用域存在歧义；不能仅因模型输出car就判为正确。',10099:'主目标为building；Portugal Street是前置条件，不能当导航目标。',12798:'L-shaped属于重要目标特征；仅building的跨度会丢形状信息。'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args();o=a.report;inputs={r['index']:r for r in json.load(open(o/'public_inputs.json'))};private={r['index']:r for r in json.load(open(o/'sampling_outcomes_POSTHOC.json'))};ps={tag:{x['index']:x for x in map(json.loads,open(o/tag/'predictions.jsonl'))} for tag in 'ABCD'};ids=json.load(open(o/'pre_D_case_indices.json'))['indices'];ids+=sorted(i for i,x in private.items() if x['group']=='random_TRAIN_uniform')[:4];cards=[];cases=[]
 for num,i in enumerate(ids):
  fig,axes=plt.subplots(1,4,figsize=(20,5));text=inputs[i]['text']
  for ax,tag in zip(axes,'ABCD'):
   g=ps[tag][i]['graph'];nodes=g['entities'];by={};row={0:0,1:0,2:0}
   for e in nodes:
    col=0 if e['role']=='target' else 2 if e['role']=='landmark' else 1;by[e['id']]=(col,row[col],e);row[col]+=1
   maxrow=max(3,max(row.values()));ax.set_xlim(-.3,2.8);ax.set_ylim(maxrow+.5,-.5);ax.axis('off');ax.set_title(tag+': '+str(len(g['relations']))+' relations')
   for edge in g['relations']:
    if edge['subject'] not in by or edge['object'] not in by:continue
    x,y,_=by[edge['subject']];xx,yy,_=by[edge['object']];ax.annotate('',xy=(xx+.2,yy),xytext=(x+.2,y),arrowprops=dict(arrowstyle='->',color='#999999',lw=.6,connectionstyle='arc3,rad=0.07'));ax.text((x+xx)/2+.2,(y+yy)/2-.08,edge['relation'],fontsize=4.5,color='#556')
   for x,y,e in by.values():
    value='\n'.join(textwrap.wrap(e['text'],18));ax.text(x+.2,y,value,ha='center',va='center',fontsize=5.5,bbox=dict(boxstyle='round,pad=.25',facecolor='#b5e7ca' if e['role']=='target' else '#ffe1a8' if e['role']=='landmark' else '#c5dcff',edgecolor='#aab'))
  fig.suptitle('Query '+str(i)+' — '+private[i]['group']+' (output graphs, not correctness labels)',fontsize=10);fig.tight_layout();fn=f'comparison_{num:02d}.png';fig.savefig(o/fn,dpi=150);plt.close(fig)
  sub=[]
  for tag in 'ABCD':
   g=ps[tag][i]['graph'];lookup={e['id']:e['text'] for e in g['entities']};triples=[(lookup.get(e['subject']),e['relation'],lookup.get(e['object'])) for e in g['relations']];sub.append(f'<td><h3>{tag}</h3><p>Target: {html.escape(str([e["text"] for e in g["entities"] if e["role"]=="target"))}</p><pre>{html.escape(json.dumps(triples,ensure_ascii=False,indent=2))}</pre></td>')
  note=NOTES.get(i,'随机TRAIN案例，不根据模型是否正确筛选。');cards.append(f'<article><h2>Query{i}</h2><p>{html.escape(text)}</p><p>{html.escape(note)}</p><a href="{fn}"><img width="1200" src="{fn}"></a><table><tr>'+''.join(sub)+'</tr></table></article>');cases.append(dict(index=i,instruction=text,Agent_language_comment_not_gold=note,group=private[i]['group']))
 (o/'qualitative_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2));(o/'cases.html').write_text('<meta charset="utf-8"><style>body{max-width:1400px;margin:25px auto;font:16px sans-serif}td{width:25%;vertical-align:top;border:1px solid #ccc}pre{white-space:pre-wrap}article{margin:25px 0;border-bottom:2px solid #aaa}</style><h1>Actual parser outputs: qualitative comparison</h1><p>These observations are not independent human gold. Examples selected before D inference; no coordinates or GT-assisted parsing.</p>'+''.join(cards))
if __name__=='__main__':main()
