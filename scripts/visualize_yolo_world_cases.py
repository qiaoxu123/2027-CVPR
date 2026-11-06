"""Representative paired cases selected by recorded outcomes, including failures."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);(root/'cases').mkdir(exist_ok=True)
 rows=[json.load(open(f)) for f in (Path(a.run)/'results').glob('*_base.json')]
 def vehicles(r,m):return len([d for d in r['methods'][m]['detections'] if d['label'] in ['car','van','truck']])
 selected=[]
 ordered=sorted(rows,key=lambda r:vehicles(r,'tile')-vehicles(r,'whole'),reverse=True)
 selected+=ordered[:4]
 zeros=[r for r in rows if vehicles(r,'tile')==0 and sum(g['eligible'] and g['category']=='vehicle' for g in r['proxies'])>=5]
 selected+=sorted(zeros,key=lambda r:-sum(g['eligible'] and g['category']=='vehicle' for g in r['proxies']))[:3]
 selected+=sorted(rows,key=lambda r:-sum(g['category']=='parking' and g['eligible'] for g in r['proxies']))[:3]
 selected+=sorted(rows,key=lambda r:-vehicles(r,'whole'))[:2]
 unique={r['sample_id']:r for r in selected};manifest=[]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 for r in unique.values():
  original=Image.open(r['image_path']).convert('RGB');panels=[]
  for m in ['whole','tile']:
   im=original.copy();d=ImageDraw.Draw(im);w=im.width
   for g in r['proxies']:
    if g['category'] not in ['vehicle','parking']:continue
    color='#00ffff' if g['eligible'] else '#aaaaaa'
    for q in g['contours']:d.line([tuple(p) for p in q]+[tuple(q[0])],fill=color,width=max(2,w//700))
   for det in r['methods'][m]['detections']:
    d.rectangle(det['box'],outline='#ff9900',width=max(2,w//700))
   im.thumbnail((800,800));canvas=Image.new('RGB',(800,850),'#171b22');canvas.paste(im,(0,50));draw=ImageDraw.Draw(canvas);draw.text((8,5),f"{r['sample_id']} | {m} | vehicles={vehicles(r,m)}",font=font,fill='white');draw.text((8,27),'Cyan=public proxy contour; Orange=YOLO detection',font=font,fill='white');panels.append(canvas)
  pair=Image.new('RGB',(1600,850));pair.paste(panels[0],(0,0));pair.paste(panels[1],(800,0));name=r['sample_id'].replace(':','_');pair.save(root/'cases'/f'{name}.png');manifest.append(dict(sample_id=r['sample_id'],map=r['map_name'],whole_vehicles=vehicles(r,'whole'),tile_vehicles=vehicles(r,'tile'),figure=f'cases/{name}.png'))
 (root/'cases.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
