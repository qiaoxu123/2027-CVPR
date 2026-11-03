"""Post-hoc TRAIN visualizations: GT world-point projections are NOT pixel boxes."""
import argparse
import html
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from graphnav.landmark_projection import world_to_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','legacy-predictions','small-predictions','roles','output'):
        p.add_argument('--'+key, required=True, type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('preserve previous gallery')
    a.output.mkdir(parents=True)
    def lines(name):
        return [json.loads(x) for x in (a.observations/name).read_text().splitlines()]
    frames=lines('public_frames.jsonl')
    labels={r['frame_key']:r for r in lines('labels_private.jsonl') if r['status']=='captured'}
    legacy={r['frame_key']:r for r in json.loads(a.legacy_predictions.read_text())['predictions']}
    small={r['frame_key']:r for r in json.loads(a.small_predictions.read_text())['predictions']}
    roles=json.loads(a.roles.read_text())
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    cards=[]
    selected=[]
    for task in dict.fromkeys(f['task_key'] for f in frames):
        task_frames=[f for f in frames if f['task_key']==task]
        # Display the last captured reference state, not a GT-selected success frame.
        selected.append(task_frames[-1])
    for frame in selected:
        key=frame['frame_key']; gt=np.array(labels[key]['target_world'][:2])
        image=Image.open(a.observations/frame['image']).convert('RGB')
        canvas=Image.new('RGB',(1152,490),'white'); draw=ImageDraw.Draw(canvas)
        variants=[('Legacy ROI32/64/128',legacy[key],'original_text'),
                  ('Local ROI8/16/32',small[key],'original_text'),
                  ('Local ROI + target text',small[key],'target_text')]
        for panel,(title,pred,method) in enumerate(variants):
            canvas.paste(image,(384*panel,0))
            points=[(gt,'lime','GT-world proxy')]
            top=pred['methods'][method]['top_world']
            if top: points.append((np.array(top[0]),'red','predicted local cell'))
            for point,color,label in points:
                xy=world_to_frame(np.array([point]),frame['footprint_world'],(384,384,3))[0]
                if ((xy>=0)&(xy<=383)).all():
                    x,y=float(xy[0]+384*panel),float(xy[1]);draw.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3)
                    draw.text((x+7,max(0,y-12)),label,fill=color,font=font,stroke_width=1,stroke_fill='black')
            error=float(np.linalg.norm(np.array(top[0])-gt)) if top else None
            lines_out=[title, f"local spatial NE={error:.1f}m" if error is not None else 'No supported ROI',
                       f"supported cells={len(pred['supported_world'])}", 'Green is projected world point,', 'NOT verified target identity/BBox.']
            for line,text in enumerate(lines_out):draw.text((384*panel+6,390+18*line),text,fill='black',font=font)
        path=a.output/f'{key}.png';canvas.save(path)
        relative=roles[frame['task_key']]
        cards.append(f'<article><h2>{html.escape(key)} / {html.escape(labels[key]["category"])}</h2><p>{html.escape(frame["instruction"])}</p><a href="{path.name}"><img width="1152" src="{path.name}"></a><p>Target: {html.escape(relative["target_text"])}<br>References: {html.escape(relative["reference_text"])}</p></article>')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed-yaw TRAIN mechanism cases</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{max-width:100%}article{border-bottom:1px solid #ccc;padding:20px 0}</style><h1>24个TRAIN实例：固定yaw局部匹配诊断</h1><p>每个任务展示最后一个已采集参考轨迹状态，不按成功结果挑图。绿色仅为GT世界点投影，并非人工确认实例框；真实可见性未知。三列RGB完全相同，仅ROI/文本查询不同。没有Goal Belief更新或闭环SR结论。</p>'+''.join(cards))


if __name__=='__main__':
    main()
