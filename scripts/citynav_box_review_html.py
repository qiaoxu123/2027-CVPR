"""Dependency-free local box review UI; explicit human attestation, no auto-approval."""
import json


def write_review_html(records,path):
    payload=json.dumps(records).replace('</','<\\/')
    page=r'''<!doctype html><meta charset="utf-8"><title>TRAIN automatic box review — NOT pixel GT</title>
<style>body{font:15px sans-serif;background:#f5f5f5;margin:20px}main{display:flex;gap:20px}canvas{width:768px;max-width:60vw;background:#222;touch-action:none;image-rendering:pixelated}aside{max-width:600px}button,input,select{margin:4px;padding:6px}#warning{color:#a00;font-weight:bold}pre{white-space:pre-wrap;font-size:12px}</style>
<h2>TRAIN 自动框核验（未核验框不是像素GT）</h2><p id="warning">本页面没有自动批准。确认/修正需要真人姓名、可见性判断和真人核验声明。拖动图像可重画框；原图坐标不变。</p>
<button onclick="move(-1)">上一张</button><button onclick="move(1)">下一张</button><span id="position"></span>
<label>候选框数<select id="number" onchange="paint()"><option>15</option><option>30</option><option>60</option><option>900</option><option>0</option></select></label>
<label><input type="checkbox" id="poly" checked onchange="paint()">投影/参照多边形</label>
<main><canvas id="image" width="384" height="384"></canvas><aside><p id="instruction"></p>
<label>核验人<input id="reviewer" placeholder="真人姓名/标识"></label><label><input type="checkbox" id="attest">我是真人且已检查当前RGB</label>
<p>真实可见性<select id="visibility"><option value="unknown">无法判断</option><option value="visible">可见</option><option value="partially_visible">部分可见/遮挡</option><option value="not_visible">不可见</option></select></p>
<p>像素框 [x0,y0,x1,y1) <input id="bbox" size="34" onchange="setBox()"></p>
<button onclick="save('confirmed')">确认自动框</button><button onclick="save('corrected')">保存修正框</button><button onclick="save('rejected')">拒绝</button><button onclick="save('not_visible')">标记不可见</button>
<p>备注 <input id="notes" size="50"></p><p id="status"></p><pre id="metadata"></pre></aside></main>
<p>红色=自动提案；紫色=当前编辑框；绿色=目标轮廓投影；黄色=参照；蓝色=原生AerialVG候选及原生排序。完整900框可切换查看。</p>
<button onclick="exportFile()">导出核验JSON</button><input type="file" id="import" accept="application/json"><p>编辑草稿保存在本浏览器localStorage；跨设备请导出JSON。没有人工操作的样本永远保持unreviewed。</p>
<script>const rows=__DATA__;let i=0,box=null,drag=null,img=new Image();const canvas=document.getElementById('image'),ctx=canvas.getContext('2d');let edits={};
function key(){return 'citynav-human-box-v1:'+rows[i].sample_id+':'+rows[i].image_sha}
function load(){const r=rows[i];let stored=edits[r.sample_id]||JSON.parse(localStorage.getItem(key())||'null');box=(stored&&stored.bbox)||r.auto_bbox_xyxy;
 document.getElementById('position').textContent=`${i+1}/${rows.length} ${r.sample_id}`;document.getElementById('instruction').textContent=r.instruction;
 document.getElementById('bbox').value=JSON.stringify(box);document.getElementById('visibility').value=(stored&&stored.visibility)||'unknown';document.getElementById('notes').value=(stored&&stored.notes)||'';document.getElementById('attest').checked=false;
 document.getElementById('metadata').textContent=JSON.stringify({instance:r.instance_id,category:r.category,groups:[r.episode,r.instruction_group,r.instance_group],projection:r.projection,quality:r.quality,relations:r.relations},null,2);
 document.getElementById('status').textContent=(stored&&stored.status)||'UNREVIEWED';img.onload=paint;img.src=r.image;}
function move(n){i=(i+n+rows.length)%rows.length;load()}
function stroke(b,color,width=1){if(!b)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}
function polygon(points,color){if(!points||!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();points.forEach((p,j)=>j?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.stroke()}
function paint(){const r=rows[i];ctx.clearRect(0,0,384,384);ctx.drawImage(img,0,0,384,384);
 if(document.getElementById('poly').checked){polygon(r.projection.contour_px,'#00ff00');r.references.forEach(e=>polygon(e.projection.contour_px,'#ffff00'))}
 const n=Number(document.getElementById('number').value);r.raw_order.slice(0,n).forEach((q,j)=>{const [cx,cy,w,h]=r.raw_boxes[q];stroke([(cx-w/2)*384,(cy-h/2)*384,(cx+w/2)*384,(cy+h/2)*384],'#0099ff',.6);ctx.fillStyle='#00ffff';ctx.font='7px sans-serif';ctx.fillText(`${j+1}:${q}`,cx*384,cy*384)});stroke(r.auto_bbox_xyxy,'red',1.5);stroke(box,'#ff00ff',1.5)}
function setBox(){try{box=JSON.parse(document.getElementById('bbox').value);paint()}catch(e){alert('框必须为JSON数组')}}
function point(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(384,(e.clientX-r.left)*384/r.width)),Math.max(0,Math.min(384,(e.clientY-r.top)*384/r.height))]}
canvas.onpointerdown=e=>{drag=point(e);canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;const p=point(e);box=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.max(drag[0],p[0]),Math.max(drag[1],p[1])];document.getElementById('bbox').value=JSON.stringify(box.map(x=>+x.toFixed(2)));paint()};canvas.onpointerup=()=>drag=null;
function save(status){const r=rows[i],reviewer=document.getElementById('reviewer').value.trim(),human=document.getElementById('attest').checked;if(!reviewer||!human){alert('需要真人核验人和核验声明');return}
 let visibility=document.getElementById('visibility').value;if(status==='not_visible')visibility='not_visible';if(['confirmed','corrected'].includes(status)&&!['visible','partially_visible'].includes(visibility)){alert('确认框需明确可见或部分可见；未知请拒绝/保持未核验');return}
 let chosen=status==='confirmed'?r.auto_bbox_xyxy:box;if(['confirmed','corrected'].includes(status)&&(!chosen||chosen.length!==4||chosen.some(x=>!Number.isFinite(x))||chosen[2]<=chosen[0]||chosen[3]<=chosen[1])){alert('有效像素框缺失');return}
 const entry={sample_id:r.sample_id,image_sha:r.image_sha,status,source:'human',reviewer,human_attested:true,visibility,bbox:chosen,notes:document.getElementById('notes').value,reviewed_at:new Date().toISOString()};edits[r.sample_id]=entry;localStorage.setItem(key(),JSON.stringify(entry));document.getElementById('status').textContent=status;}
function exportFile(){rows.forEach(r=>{const k='citynav-human-box-v1:'+r.sample_id+':'+r.image_sha,entry=JSON.parse(localStorage.getItem(k)||'null');if(entry)edits[r.sample_id]=entry});const data={format:'citynav-box-review-v1',reviews:Object.values(edits)};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='citynav_human_reviews.json';a.click();URL.revokeObjectURL(url)}
document.getElementById('import').onchange=async e=>{const data=JSON.parse(await e.target.files[0].text());(data.reviews||[]).forEach(r=>{const row=rows.find(x=>x.sample_id===r.sample_id);if(row&&row.image_sha===r.image_sha)edits[r.sample_id]=r});load()};load();</script>'''
    path.write_text(page.replace('__DATA__',payload))
