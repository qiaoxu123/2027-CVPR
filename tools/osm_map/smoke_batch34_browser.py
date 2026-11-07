"""Read-only public browser batch audit; never accepts or edits human controls."""
import os,json,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
base=os.environ.get('CONTROL_TEST_URL','http://101.43.173.9:42024');root=Path(os.environ.get('CONTROL_STORE','~/.local/share/citynav-osm-controls/data')).expanduser();out=Path('reports/citynav_osm_batch34');before={c:hashlib.sha256((root/(c+'.json')).read_bytes()).hexdigest() for c in ['birmingham','cambridge']}
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox']);ctx=browser.new_context(http_credentials={'username':os.environ['G5_REVIEW_USER'],'password':os.environ['G5_REVIEW_PASSWORD']},viewport={'width':1800,'height':1150});page=ctx.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(base+'/control');page.wait_for_function("document.querySelectorAll('#batchRows tr').length===34",timeout=30000);assert '待验证 21' in page.locator('#batchCounts').inner_text()
 for scene in ['cambridge_block_33','birmingham_block_3']:
  page.get_by_role('button',name=scene,exact=True).click();page.wait_for_function('(n)=>document.querySelector("#scene").value===n',arg=scene);page.wait_for_function('(n)=>document.querySelector("#autoImage").src.includes(n)&&document.querySelector("#autoImage").naturalWidth>0',arg=scene,timeout=30000)
  if not page.locator('#showGhost').is_checked():page.locator('#showGhost').check()
  page.wait_for_function('ghost && ghost.img && ghost.img.naturalWidth>0',timeout=30000);page.screenshot(path=str(out/(scene+'_public.png')),full_page=True)
 assert not errors,errors
 api=ctx.request.get(base+'/api/registration-status').json();assert api['verified']==1 and api['precise_2m']==0
 assert ctx.request.get(base+'/api/osm-export/birmingham_block_3').status==200
 assert ctx.request.get(base+'/api/osm-export/cambridge_block_3').status==403
 browser.close()
after={c:hashlib.sha256((root/(c+'.json')).read_bytes()).hexdigest() for c in before};assert before==after
(out/'public_browser.json').write_text(json.dumps(dict(url=base+'/control',scenes=34,counts={'verified':1,'pending':21,'unreliable':12,'precise_2m':0},js_errors=errors,scene_switch_and_automatic_overlay=True,verified_export_status=200,unverified_export_status=403,manual_controls_unchanged=True),indent=2));print('PUBLIC BROWSER PASS, 34 rows, overlays and export guards verified')
