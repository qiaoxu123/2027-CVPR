"""Explicit browser smoke on an EMPTY control set only; synthetic clicks are removed."""
import os,json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
base=os.environ.get('CONTROL_TEST_URL','http://127.0.0.1:8793');out=Path(os.environ.get('CONTROL_TEST_OUTPUT','reports/citynav_control_points'));out.mkdir(exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox']);context=browser.new_context(http_credentials={'username':os.environ['G5_REVIEW_USER'],'password':os.environ['G5_REVIEW_PASSWORD']},viewport={'width':1700,'height':1100});page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(base+'/control');page.wait_for_function("window.L && document.querySelector('#status').textContent.includes('已恢复')",timeout=30000);page.wait_for_function("document.querySelectorAll('#left .leaflet-image-layer').length>0");page.wait_for_timeout(4000)
 initial=context.request.get(base+'/api/controls/cambridge').json();assert initial['points']==[], 'Do not overwrite existing human annotation'
 page.locator('#left').click(position={'x':300,'y':220});page.locator('#right').click(position={'x':330,'y':230});page.wait_for_function("document.querySelector('#status').textContent.includes('已保存')",timeout=30000);saved=context.request.get(base+'/api/controls/cambridge').json();assert len(saved['points'])==1
 page.reload();page.wait_for_function("document.querySelectorAll('#tbody tr').length===1",timeout=30000);assert len(context.request.get(base+'/api/controls/cambridge/export').json()['points'])==1
 page.locator('#tbody tr button').filter(has_text='删除').click();page.wait_for_function("document.querySelector('#status').textContent.includes('已保存')",timeout=30000);assert context.request.get(base+'/api/controls/cambridge').json()['points']==[]
 page.locator('#undo').click();page.wait_for_function("document.querySelectorAll('#tbody tr').length===1");page.locator('#tbody tr button').filter(has_text='删除').click();page.wait_for_function("document.querySelector('#status').textContent.includes('已保存')",timeout=30000)
 page.screenshot(path=str(out/'control_page.png'),full_page=True);assert not errors,errors
 result=dict(base_url=base,ui_click_pair=True,save=True,reload=True,export=True,delete=True,undo=True,restored_empty=True,page_errors=errors,osm_tile_responses='see app log',annotation_note='Automated synthetic click pair removed; not real geographic control points')
 (out/('browser_public.json' if '127.0.0.1' not in base else 'browser_local.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result));browser.close()
