import json
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 b=p.chromium.launch(args=['--no-sandbox']);page=b.new_page(viewport={'width':1500,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto('http://127.0.0.1:8931/index.html');page.wait_for_function("document.querySelector('#stats table')!==null")
 assert page.evaluate('rows.length')==7619
 page.fill('#idx','7');page.locator('#idx').dispatch_event('change');assert 'One Stop' in page.locator('#original').inner_text();assert 'drop_one_from_full' in page.locator('#word').inner_text();page.select_option('#graph','full');page.click('#next');assert page.locator('#idx').input_value()=='8';page.click('#prev');assert page.locator('#idx').input_value()=='7';page.click('#lost');assert page.evaluate("rows[+document.querySelector('#idx').value].full_hit1&&!rows[+document.querySelector('#idx').value].mini_hit1")
 page.screenshot(path='reports/structured_text_gap/browser_smoke.png');b.close();assert not errors,errors
 Path('reports/structured_text_gap/browser_test.json').write_text(json.dumps(dict(status='passed',queries=7619,task_selection=True,graph_switch=True,next_previous=True,loss_filter=True,word_test_display=True,javascript_errors=errors),indent=2))
