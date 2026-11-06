"""Optional Playwright/Chromium browser verification; serve atlas over HTTP first."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox']);page=await browser.new_page(viewport={'width':1500,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(__import__('os').environ.get('CITYNAV_ATLAS_URL','http://127.0.0.1:8927/index.html'));await page.wait_for_function("document.querySelector('#c').width>=800 && document.querySelector('#meta').textContent.includes('detections')")
  count=await page.locator('#task option').count();assert count==120
  first=await page.locator('#task').input_value();await page.select_option('#bank','color');await page.select_option('#method','tile');await page.uncheck('#poly');await page.check('#poly');await page.click('#next');await page.wait_for_function("document.querySelector('#meta').textContent.startsWith(document.querySelector('#task').value.replaceAll('_',':'))")
  await page.click('#prev');await page.wait_for_function("document.querySelector('#meta').textContent.startsWith(document.querySelector('#task').value.replaceAll('_',':'))");assert await page.locator('#task').input_value()==first
  checks=await page.evaluate('''async()=>{let options=[...document.querySelector('#task').options];let all=[];for(let o of options){let r=await fetch('results/'+o.value+'.json');let d=await r.json();let im=new Image;im.src='images/'+o.value+'.jpg';await im.decode();if(im.naturalWidth!==d.width||im.naturalHeight!==d.height)throw Error('image size');if(Object.keys(d.banks).length!==3)throw Error('banks');all.push(o.value)}return all.length}''')
  await page.screenshot(path='reports/yolo_world_v2s/browser_smoke.png',full_page=False);await browser.close();assert not errors,errors
  Path('reports/yolo_world_v2s/browser_test.json').write_text(json.dumps(dict(status='passed',tasks=count,images_and_json_validated=checks,banks=3,methods=2,next_previous=True,overlay_toggle=True,javascript_errors=errors),indent=2))
asyncio.run(main())
