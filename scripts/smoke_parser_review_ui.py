"""Automated UI functionality only; never creates dataset gold labels."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
 p=argparse.ArgumentParser();p.add_argument('--html',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
 with sync_playwright() as play:
  browser=play.chromium.launch(headless=True,args=['--no-sandbox']);context=browser.new_context(accept_downloads=True);page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(a.html.resolve().as_uri());assert page.locator('#candidates').is_hidden();checks.append('candidate graphs initially hidden')
  eligible=page.evaluate("D.inputs.findIndex(r=>r.landmark_catalog.some(n=>r.text.includes(n)))");assert eligible>=0;page.select_option('#select',str(eligible));text=page.locator('#source').inner_text();phrase=text[:min(18,len(text))];landmark=page.evaluate("D.inputs[pos].landmark_catalog.find(n=>D.inputs[pos].text.includes(n))");assert landmark
  def select_phrase(value):
   page.evaluate('''v=>{let n=document.getElementById('source').firstChild,a=n.textContent.indexOf(v);let r=document.createRange();r.setStart(n,a);r.setEnd(n,a+v.length);let s=window.getSelection();s.removeAllRanges();s.addRange(r)}''',value)
  select_phrase(phrase);page.click('#addentity');graph=json.loads(page.locator('#editor').input_value());assert graph['entities'][0]['text']==phrase;checks.append('source selection adds literal target span')
  page.select_option('#entityrole','landmark');page.select_option('#entitytype','road');select_phrase(landmark);page.click('#addentity');page.select_option('#subject','human_0');page.select_option('#object','human_1');page.fill('#relation','on');page.click('#addrelation');assert len(json.loads(page.locator('#editor').input_value())['relations'])==1;checks.append('directed relation added')
  page.fill('#reviewer','AUTOMATED_UI_TEST_NOT_HUMAN_GOLD');page.check('#complete');page.click('#save');assert '保存成功' in page.locator('#status').inner_text();checks.append('save and independent-blind status')
  page.click('#next');page.click('#prev');assert len(json.loads(page.locator('#editor').input_value())['entities'])==2;page.reload();assert len(json.loads(page.locator('#editor').input_value())['entities'])==2;checks.append('navigation and page reload restore annotation')
  with page.expect_download() as info:page.click('#export')
  download=info.value;exported=json.loads(Path(download.path()).read_text());assert len(exported)==1 and exported[0]['independent'];checks.append('JSON export works')
  with page.expect_download() as info:page.click('#exportcsv')
  assert 'graph' in Path(info.value.path()).read_text();checks.append('CSV export works')
  page.click('#reveal');page.select_option('#parser','B');assert not page.locator('#candidates').is_hidden();page.click('#save')
  with page.expect_download() as info:page.click('#export')
  exported=json.loads(Path(info.value.path()).read_text());assert exported[0]['independent'] is False;checks.append('viewed candidates excluded from independent gold')
  page.screenshot(path=str(a.output.with_suffix('.png')),full_page=False);assert not errors,errors;context.close();browser.close()
 a.output.write_text(json.dumps(dict(checks=checks,passed=True,automated_test_only=True,human_gold_records_created=0),indent=2));print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
