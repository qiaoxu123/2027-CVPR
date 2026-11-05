"""Exercise the static atlas with an isolated Firefox/WebDriver; no existing service."""
import argparse
import base64
from functools import partial
import http.server
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),partial(http.server.SimpleHTTPRequestHandler,directory=str(a.directory.resolve())))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    port=free_port();log=(a.directory/'browser_driver.txt').open('w')
    profile_root=Path.home()/'.cache'/'pgvb_nearfield_browser';profile_root.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['/snap/firefox/current/usr/lib/firefox/geckodriver','--host','127.0.0.1','--port',str(port),'--profile-root',str(profile_root)],stdout=log,stderr=subprocess.STDOUT)
    session=None
    def request(method,path,data=None):
        body=json.dumps(data).encode() if data is not None else None
        req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=body,method=method,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=55) as response:return json.load(response)['value']
    def execute(script):return request('POST',f'/session/{session}/execute/sync',dict(script=script,args=[]))
    try:
        for _ in range(50):
            try:request('GET','/status');break
            except Exception:time.sleep(.1)
        result=request('POST','/session',dict(capabilities={'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'binary':'/snap/firefox/current/usr/lib/firefox/firefox','args':['-headless']}}}))
        session=result['sessionId']
        request('POST',f'/session/{session}/window/rect',dict(width=1450,height=1100))
        request('POST',f'/session/{session}/url',dict(url=f'http://127.0.0.1:{server.server_port}/index.html'))
        count=execute('return document.querySelector("#samples").options.length');assert count==93
        assert execute('return document.querySelector("#samples").options.length')==93
        initial=execute('return document.querySelector("#samples").selectedIndex');execute('document.querySelectorAll("button")[1].click();return true');assert execute('return document.querySelector("#samples").selectedIndex')==initial+1
        execute('document.querySelectorAll("button")[0].click();return true');assert execute('return document.querySelector("#samples").selectedIndex')==initial
        execute('let s=document.querySelector("#samples");s.selectedIndex=30;s.dispatchEvent(new Event("change"));return true');assert execute('return document.querySelector("#samples").selectedIndex')==30
        execute('let c=document.querySelector("#candidate");c.selectedIndex=1;c.dispatchEvent(new Event("change"));return true')
        assert execute('return document.querySelector("#candidate").selectedIndex')==1
        execute('document.querySelector("#tokens").click();return true')
        assert execute('return document.querySelector("#tokens").checked') is True
        # Screenshot a real RGB/map/score state, allowing image.onload to finish.
        for _ in range(50):
            ready=execute('return document.querySelector("#raw").getContext("2d").getImageData(0,0,384,384).data.some(v=>v>0)')
            if ready:break
            time.sleep(.1)
        assert ready
        image=request('GET',f'/session/{session}/screenshot')
        (a.directory/'browser_smoke.png').write_bytes(base64.b64decode(image))
        (a.directory/'browser_test.json').write_text(json.dumps(dict(browser='isolated Firefox headless',samples=count,task_switch=True,candidate_switch=True,polygon_tokens_toggle=True,RGB_canvas_nonempty=True,screenshot='browser_smoke.png',note='Temporary local HTTP and WebDriver services shut down after test.'),indent=2)+'\n')
        print('Polygon matching browser interactions passed')
    finally:
        if session:
            try:request('DELETE',f'/session/{session}')
            except Exception:pass
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        server.shutdown();server.server_close();log.close()
        log_path=a.directory/'browser_driver.txt'
        log_path.write_text('\n'.join(line.rstrip() for line in log_path.read_text().splitlines())+'\n')

if __name__=='__main__':main()
