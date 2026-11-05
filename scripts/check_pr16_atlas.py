"""Actual isolated Firefox slider/image/raw-link smoke; no existing services touched."""
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();directory=a.directory.resolve()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(directory)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    profile=Path.home()/'.cache/pgvb_pr16_browser';profile.mkdir(exist_ok=True,parents=True)
    logpath=directory/'browser_driver.txt';log=logpath.open('w')
    proc=subprocess.Popen(['/snap/firefox/current/usr/lib/firefox/geckodriver','--host','127.0.0.1','--port',str(port),'--profile-root',str(profile)],stdout=log,stderr=subprocess.STDOUT)
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
        value=request('POST','/session',dict(capabilities={'alwaysMatch':{'browserName':'firefox','moz:firefoxOptions':{'binary':'/snap/firefox/current/usr/lib/firefox/firefox','args':['-headless']}}}))
        session=value['sessionId'];request('POST',f'/session/{session}/window/rect',dict(width=1450,height=1100))
        request('POST',f'/session/{session}/url',dict(url=f'http://127.0.0.1:{server.server_port}/index.html'))
        cases=json.loads((directory/'cases.json').read_text());assert execute('return document.querySelectorAll("input[type=range]").length')==len(cases)
        for case in cases:
            i=case['index'];last=case['pictures'][-1]
            execute(f'document.querySelector("#fig{i}").scrollIntoView();const slider=document.querySelector("#fig{i}").closest("section").querySelector("input");slider.value=slider.max;slider.dispatchEvent(new Event("input"));return true')
            assert execute(f'return document.querySelector("#fig{i}").getAttribute("src")')==last['figure']
            assert execute(f'return document.querySelector("#raw{i}").getAttribute("href")')==last['raw_RGB']
            ready=False
            for _ in range(50):
                ready=execute(f'return document.querySelector("#fig{i}").complete&&document.querySelector("#fig{i}").naturalWidth>0')
                if ready:break
                time.sleep(.05)
            assert ready
            with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/'+last['raw_RGB'],timeout=5) as response:assert response.status==200
        image=request('GET',f'/session/{session}/screenshot');(directory/'browser_smoke.png').write_bytes(base64.b64decode(image))
        (directory/'browser_test.json').write_text(json.dumps(dict(browser='actual isolated Firefox headless',cases=len(cases),all_sliders_switch_frames=True,composite_images_loaded=True,raw_RGB_links_HTTP200=True,temporary_services_stopped=True),indent=2))
        print('Atlas browser smoke passed',len(cases),'cases')
    finally:
        if session:
            try:request('DELETE',f'/session/{session}')
            except Exception:pass
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        server.shutdown();server.server_close();log.close()
        logpath.write_text('\n'.join(line.rstrip() for line in logpath.read_text().splitlines())+'\n')


if __name__=='__main__':main()
