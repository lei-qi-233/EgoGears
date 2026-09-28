import json, re
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT=Path(__file__).resolve().parent/'annotation_app'
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path in ('/','/index.html'):
   p=ROOT/'index.html'; mime='text/html; charset=utf-8'
  elif self.path.startswith('/media/'):
   key=self.path.split('/')[-1]
   entry=json.loads((ROOT/'media.json').read_text()).get(key)
   if not entry: self.send_error(404);return
   p=Path(entry);mime='video/mp4'
  else: self.send_error(404);return
  size=p.stat().st_size; start=0;end=size-1
  header=self.headers.get('Range')
  if header:
   m=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
   if not m: self.send_error(416);return
   start=int(m[1]);end=min(int(m[2]) if m[2] else end,end)
   if start>end: self.send_error(416);return
  self.send_response(206 if header else 200)
  self.send_header('Content-Type',mime)
  self.send_header('Content-Length',str(end-start+1))
  self.send_header('Accept-Ranges','bytes')
  self.send_header('Cache-Control','no-cache')
  if header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  try:
   with p.open('rb') as f:
    f.seek(start); remaining=end-start+1
    while remaining:
     data=f.read(min(1024*1024,remaining))
     if not data:break
     self.wfile.write(data);remaining-=len(data)
  except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
 print('Review server http://127.0.0.1:18765',flush=True)
 ThreadingHTTPServer(('127.0.0.1',18765),Handler).serve_forever()
