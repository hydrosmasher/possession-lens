"""Loopback-only judge dashboard; no hosted integrations or uploads."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import json
from .dashboard import demo_payload,from_run

def serve(run_dir=None,port=8765):
    payload=from_run(run_dir) if run_dir else demo_payload();web=Path(__file__).parent/'web'
    example=web.parents[1]/'examples/external-named-analysis.json'
    if example.exists():payload['bundled_named_example']=json.loads(example.read_text())
    library=web.parents[1]/'examples/match-library.json'
    if library.exists():payload['match_library']=json.loads(library.read_text())
    evidence=web.parents[1]/'reports/external/rnn-europe-context-001-report.json'
    if evidence.exists():
        from .evidence import summary
        payload['external_evidence']=summary(json.loads(evidence.read_text()))
    from .evidence import research_summary
    research=research_summary(web.parents[1])
    if research:payload['research_status']=research
    static={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path in static:
                name,mime=static[path];body=(web/name).read_bytes()
            elif path=='/api/dashboard':mime='application/json';body=json.dumps(payload,allow_nan=False).encode()
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'");self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);print(f'Possession Lens: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
