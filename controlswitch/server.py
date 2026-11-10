"""Small standard-library HTTP server; loopback only, no hosted integrations."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
from urllib.parse import urlsplit
from .pipeline import demo_data

def serve(run_dir=None, port=8765):
    run_dir=Path(run_dir).resolve() if run_dir else None
    if run_dir:
        summary=json.loads((run_dir/'summary.json').read_text())
        timeline=json.loads((run_dir/'timeline.json').read_text())
        if summary.get('schema_version')!=1 or summary.get('mode')!='licensed_local':
            raise ValueError('Unsupported run summary')
    else:
        summary,timeline=demo_data()
    web=Path(__file__).parent/'web'
    static={'/':('index.html','text/html; charset=utf-8'),
            '/external-pilot-results.html':('external-pilot-results.html','text/html; charset=utf-8'),
            '/interaction-forecast-results.html':('interaction-forecast-results.html','text/html; charset=utf-8'),
            '/app.js':('app.js','text/javascript; charset=utf-8'),
            '/style.css':('style.css','text/css; charset=utf-8')}
    names={v['name'] for v in summary['variants']}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            download=None
            if path in static:
                filename,mime=static[path]
                if not (web/filename).is_file():self.send_error(404);return
                body=(web/filename).read_bytes()
            elif path in ('/api/summary','/api/timeline'):
                mime='application/json';body=json.dumps(summary if path.endswith('summary') else timeline,allow_nan=False).encode()
            elif path.startswith('/download/') and run_dir:
                name=path.removeprefix('/download/').removesuffix('.csv')
                if name not in names or path!=f'/download/{name}.csv':
                    self.send_error(404);return
                mime='text/csv';download=f'{name}.csv';body=(run_dir/download).read_bytes()
            else:
                self.send_error(404);return
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Control Switch: http://127.0.0.1:{port}',flush=True)
    print('Licensed local run' if run_dir else 'Synthetic demo; no licensed match data loaded',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
