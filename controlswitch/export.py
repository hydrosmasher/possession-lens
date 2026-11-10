"""Export only the invented demo; never embeds licensed competition data."""
from pathlib import Path
import json
from .pipeline import demo_data

def export_demo(out):
    web=Path(__file__).parent/'web'
    summary,timelines=demo_data()
    payload=json.dumps([summary,timelines],allow_nan=False).replace('<','\\u003c')
    html=(web/'index.html').read_text()
    html=html.replace('<link rel="stylesheet" href="/style.css">','<style>'+(web/'style.css').read_text()+'</style>')
    html=html.replace('<script src="/app.js" defer></script>','')
    html=html.replace('</body>','<script>window.__CONTROL_SWITCH_DATA__='+payload+';</script><script>'+(web/'app.js').read_text()+'</script></body>')
    html=html.replace('href="/" aria-label="Possession Lens home"','href="#" aria-label="Possession Lens home"')
    html=html.replace('href="/external-pilot-results.html"','href="external-pilot-results.html"').replace('href="/interaction-forecast-results.html"','href="interaction-forecast-results.html"')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(html)
