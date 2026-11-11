import unittest,tempfile,json
from pathlib import Path
from controlswitch.dashboard import demo_payload,export_html,export_local

class DashboardTests(unittest.TestCase):
 def test_offline_demo_has_no_remote_assets_and_is_clearly_invented(self):
    with tempfile.TemporaryDirectory() as tmp:
        p=export_html(Path(tmp)/'dashboard.html');s=p.read_text()
        self.assertIn('window.__POSSESSION_LENS_DATA__=',s)
        self.assertIn('synthetic_demo',s)
        self.assertNotIn('<script src=',s);self.assertNotIn('<link rel="stylesheet"',s)
        self.assertNotIn('http://fonts.',s);self.assertNotIn('https://fonts.',s)
 def test_private_run_export_preserves_track_identity_warning(self):
    with tempfile.TemporaryDirectory() as tmp:
        r=Path(tmp)/'run';r.mkdir()
        (r/'team-statistics.json').write_text(json.dumps({'teams':{'H':'Blue','A':'Red'},'model_sha256':'synthetic'}))
        (r/'timeline.json').write_text(json.dumps([[0,25,'H'],[25,50,'A']]))
        (r/'tracklet-proxies.json').write_text(json.dumps({'rows':[],'warning':'unverified tracklets'}))
        p=export_local(r,Path(tmp)/'local');d=json.loads((p.parent/'match-analysis.json').read_text())
        self.assertEqual(d['mode'],'licensed_local');self.assertFalse(d['provenance']['individual_identities_verified'])
        self.assertIn('unverified',d['tracklets']['warning'])
