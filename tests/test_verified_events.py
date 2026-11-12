import unittest,json
from pathlib import Path
class VerifiedEventTests(unittest.TestCase):
 def test_provider_totals_and_roster_joins_reconcile(self):
  p=Path(__file__).resolve().parents[1]/'examples/verified-events-J03WN1.json';r=json.loads(p.read_text());self.assertEqual(r['source_kind'],'licensed_provider_events')
  self.assertEqual(r['git_blob'],'67b982007b8e3f5de83dcd104779acc92b76a813');self.assertEqual(r['teams']['H']['goals'],3);self.assertEqual(r['teams']['A']['goals'],0)
  self.assertEqual(r['teams']['H']['shots'],10);self.assertEqual(r['teams']['A']['shots'],11)
  for t in 'HA':
   self.assertIsNone(r['teams'][t]['touches'])
   for k in ['completed_passes','shots','shots_on_target','goals']:self.assertEqual(sum(p[k] for p in r['players'].values() if p['team']==t),r['teams'][t][k])
  for e in r['timeline']:self.assertIn(e['person_id'],r['players']);self.assertEqual(e['player_name'],r['players'][e['person_id']]['player_name'])
