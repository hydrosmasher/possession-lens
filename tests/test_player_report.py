import tempfile,json,unittest
from pathlib import Path
import numpy as np
from controlswitch.statistics import tracklet_statistics,team_statistics
from controlswitch.player_report import export_player_report
class PlayerReportTests(unittest.TestCase):
 def test_intervals_and_unassigned_reconcile_and_corruption_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);labels=np.array(list('HHHAA'));xy=np.zeros((5,2,2));xy[:,1]=10;ball=np.zeros((5,2));ball[3:]=10
   ledger=tracklet_statistics(labels,xy,ball,np.array([0,1]))
   (p/'team-statistics.json').write_text(json.dumps(team_statistics(labels)))
   (p/'tracklet-proxies.json').write_text(json.dumps(ledger))
   report=export_player_report(p,p/'out');self.assertTrue(report['team_ledgers_reconciled']);self.assertFalse(report['identities_verified'])
   ledger['bouts'][0]['seconds']+=1;(p/'tracklet-proxies.json').write_text(json.dumps(ledger))
   with self.assertRaises(ValueError):export_player_report(p,p/'bad')
