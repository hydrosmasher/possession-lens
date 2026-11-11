import unittest
import numpy as np
from controlswitch.statistics import tracklet_statistics

class TrackletLedgerTests(unittest.TestCase):
 def test_numerical_tie_is_not_arbitrarily_credited(self):
    xy=np.array([[[-1.,0],[1.,0]]]);result=tracklet_statistics(['H'],xy,np.zeros((1,2)),np.array([0,0]))
    rows=[r for r in result['rows'] if r['team']=='H']
    self.assertEqual(rows[-1]['percent_of_team_possession'],100)
    self.assertEqual(result['unassigned_reasons']['H']['numerical_distance_tie'],1)
    self.assertEqual(result['bouts'],[])
 def test_accounting_and_bouts_preserve_gaps_and_team(self):
    xy=np.array([[[0.,0],[8.,0]]]*5);ball=np.array([[0.,0],[0.,0],[np.nan,np.nan],[0.,0],[8.,0]])
    result=tracklet_statistics(['H','H','H','H','A'],xy,ball,np.array([0,1]))
    for team in 'HA':self.assertAlmostEqual(sum(r['percent_of_team_possession'] or 0 for r in result['rows'] if r['team']==team),100)
    self.assertEqual([r['end_frame_exclusive']-r['start_frame'] for r in result['bouts']],[2,1,1])
    self.assertEqual(result['unassigned_reasons']['H']['missing_ball'],1)
