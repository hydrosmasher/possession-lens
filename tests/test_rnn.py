import tempfile
import unittest
from pathlib import Path
import numpy as np
from controlswitch.rnn import initialise,normalise,loss_gradients,predict
from controlswitch.statistics import team_statistics,tracklet_statistics

class RecurrentTests(unittest.TestCase):
 def test_bptt_gradients_match_finite_differences(self):
    x=np.array([[.2,.4],[.4,.1],[.3,-.2]],dtype=np.float32);y=np.array([0,1,2]);m=initialise(x,hidden=3);z=normalise(m,x)
    loss,g,_=loss_gradients(m,z,y)
    for k,index in [('wx',(0,1)),('wh',(1,2)),('wy',(1,0)),('bh',(0,)),('by',(2,))]:
        original=m[k][index];eps=.002
        m[k][index]=original+eps;plus=loss_gradients(m,z,y)[0]
        m[k][index]=original-eps;minus=loss_gradients(m,z,y)[0];m[k][index]=original
        self.assertAlmostEqual(float(g[k][index]),(plus-minus)/(2*eps),delta=.0002)
 def test_hidden_state_is_preserved_across_inference_chunks(self):
    x=np.random.default_rng(42).normal(size=(20,3)).astype(np.float32);m=initialise(x,hidden=4)
    np.testing.assert_allclose(predict(m,x,chunk=3),predict(m,x,chunk=100),atol=1e-6)
 def test_team_denominator_excludes_dead_and_unknown(self):
    stats=team_statistics(list('HHADXX'))
    self.assertAlmostEqual(stats['home_possession_percent'],200/3)
    self.assertEqual(team_statistics(list('DDXX'))['home_possession_percent'],None)
 def test_missing_ball_is_unassigned_not_credited_to_player(self):
    xy=np.zeros((2,2,2));xy[:,1]=20;ball=np.array([[0.,0.],[np.nan,np.nan]])
    result=tracklet_statistics(np.array(['H','H']),xy,ball,np.array([0,1]))
    rows=[r for r in result['rows'] if r['team']=='H']
    self.assertEqual(rows[0]['percent_of_team_possession'],50)
    self.assertEqual(rows[-1]['percent_of_team_possession'],50)

if __name__=='__main__':unittest.main()
