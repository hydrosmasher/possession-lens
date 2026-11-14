import itertools,unittest
import numpy as np
from controlswitch.structured_had import supported_score,decode_two,hierarchical_decode,predict

class StructuredHADTests(unittest.TestCase):
    def test_binary_decoder_matches_exhaustive_search(self):
        p=np.array([[.7,.3],[.2,.8],[.1,.9],[.8,.2]]);cost=np.array([1.,.5,2.,.3])
        def objective(path):return sum(np.log(p[i,k]) for i,k in enumerate(path))-sum(cost[i] for i in range(1,4) if path[i]!=path[i-1])
        best=max(itertools.product(range(2),repeat=4),key=objective);np.testing.assert_array_equal(decode_two(p,cost),best)
    def test_uncertain_owner_does_not_imply_dead_ball(self):
        p=np.tile([.3,.3,.4],(50,1));self.assertTrue(np.all(hierarchical_decode(p)<2))
    def test_genuine_stoppage_separates_same_team_spells(self):
        p=np.vstack([np.tile([.999,.0005,.0005],(30,1)),np.tile([.000001,.000001,.999998],(20,1)),np.tile([.999,.0005,.0005],(30,1))])
        result=hierarchical_decode(p);np.testing.assert_array_equal(result,np.array([0]*30+[2]*20+[0]*30))
    def test_support_is_team_invariant_and_penalises_stationary_evidence(self):
        p=np.tile([.9,.1,0],(100,1));q=np.ones(100);a=supported_score(p,q);np.testing.assert_allclose(a,supported_score(p[:,[1,0,2]],q));self.assertTrue(np.all(a<.5))
        p[50:]=[.1,.9,0];self.assertGreater(supported_score(p,q)[50],a[50])
    def test_dispatch_rejects_unknown_method(self):
        with self.assertRaises(ValueError):predict(np.ones((2,3)),np.ones(2),'invalid')
if __name__=='__main__':unittest.main()
