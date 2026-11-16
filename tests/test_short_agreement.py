import unittest,numpy as np
from controlswitch.short_agreement import recover_with_agreement
class ShortAgreementTests(unittest.TestCase):
    def test_abstains_on_prior_disagreement_or_ambiguous_receipt(self):
        frozen=np.zeros(100,int);proposal=frozen.copy();proposal[40:50]=1;seen=np.ones(100,bool);score=np.tile([.1,.9],(100,1));valid=np.ones((100,2),bool);prior=np.tile([.7,.3],(100,1))
        p,_=recover_with_agreement(frozen,proposal,seen,score,valid,valid,.5,prior);np.testing.assert_array_equal(p,frozen)
        prior[40:50]=[.3,.7];p,_=recover_with_agreement(frozen,proposal,seen,score,valid,valid,.5,prior);assert (p[40:50]==1).all()
        score[:,0]=.9;p,_=recover_with_agreement(frozen,proposal,seen,score,valid,valid,.5,prior);np.testing.assert_array_equal(p,frozen)
