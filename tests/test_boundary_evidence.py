import unittest
import numpy as np
from controlswitch.boundary_evidence import combine_evidence
from controlswitch.structured_had import supported_score

class BoundaryEvidenceTests(unittest.TestCase):
    def test_minimum_requires_both_sources(self):
        p=np.tile([.9,.1,0.],(100,1));p[50:]=[.1,.9,0.];q=np.full(100,.7)
        support=supported_score(p,np.ones(100));a=combine_evidence(p,q,'minimum')
        self.assertTrue(np.all(a<=q));self.assertTrue(np.all(a<=support))
        np.testing.assert_allclose(combine_evidence(p[:,[1,0,2]],q,'minimum'),a)
    def test_geometric_evidence_is_bounded_and_zero_when_event_absent(self):
        p=np.tile([.6,.4,0.],(20,1));a=combine_evidence(p,np.zeros(20),'geometric');self.assertFalse(a.any())
        b=combine_evidence(p,np.ones(20),'geometric');self.assertTrue(((b>=0)&(b<=1)).all())
    def test_unknown_combination_rejected(self):
        with self.assertRaises(ValueError):combine_evidence(np.ones((3,3)),np.ones(3),'unknown')
if __name__=='__main__':unittest.main()
