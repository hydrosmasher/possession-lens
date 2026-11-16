import unittest,numpy as np
from controlswitch.short_evidence_audit import diagnose
class ShortEvidenceAuditTests(unittest.TestCase):
    def inputs(self):
        y=np.r_[np.zeros(20,int),np.ones(20,int),np.zeros(20,int)];p=np.zeros(60,np.int8);q=np.tile([.95,.04,.01],(60,1));return y,p,q
    def test_recognises_evidence_suppressed_by_decoder(self):
        y,p,q=self.inputs();q[20:40]=[.1,.89,.01];rows,raw=diagnose(y,p,q);self.assertEqual(next(x['cause'] for x in rows if x['team']==1),'decoder_suppressed_recoverable');np.testing.assert_array_equal(raw,y)
    def test_distinguishes_weak_fragmented_and_dead_blocked(self):
        y,p,q=self.inputs();rows,_=diagnose(y,p,q);self.assertEqual(next(x['cause'] for x in rows if x['team']==1),'weak_team_evidence')
        q[20:40:2]=[.1,.89,.01];rows,_=diagnose(y,p,q);self.assertEqual(next(x['cause'] for x in rows if x['team']==1),'fragmented_or_misaligned_evidence')
        p[20:40]=2;rows,raw=diagnose(y,p,q);self.assertEqual(next(x['cause'] for x in rows if x['team']==1),'dead_mask_blocks');np.testing.assert_array_equal(raw==2,p==2)
