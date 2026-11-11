import unittest
import tempfile
from pathlib import Path
from controlswitch.external import MATCHES,FRESH_MATCHES,source_files
from controlswitch.rnn_fresh_test import verify_metadata

class FreshProtocolTests(unittest.TestCase):
 def test_fresh_match_does_not_change_original_split(self):
    self.assertEqual(set(MATCHES),{'J03WOH','J03WPY','J03WMX'})
    self.assertTrue(set(MATCHES).isdisjoint(FRESH_MATCHES))
    self.assertIn('J03WN1.xml',source_files('J03WN1')[0])
 def test_role_validation_rejects_swapped_teams(self):
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'meta.xml'
        prefix='<MatchInformation><General MatchId="DFL-MAT-J03WN1" CompetitionId="DFL-COM-000001" '
        p.write_text(prefix+'HomeTeamId="DFL-CLU-00000S" GuestTeamId="DFL-CLU-00000B" /></MatchInformation>')
        self.assertEqual(verify_metadata(p)['match_id'],'J03WN1')
        p.write_text(prefix+'HomeTeamId="DFL-CLU-00000B" GuestTeamId="DFL-CLU-00000S" /></MatchInformation>')
        with self.assertRaises(ValueError):verify_metadata(p)
        with self.assertRaises(ValueError):verify_metadata(p,'J03WMX')
