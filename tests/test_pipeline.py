import tempfile
import unittest
from pathlib import Path
import numpy as np
from controlswitch.pipeline import read_tracking, discover, describe, intervals
from starter.baseline import nearest_player_labels, smooth

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def raw(self,text):
        p=self.root/'tracking.txt';p.write_text(text);return p
    def test_parser_matches_official_with_missing_ball_players_and_ties(self):
        p=self.raw('1:0.4,0.5,1;0.6,0.5,1:0.5,0.5,1\n2:NaN,NaN,NaN;0.6,0.5,1:0.6,0.5,1\n3:0.4,0.5,1;0.6,0.5,1:NaN,NaN,NaN\n')
        teams=np.array([0,1])
        actual,seen,_=read_tracking(p,teams,3)
        expected=nearest_player_labels(str(p),teams,3,(105.3,68))
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(seen.tolist(),[True,True,False])
        for w in [0,25,50,125]:np.testing.assert_array_equal(smooth(actual,w),smooth(expected,w))
    def test_reject_duplicate_or_skipped_ids(self):
        for fid in [1,3]:
            p=self.raw(f'1:0,0,0:0,0,0\n{fid}:0,0,0:0,0,0\n')
            with self.assertRaisesRegex(ValueError,'alignment'):read_tracking(p,np.array([0]),2)
    def test_reject_short_file(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):read_tracking(self.raw('1:0,0,0:0,0,0\n'),np.array([0]),2)
    def test_reject_slot_mismatch(self):
        with self.assertRaisesRegex(ValueError,'slot'):read_tracking(self.raw('1:0,0,0:0,0,0\n'),np.array([0,1]),1)
    def test_discovery_supports_split_folders_and_rejects_duplicate(self):
        for folder,name in [('a','BAR-ATH_tactical_atd.xml'),('b','BAR-ATH_tactical_atd.txt')]:
            d=self.root/folder;d.mkdir();(d/name).touch()
        self.assertEqual(set(discover(self.root)),{'metadata','tracking'})
        (self.root/'BAR-ATH_tactical_atd.txt').touch()
        with self.assertRaisesRegex(ValueError,'Ambiguous'):discover(self.root)
    def test_half_boundary_does_not_join_two_possessions(self):
        labels=np.array(list('HHXXHH'))
        stats=describe(labels,[(0,2),(4,6)])
        self.assertEqual(stats['possession_count'],2)
        self.assertEqual(stats['counts']['X'],0)
        self.assertEqual(intervals(np.array(list('HHAD'))),[[0,2,'H'],[2,3,'A'],[3,4,'D']])

if __name__=='__main__':unittest.main()
