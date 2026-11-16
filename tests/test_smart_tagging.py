import unittest,tempfile
from pathlib import Path
import numpy as np
from controlswitch.smart_tagging import load,phase_evidence
class SmartTaggingTests(unittest.TestCase):
    def xml(self):return '''<file><ALL_INSTANCES><instance><start>1</start><end>3</end><code>BUILD UP</code><label><group>Team</group><text>Barcelona</text></label></instance><instance><start>2</start><end>4</end><code>DEFENSIVE TRANSITION</code><label><group>team</group><text>Atletic Club</text></label></instance><instance><start>3</start><end>5</end><code>SET PIECES</code><label><group>Team</group><text>N/A</text></label><label><group>Type</group><text>Throw-in</text></label></instance></ALL_INSTANCES></file>'''
    def test_case_normalisation_documented_alias_and_video_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tag.xml';p.write_text(self.xml());items,a=load(p,['Barcelona','Athletic Club'],25,150)
        assert items[0]['start']==25 and items[0]['end']==75;assert items[1]['team']==1;assert items[2]['team'] is None;assert a['unknown_team_instances']==1
        f=phase_evidence(items,150);assert f['unopposed_team_evidence'][60,0];assert f['restart_window'][90];assert not f['unopposed_team_evidence'][110].any()
    def test_conflicting_ownership_abstains_and_goals_are_not_possession(self):
        intervals=[{'start':0,'end':30,'team':0,'code':'build up'},{'start':10,'end':20,'team':1,'code':'progression'},{'start':30,'end':40,'team':0,'code':'goals'}];f=phase_evidence(intervals,50);assert f['conflict'][15] and not f['unopposed_team_evidence'][15].any();assert not f['unopposed_team_evidence'][35].any()
    def test_bad_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tag.xml';p.write_text(self.xml().replace('<start>1</start>','<start>nan</start>'))
            with self.assertRaises(ValueError):load(p,['Barcelona','Athletic Club'])
