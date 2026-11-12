import tempfile,unittest
from pathlib import Path
from controlswitch.player_mapping import external_roster,aggregate_named
class MappingTests(unittest.TestCase):
 def test_roster_requires_matching_match_and_team_ids(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'metadata.xml';p.write_text('<Root><General MatchId="DFL-MAT-J03WN1" HomeTeamId="DFL-CLU-00000S" GuestTeamId="DFL-CLU-00000B"/><Team TeamId="DFL-CLU-00000S"><Player PersonId="a" FirstName="Ada" LastName="Example" ShirtNumber="7"/></Team></Root>')
   r,_=external_roster(p,'J03WN1');self.assertEqual(r[(0,'a')]['player_name'],'Ada Example')
   with self.assertRaises(ValueError):external_roster(p,'J03WQQ')
 def test_half_slots_join_by_stable_id_and_preserve_unassigned(self):
  roster={(0,'a'):{'person_id':'a','player_name':'Ada Example','shirt_number':'7'}}
  ledger={'rows':[{'team':'H','track_slot':1,'associated_control_seconds':2},{'team':'H','track_slot':'UNASSIGNED','associated_control_seconds':1},{'team':'A','track_slot':'UNASSIGNED','associated_control_seconds':0}]}
  rows=aggregate_named([(ledger,{'H':3,'A':0},['a'],[0])]*2,roster)
  self.assertEqual(rows[0]['associated_control_seconds'],4);self.assertAlmostEqual(rows[0]['percent_of_team_possession'],200/3)
  with self.assertRaises(ValueError):aggregate_named([(ledger,{'H':3,'A':0},['different'],[0])],roster)
