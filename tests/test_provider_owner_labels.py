import unittest,tempfile,json,numpy as np
from pathlib import Path
from controlswitch.skillcorner_europe import load_match
from controlswitch.player_set_state import player_nodes
class ProviderOwnerTests(unittest.TestCase):
 def test_only_exact_observed_team_consistent_owner_is_labelled(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'test';p.mkdir();meta={'players':[{'trackable_object':11,'team_id':1},{'trackable_object':22,'team_id':2}],'home_team':{'id':1},'ball':{'trackable_object':99},'pitch_length':105,'pitch_width':68};(p/'match_data.json').write_text(json.dumps(meta));frames=[]
   for period in (1,2):
    for j in range(12):
     owner=11 if j!=1 else 888;group='home team' if j!=6 else 'away team';data=[{'trackable_object':11,'x':j*.02,'y':0},{'trackable_object':22,'x':10,'y':0},{'trackable_object':99,'x':j*.02+.1,'y':0}]
     if j==2:data=[o for o in data if o['trackable_object']!=11]
     if j==4:data=[o for o in data if o['trackable_object']!=99]
     frames.append({'frame':j,'period':period,'possession':{'trackable_object':owner,'group':group},'data':data})
   (p/'structured_data.json').write_text(json.dumps(frames));old,_=load_match(td,'test');new,audit=load_match(td,'test',include_owner=True)
   for (a,pitch),(b,_) in zip(old,new):
    self.assertNotIn('owner',a)
    for key in ('xy','ball','speed','ball_speed','y'):np.testing.assert_array_equal(a[key],b[key])
    self.assertEqual(b['owner'][0],0)
    for i in (5,10,15):self.assertEqual(b['owner'][i],-1)
    nodes,indices=player_nodes(b,pitch,return_indices=True);matches=indices[0].reshape(-1)==b['owner'][0];self.assertEqual(matches.sum(),1);self.assertEqual(nodes[0].reshape(-1,21)[matches][0,20],1)
   self.assertGreater(audit[0]['owner_labelled_native_frames'],0)
if __name__=='__main__':unittest.main()
