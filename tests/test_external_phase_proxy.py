import unittest,tempfile
from pathlib import Path
from controlswitch.external_phase_proxy import tracking_clocks,event_phases,corrupt
from controlswitch.external import ALL_MATCHES
class ExternalPhaseProxyTests(unittest.TestCase):
    def test_clocks_align_without_possession_fields(self):
        parts=[]
        for half,hour,start in [('firstHalf','12',100),('secondHalf','13',200)]:
            frames=''.join(f'<Frame N="{start+i}" T="2020-01-01T{hour}:00:00.{ms}+00:00" BallStatus="0" BallPossession="2"/>' for i,ms in enumerate(('000','040','080')));parts.append(f'<FrameSet GameSection="{half}" MatchId="DFL-MAT-J03WMX" TeamId="BALL">{frames}</FrameSet>')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tracking.xml';p.write_text('<root>'+''.join(parts)+'</root>');clocks=tracking_clocks(p,'J03WMX');team=ALL_MATCHES['J03WMX']['home'];e=Path(tmp)/'events.xml';e.write_text(f'<root><Event EventId="1" MatchId="DFL-MAT-J03WMX" EventTime="2020-01-01T12:00:00.040+00:00"><Play Team="{team}" BallPossessionPhase="1"/></Event></root>');phases,audit=event_phases(e,'J03WMX',clocks)
        self.assertEqual(clocks['firstHalf']['first_frame'],100);self.assertEqual(phases['firstHalf'][0]['team'],0);assert not audit['reference_possession_labels_used'];assert phases['firstHalf'][0]['end']==3
    def test_corruption_reproducible_and_inputs_unchanged(self):
        items=[{'start':100*i,'end':100*i+50,'team':i%2,'code':'build up'} for i in range(10)];copy=[dict(x) for x in items];a=corrupt(items,1000,175);self.assertEqual(a,corrupt(items,1000,175));self.assertEqual(items,copy);assert all(0<=x['start']<x['end']<=1000 for x in a)
