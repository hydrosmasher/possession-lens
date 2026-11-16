import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from controlswitch import recurrent_pipeline as pipeline
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'Optional recurrent export dependency')
class QualityExportTests(unittest.TestCase):
    def test_quality_report_does_not_change_exported_labels(self):
        n=200;labels=np.array(['H']*80+['A']*70+['D']*50)
        s={'ball':np.zeros((n,2)),'xy':np.ones((n,6,2)),'teams':np.array([0,0,0,1,1,1])};s['ball'][30:100]=np.nan
        bundle={'mix':.5,'config':{'candidate':'test','base_config':{},'gru':{}},'base':{'config':{'assets':{}}}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';config.write_text('{}');meta=root/'metadata.txt';meta.write_text('synthetic');track=root/'tracking.txt';track.write_text('synthetic');out=root/'out'
            with patch.object(pipeline,'N_FRAMES',n),patch.object(pipeline,'BLOCKS',[(0,n)]),patch.object(pipeline,'load_pipeline',return_value=bundle),patch('controlswitch.submission.read_observations',return_value=(s,np.array([105,68]),['Home','Away'],{'metadata':meta,'tracking':track})),patch('controlswitch.submission.predict_halves',return_value=labels.copy()):
                pipeline.export(root,out,config)
            actual=pd.read_csv(out/'possession.csv');np.testing.assert_array_equal(actual['label'].to_numpy(),labels)
            quality=json.loads((out/'observation-quality.json').read_text());assert quality['summary']['ball_missing_frames']==70;assert not quality['possession_labels_used'];assert not quality['fitting_performed']
