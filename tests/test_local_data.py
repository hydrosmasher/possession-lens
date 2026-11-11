import unittest,tempfile
from pathlib import Path
from controlswitch.local_data import import_local
from controlswitch.external import source_files

class LocalImportTests(unittest.TestCase):
 def test_import_is_offline_idempotent_and_keeps_original(self):
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'Downloads';source.mkdir();dest=Path(tmp)/'data/external/raw'
        name=source_files('J03WN1')[1];p=source/name
        p.write_text('<Positions><MetaData MatchId="DFL-MAT-J03WN1"/><Frame N="1"/></Positions>')
        (source/'unfinished.xml.crdownload').write_text('partial')
        r=import_local(source,dest,['J03WN1']);self.assertEqual(len(r['assets']),1);self.assertTrue(p.exists())
        r=import_local(source,dest,['J03WN1']);self.assertEqual(r['assets'][0]['status'],'already_imported')
        p.write_text('<Positions><MetaData MatchId="DFL-MAT-J03WN1"/><Frame N="2"/></Positions>')
        with self.assertRaises(ValueError):import_local(source,dest,['J03WN1'])
 def test_incorrect_identity_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'Downloads';source.mkdir()
        (source/source_files('J03WN1')[1]).write_text('<Positions><MetaData MatchId="DFL-MAT-J03WMX"/><Frame/></Positions>')
        with self.assertRaises(ValueError):import_local(source,Path(tmp)/'out',['J03WN1'])
 def test_legacy_downloader_cannot_silently_access_network(self):
    from unittest.mock import patch
    from controlswitch.external import download_external
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'Downloads';source.mkdir()
        with patch('urllib.request.urlopen',side_effect=AssertionError('Network must stay off')) as fetch:
            with self.assertRaisesRegex(ValueError,'Network is off'):
                download_external(Path(tmp)/'out',['J03WN1'],source=source)
            fetch.assert_not_called()
