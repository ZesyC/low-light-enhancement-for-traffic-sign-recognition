"""Direct ZIP input must stay pinned and never require notebook credentials."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.tools.baseline_input import download_original, normalize_original


class BaselineInputTests(unittest.TestCase):
    def setUp(self):
        folder=Path(__file__).resolve().parents[1]/'tmp/tests'
        folder.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=folder)
        self.path=Path(self.temp.name)/'baseline.zip'

    def tearDown(self):
        self.temp.cleanup()

    def responses(self, content):
        listing=MagicMock()
        listing.json.return_value={'files':[{'fileName':'gtsrb-notebook-results.zip','url':'https://storage.googleapis.com/test/zip'}]}
        download=MagicMock()
        download.__enter__.return_value=download
        download.iter_content.return_value=[content[:3],content[3:]]
        return listing,download

    def test_download_discovers_v1_url_without_auth_and_verifies_bytes(self):
        content=b'PK fixture bytes'
        listing,download=self.responses(content)
        with patch('src.tools.baseline_input.requests.post',return_value=listing) as post, \
             patch('src.tools.baseline_input.requests.get',return_value=download) as get, \
             patch('src.tools.baseline_input.ORIGINAL_SHA256',hashlib.sha256(content).hexdigest()):
            self.assertEqual(download_original(self.path).read_bytes(),content)
        self.assertEqual(post.call_args.kwargs['json']['versionLabel'],'v1')
        self.assertNotIn('auth',post.call_args.kwargs)
        self.assertNotIn('auth',get.call_args.kwargs)

    def test_wrong_download_hash_is_rejected_and_partial_removed(self):
        listing,download=self.responses(b'incorrect archive')
        with patch('src.tools.baseline_input.requests.post',return_value=listing), \
             patch('src.tools.baseline_input.requests.get',return_value=download):
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                download_original(self.path)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix('.partial.zip').exists())

    def test_unknown_archive_cannot_use_legacy_normalization(self):
        self.path.write_bytes(b'untrusted input')
        with self.assertRaisesRegex(ValueError,'pinned original'):
            normalize_original(self.path,self.path.parent/'normalized.zip')
