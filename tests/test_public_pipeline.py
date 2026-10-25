"""Publishing contracts: preserve evidence, never delete outside generated cache."""
import json
from pathlib import Path
import unittest
import zipfile
import io
import csv
import shutil
import errno
from unittest.mock import patch

import numpy as np

import test_research_evidence as fixtures
from src.experiment import release_image_cache
from src.tools.result_archive import pack


class PublicPipelineTests(unittest.TestCase):
    setUp=fixtures.EvidenceContractTests.setUp
    tearDown=fixtures.EvidenceContractTests.tearDown

    def test_notebook_exports_never_rename_across_runtime_and_output_mounts(self):
        import contextlib
        from src.common import digest
        notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/gtsrb-low-light-enhancement-for-traffic-signs.ipynb').read_text(encoding='utf-8'))
        code=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and "final_zip = PUBLIC/" in ''.join(c['source']))
        runtime=self.root/'runtime';public=self.root/'public'
        runtime.mkdir();public.mkdir()
        stages=[];packed=[]
        def fake_pack(folder,output,**kwargs):
            packed.append(Path(output))
            with zipfile.ZipFile(output,'w') as z:z.writestr('READ_ME_FIRST.md','fixture')
        original=Path.replace
        def same_mount_only(source,target):
            if not source.is_relative_to(public) or not Path(target).is_relative_to(public):
                raise OSError(errno.EXDEV,'Invalid cross-device link')
            return original(source,target)
        scope=dict(ROOT=runtime,PUBLIC=public,OUT=runtime/'outputs',EXT=runtime/'extension',
            pack=fake_pack,status=lambda *args,**kwargs:stages.append((args,kwargs)),
            digest=digest,RUN_EXTENSION_FULL_TEST=True)
        with patch.object(Path,'replace',same_mount_only),contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code,'notebook final export','exec'),scope)
        self.assertTrue(all(p.parent==public for p in packed))
        self.assertEqual(stages[-1][0],('completed',))
        self.assertTrue((public/'gtsrb-notebook-results.zip').is_file())
        self.assertTrue((public/'gtsrb-subset-results.zip').is_file())

    def test_extension_archive_keeps_scopes_separate(self):
        from src.tools.research_evidence import read_csv,write_csv
        rows=read_csv(self.root/'summary.csv')
        for r in rows:r.update(psnr_mean=20,ssim_mean=.5,latency_ms=1)
        write_csv(self.root/'summary.csv',rows)
        section=self.root/'extension/subset';section.mkdir(parents=True)
        for name in ['protocol.json','runs.json','summary.csv']:shutil.copy2(self.root/name,section/name)
        shutil.copytree(self.root/'predictions',section/'predictions')
        target=self.root/'extended.zip'
        pack(self.root,target,subset=self.root,extension=self.root/'extension',include_checkpoints=False)
        with zipfile.ZipFile(target) as z:
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
            self.assertIn('07_supplemental/subset/protocol.json',z.namelist())
            self.assertNotIn('07_supplemental/ablation/protocol.json',z.namelist())
            self.assertIn('08_processing_extension/subset/confusion_matrices.npz',z.namelist())
            table=list(csv.DictReader(io.StringIO(z.read('08_processing_extension/subset/predictions.csv').decode())))
            self.assertEqual(len(table),6)

    def test_readable_archive_preserves_predictions_and_confusions(self):
        from src.tools.research_evidence import read_csv, write_csv
        rows=read_csv(self.root/'summary.csv')
        for r in rows:r.update(psnr_mean=20,ssim_mean=.5,latency_ms=1)
        write_csv(self.root/'summary.csv',rows)
        output=self.root/'readable.zip'
        pack(self.root,output,include_checkpoints=False)
        with zipfile.ZipFile(output) as z:
            rows=list(csv.DictReader(io.StringIO(z.read('03_predictions/test.csv').decode())))
            self.assertEqual(len(rows),6)
            self.assertEqual(len({r['run_id'] for r in rows}),2)
            matrices=np.load(io.BytesIO(z.read('02_analysis/confusion_matrices.npz')),allow_pickle=False)
            self.assertEqual(matrices['matrices'].shape,(2,43,43))
            self.assertTrue(np.all(matrices['matrices'].sum(axis=(1,2))==3))
            self.assertIn('READ_ME_FIRST.md',z.namelist())
            self.assertIn('SHA256.json',z.namelist())

    def test_cache_release_rejects_arbitrary_parent_directory(self):
        path=self.root/'do-not-delete'
        path.mkdir()
        with self.assertRaisesRegex(ValueError,'outside'):
            release_image_cache(self.root,[dict(path=str(path/'image.npy'),sample_id='a')])
        self.assertTrue(path.exists())

    def test_cache_release_requires_completed_matching_metadata(self):
        folder=self.root/'data/cache'/'0123456789abcdef'
        folder.mkdir(parents=True)
        meta=folder/'metadata.json'
        meta.write_text(json.dumps({'status':'running','sample_ids':['a']}))
        records=[dict(path=str(folder/'image.npy'),sample_id='a')]
        with self.assertRaisesRegex(ValueError,'metadata'):
            release_image_cache(self.root,records)
        self.assertTrue(folder.exists())
        meta.write_text(json.dumps({'status':'completed','sample_ids':['a']}))
        release_image_cache(self.root,records)
        self.assertFalse(folder.exists())


if __name__=='__main__':
    unittest.main()
