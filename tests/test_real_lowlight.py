"""Ensure the real-image experiment rejects invalid annotations and duplicate crops."""
import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from src.tools.real_lowlight import load_manifest


class RealManifestTests(unittest.TestCase):
    def setUp(self):
        scratch=Path(__file__).resolve().parents[1]/'tmp/tests'
        scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch)
        self.root=Path(self.temp.name)
        Image.new('RGB',(40,30),(20,20,20)).save(self.root/'fixture.png')
        self.row=dict(sample_id='fixture',image_path='fixture.png',class_id=14,group_id='fixture-track',
                      x1=1,y1=2,x2=30,y2=25,source_url='unit-test-fixture',license='unit-test-only',
                      annotator='test',capture_kind='real_lowlight')

    def tearDown(self):
        self.temp.cleanup()

    def write(self,rows):
        path=self.root/'manifest.csv'
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.row));w.writeheader();w.writerows(rows)
        return path

    def test_empty_template_cannot_produce_real_results(self):
        with self.assertRaisesRegex(ValueError,'nonempty'):
            load_manifest(self.write([]),self.root)

    def test_unknown_class_is_rejected_before_inference(self):
        with self.assertRaisesRegex(ValueError,'0..42'):
            load_manifest(self.write([dict(self.row,class_id=43)]),self.root)

    def test_duplicate_crop_is_rejected_even_with_another_sample_id(self):
        with self.assertRaisesRegex(ValueError,'Duplicate crop'):
            load_manifest(self.write([self.row,dict(self.row,sample_id='other')]),self.root)

    def test_bbox_outside_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'ROI'):
            load_manifest(self.write([dict(self.row,x2=41)]),self.root)

    def test_valid_roi_matches_classifier_input_contract(self):
        samples=load_manifest(self.write([self.row]),self.root)
        self.assertEqual(samples[0][1].shape,(64,64,3))
        self.assertEqual(samples[0][0]['class_id'],14)


if __name__=='__main__':
    unittest.main()
