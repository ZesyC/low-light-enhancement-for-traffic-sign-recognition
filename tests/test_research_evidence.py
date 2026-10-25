"""Contract tests: corrupt evidence must not be published as verified results."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.common import config_id, digest
from src.metrics import classification
from src.tools.research_evidence import audited_runs, bundle, write_csv


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/'tmp/tests'
        scratch.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        (self.root/'predictions').mkdir()
        runs=[dict(condition='medium',method=m,noise_seed=101,train_seed=11) for m in ['identity','clahe']]
        protocol={'runs':runs,'sample_ids':['a','b','c'],'subset_id':'fixture','pilot':True}
        (self.root/'protocol.json').write_text(json.dumps(protocol))
        (self.root/'runs.json').write_text(json.dumps([dict(r,status='completed') for r in runs]))
        summaries=[]
        for run in runs:
            preds=[1,1,2] if run['method']=='identity' else [0,0,2]
            rows=[dict(sample_id=i,group_id=i,class_id=c,prediction=p,noise_seed=101,output_path='fixture.npy') for i,c,p in zip(['a','b','c'],[0,1,2],preds)]
            score=classification([0,1,2],preds)
            rid=config_id(run)
            write_csv(self.root/'predictions'/f'{rid}.csv',rows)
            summaries.append(dict(run_id=rid,**run,n_images=3,**{k:score[k] for k in ['accuracy','macro_f1','weighted_f1']}))
        write_csv(self.root/'summary.csv',summaries)

    def tearDown(self):
        self.temp.cleanup()

    def test_bundle_contains_hashes_of_actual_prediction_files(self):
        target=self.root/'evidence.zip'
        bundle(self.root,target)
        with zipfile.ZipFile(target) as z:
            hashes=json.loads(z.read('SHA256.json'))
            for rel,sha in hashes.items():
                self.assertEqual(sha,digest(self.root/rel))
            self.assertEqual(len([n for n in hashes if n.startswith('predictions/')]),2)

    def test_changed_reported_accuracy_is_rejected(self):
        path=self.root/'summary.csv'
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
        rows[0]['accuracy']='1.0'; write_csv(path,rows)
        with self.assertRaisesRegex(ValueError,'Metric mismatch'):
            audited_runs(self.root)

    def test_duplicate_sample_cannot_replace_missing_sample(self):
        path=next((self.root/'predictions').glob('*.csv'))
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
        rows[2]=rows[0]; write_csv(path,rows)
        with self.assertRaisesRegex(ValueError,'Missing or duplicate'):
            audited_runs(self.root)

    def test_same_metrics_do_not_hide_inconsistent_ground_truth(self):
        path=next((self.root/'predictions').glob('*.csv'))
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
        rows[0]['group_id']='different'
        write_csv(path,rows)
        with self.assertRaisesRegex(ValueError,'Label/group mismatch'):
            audited_runs(self.root)

    def test_partial_matrix_is_not_accepted(self):
        path=self.root/'runs.json'
        rows=json.loads(path.read_text()); rows[1]['status']='failed'; path.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            audited_runs(self.root)


if __name__=='__main__':
    unittest.main()
