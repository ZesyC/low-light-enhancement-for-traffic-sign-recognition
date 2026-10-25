"""Reject incomplete/tampered scientific evidence before report generation."""
import csv
import io
import json
import unittest
import zipfile

from src.common import config_id
from src.metrics import classification
from src.tools.verified_results import audit_scope


def fixture(duplicate_sample=False, wrong_metric=False):
    methods = ['identity', 'gamma', 'he', 'clahe', 'retinex', 'zero_dce', 'retinexformer', 'diffusion']
    runs = [dict(condition='clean', method='clean', noise_seed='', train_seed=t) for t in [11, 22, 33]]
    runs += [dict(condition=c, method=m, noise_seed=n, train_seed=t)
        for c in ['mild', 'medium', 'severe'] for m in methods for n in [101, 202, 303] for t in [11, 22, 33]]
    protocol = dict(pilot=False, subset_id='fixture', sample_ids=[str(i) for i in range(43)], runs=runs,
        config=dict(conditions={c: {} for c in ['mild', 'medium', 'severe']}, noise_seeds=[101, 202, 303]))
    score = classification(list(range(43)), list(range(43)))
    summaries, predictions = [], []
    for run in runs:
        summaries.append(dict(run_id=config_id(run), **run, n_images=43,
            **{k: score[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']},
            psnr_mean=20, ssim_mean=.8, latency_ms=1))
        for label in range(43):
            predictions.append(dict(run_id=config_id(run), **run, sample_id=str(label),
                class_id=label, prediction=label, group_id='g' + str(label)))
    if duplicate_sample:
        predictions[42]['sample_id'] = '0'
    if wrong_metric:
        summaries[0]['macro_f1'] = .5

    def serialized(rows):
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('protocol.json', json.dumps(protocol))
        archive.writestr('summary.csv', serialized(summaries))
        archive.writestr('predictions.csv', serialized(predictions))
    buffer.seek(0)
    return buffer


class VerifiedResultsTests(unittest.TestCase):
    def test_complete_matrix_reproduces_metrics_and_coverage(self):
        with zipfile.ZipFile(fixture()) as archive:
            _, _, aggregate, audit = audit_scope(archive, 'protocol.json', 'summary.csv', 'predictions.csv')
        self.assertEqual(audit['prediction_rows'], 219 * 43)
        self.assertEqual(audit['runs'], 219)
        self.assertEqual(len(aggregate), 25)

    def test_duplicate_sample_is_rejected(self):
        with zipfile.ZipFile(fixture(duplicate_sample=True)) as archive:
            with self.assertRaisesRegex(ValueError, 'duplicate sample'):
                audit_scope(archive, 'protocol.json', 'summary.csv', 'predictions.csv')

    def test_summary_metric_must_match_raw_predictions(self):
        with zipfile.ZipFile(fixture(wrong_metric=True)) as archive:
            with self.assertRaisesRegex(ValueError, 'Metric mismatch'):
                audit_scope(archive, 'protocol.json', 'summary.csv', 'predictions.csv')
