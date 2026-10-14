"""Check full-test scope, classifier identity and supplementary degradation factors."""
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.common import write_csv, write_json
from scripts.finalize_research import build
from scripts.run_research_evidence_kaggle import degradation_variants


medium = {'gamma_dark': 2.2, 'poisson_peak': 60, 'gaussian_sigma': 0.01}
variants = degradation_variants(medium)
assert variants['gamma_both'] == medium and medium['poisson_peak'] == 60
assert variants['gamma_only'] == {'gamma_dark': 2.2, 'poisson_peak': None, 'gaussian_sigma': 0}
assert variants['gamma_gaussian']['poisson_peak'] is None
assert variants['gamma_poisson']['gaussian_sigma'] == 0
with TemporaryDirectory() as folder:
    output = Path(folder)
    methods = ['identity', 'gamma', 'he', 'clahe', 'retinex', 'zero_dce', 'retinexformer', 'diffusion']
    config = json.loads(Path('configs/resnet18.json').read_text())
    selection = {'sample_ids': list(range(1290)), 'methods': methods, 'chosen_method': 'gamma',
                 'enhancers': config['enhancers'], 'scores': [{'method': m, 'macro_f1': 0.5, 'latency_ms': 1} for m in methods]}
    protocol = {'pilot': False, 'runs': [{}] * 219, 'subset_id': 'official_test', 'sample_ids': list(range(12630)),
                'protocol': 'A', 'config': config, 'selection': selection}
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'runs.json', [{'status': 'completed'}] * 219)
    (output / 'report.md').write_text('Synthetic test fixture, not experimental results')
    entries = [{'condition': c, 'method': m, 'accuracy_mean': 0.5, 'macro_f1_mean': 0.5,
                'weighted_f1_mean': 0.5, 'psnr_mean_mean': 20, 'ssim_mean_mean': 0.5}
               for c in ['mild', 'medium', 'severe'] for m in methods]
    write_csv(output / 'aggregate.csv', [dict(entries[0], condition='clean', method='clean'), *entries])
    write_json(output / 'bootstrap.json', [])
    figures = output / 'figures'
    figures.mkdir()
    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
    for name in ['macro_f1_levels.png', 'scatter_medium_ssim_mean_macro_f1.png']:
        (figures / name).write_bytes(png)
    build(output)
    text = (output / 'slides.html').read_text()
    discussion = (output / 'discussion.md').read_text()
    assert 'resnet18' in text and '12,630' in text
    assert 'test 430 ảnh' not in text and 'Conv 32, Conv 64' not in text
    assert 'toàn bộ official test' in discussion and '1290' in discussion
    states = json.loads((output / 'runs.json').read_text())
    states[-1]['status'] = 'failed'
    write_json(output / 'runs.json', states)
    try:
        build(output)
    except ValueError as error:
        assert 'Finish all evaluation runs' in str(error)
    else:
        raise AssertionError('Finalized incomplete evaluation')
print('PASS: actual full-test scope, ResNet metadata, incomplete-run rejection and degradation ablation factors')
