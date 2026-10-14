"""Run: python -m checks.check_full_test_staging."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from scripts.run_resnet18_full_test_kaggle import copy_previous_results

with TemporaryDirectory() as folder:
    root = Path(folder)
    inputs = root / 'inputs'
    inputs.mkdir()
    with ZipFile(inputs / 'output.zip', 'w') as archive:
        prefix = 'gtsrb-resnet18/outputs/resnet18_A/'
        archive.writestr(prefix + 'validation/selection.json', '{"locked": true}')
        archive.writestr(prefix + 'train/cnn_11.pt', b'checkpoint')
        archive.writestr('gtsrb-resnet18/data/cache/unused.npy', b'large cache')
    destination = root / 'results'
    copy_previous_results(inputs, destination)
    assert (destination / 'train/cnn_11.pt').read_bytes() == b'checkpoint'
    assert (destination / 'validation/selection.json').read_text() == '{"locked": true}'
    assert len(list(destination.rglob('*.*'))) == 2
    with ZipFile(inputs / 'output.zip', 'w') as archive:
        archive.writestr(prefix + 'validation/selection.json', '{}')
        archive.writestr(prefix + 'train/../../escape.txt', 'unsafe')
    try:
        copy_previous_results(inputs, root / 'unsafe')
    except ValueError as error:
        assert 'Unsafe' in str(error)
    else:
        raise AssertionError('Accepted unsafe archive path')
print('PASS: selective checkpoint/selection extraction and traversal rejection')
