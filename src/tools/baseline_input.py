"""Read the pinned ZIP from existing public notebook Output; no Dataset or API key."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
import zipfile

import requests

from src.common import digest, write_json

SOURCE = 'kieuthithutrang/gtsrb-low-light-enhancement-for-traffic-signs'
VERSION = 'v1'
ORIGINAL_SHA256 = '1141f3b3be6f0cb59918afa1e685947a9b07aa60e6601c4011386dc88efd1f0b'
VERIFIED_SHA256 = '01ee0fa7c2b42288c644b0c21d200344ff6cffb5c549f96b4a3abc2c917b6091'
MANIFEST_SHA256 = '0ea7537e92c03790e49460f12558e6a268a0d6f310084970baf5d2a61dc7fce5'
OUTPUT_API = 'https://api.kaggle.com/v1/kernels.KernelsApiService/ListKernelSessionOutput'


def download_original(destination):
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != ORIGINAL_SHA256:
            raise ValueError('Existing baseline ZIP checksum mismatch')
        return destination
    owner, slug = SOURCE.split('/')
    response = requests.post(OUTPUT_API, json=dict(userName=owner, kernelSlug=slug,
        versionLabel=VERSION, pageSize=100), timeout=60)
    response.raise_for_status()
    matches = [row for row in response.json().get('files', []) if row.get('fileName') == 'gtsrb-notebook-results.zip']
    if len(matches) != 1:
        raise ValueError('Pinned public notebook Output does not contain exactly one baseline ZIP')
    url = matches[0]['url']
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError('Invalid baseline download URL')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.zip')
    h = hashlib.sha256()
    with requests.get(url, stream=True, timeout=120) as download:
        download.raise_for_status()
        with partial.open('wb') as stream:
            for chunk in download.iter_content(1024 * 1024):
                h.update(chunk)
                stream.write(chunk)
    if h.hexdigest() != ORIGINAL_SHA256:
        partial.unlink()
        raise ValueError('Downloaded baseline ZIP checksum mismatch')
    partial.replace(destination)
    return destination


def normalize_original(original, destination):
    """Repair only the known duplicate supplemental paths in this exact old export."""
    original, destination = Path(original), Path(destination)
    if digest(original) != ORIGINAL_SHA256:
        raise ValueError('Only the pinned original export may be normalized')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.zip')
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        counts = Counter(source.namelist())
        prefix = '07_supplemental/ablation/'
        if {n: c for n, c in counts.items() if c > 1} != {prefix + 'protocol.json': 2, prefix + 'summary.csv': 2}:
            raise ValueError('Unexpected legacy duplicate paths')
        expected = json.loads(source.read('SHA256.json'))
        seen, hashes, moved = Counter(), {}, []
        for info in source.infolist():
            name = info.filename
            if name == 'SHA256.json':
                continue
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name:
                raise ValueError('Unsafe baseline archive path')
            seen[name] += 1
            target = name
            if name in [prefix + 'runs.json', prefix + 'worker_devices.json'] or (
                    name in [prefix + 'protocol.json', prefix + 'summary.csv'] and seen[name] == 2):
                target = name.replace(prefix, '07_supplemental/subset/', 1)
                moved.append({'from': name, 'entry': seen[name], 'to': target})
            if target in hashes:
                raise ValueError('Unresolved duplicate archive path')
            h = hashlib.sha256()
            with source.open(info) as stream, output.open(target, 'w', force_zip64=True) as writer:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
                    writer.write(chunk)
            sha = h.hexdigest()
            if seen[name] == counts[name] and expected.get(name) != sha:
                raise ValueError('Baseline member checksum mismatch: ' + name)
            hashes[target] = sha
        recovery = {'source_archive_sha256': ORIGINAL_SHA256, 'source_preserved': True,
            'moved_entries': moved,
            'execution_result': 'Full-test completed; original notebook failed during final subset ZIP rename across mounts.',
            'changed_predictions': False,
            'legacy_first_duplicate_integrity': 'ZIP CRC verified; legacy hash map recorded only last duplicate.'}
        data = (json.dumps(recovery, indent=2) + '\n').encode()
        name = '04_protocol/export_recovery.json'
        output.writestr(name, data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        canonical = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() != MANIFEST_SHA256:
            raise ValueError('Normalized contents differ from the locally verified baseline')
        output.writestr('SHA256.json', json.dumps(hashes, indent=2))
    partial.replace(destination)
    return destination


def resolve_baseline(folder, archive=None):
    folder = Path(folder)
    if archive is None:
        archive = download_original(folder / 'gtsrb-notebook-results.zip')
    archive = Path(archive)
    sha = digest(archive)
    if sha == VERIFIED_SHA256:
        result = archive
    elif sha == ORIGINAL_SHA256:
        result = normalize_original(archive, folder / 'gtsrb-notebook-results-normalized.zip')
    else:
        raise ValueError('Baseline ZIP is not the pinned original or verified export')
    write_json(folder / 'input_provenance.json', dict(source_notebook=SOURCE, version=VERSION,
        input_archive_sha256=sha, normalized_archive_sha256=digest(result),
        verified_content_manifest_sha256=MANIFEST_SHA256, dataset_created=False, api_key_required=False))
    print('Baseline ZIP ready:', result.name, '| original/verified contents checked | no Dataset or API key', flush=True)
    return result
