"""Official GTSRB annotations, track split, canonical ROI crops and audit."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw
from .common import digest, read_csv, read_image, stable_seed, write_csv, write_json

BASE_URL = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'
ARCHIVES = {
    'GTSRB_Final_Training_Images.zip': 'f33fd80ac59bff73c82d25ab499e03a3',
    'GTSRB_Final_Test_Images.zip': 'c7e4e6327067d32654124b0fe9e82185',
    'GTSRB_Final_Test_GT.zip': 'fe31e9c9270bbcd7b84b7f21a9d9d9e5',
}


def download(root=Path('.')):
    root = Path(root)
    raw = root / 'data/raw/gtsrb'
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected_md5 in ARCHIVES.items():
        archive = raw / name
        if not archive.exists():
            print(f'Downloading {name}', flush=True)
            temporary = archive.with_suffix('.zip.part')
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(archive)
        with archive.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'md5').hexdigest()
        if actual != expected_md5:
            raise ValueError(f'Archive checksum mismatch: {archive}; move aside and download again')
        target = raw / archive.stem
        if not (target / '.complete').exists():
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as zipped:
                for item in zipped.infolist():
                    if not (target / item.filename).resolve().is_relative_to(target.resolve()):
                        raise ValueError(f'Unsafe archive member: {item.filename}')
                zipped.extractall(target)
            (target / '.complete').write_text(digest(archive))
        records.append({'url': BASE_URL + name, 'archive': name, 'sha256': digest(archive),
                        'md5': actual, 'verified_at': datetime.now(timezone.utc).isoformat(),
                        'downloaded_at': datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat()})
    write_json(root / 'data/manifests/gtsrb_sources.json', records)


def split_tracks(rows, seed=42, fraction=0.2):
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    for label in range(43):
        members = [r for r in rows if r['split'] in {'train', 'validation'} and int(r['class_id']) == label]
        groups = sorted({r['group_id'] for r in members})
        if len(groups) < 2:
            raise ValueError(f'Class {label} needs at least two train tracks')
        rng = np.random.default_rng(stable_seed(seed, label))
        rng.shuffle(groups)
        count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
        validation = set(groups[:count])
        for row in members:
            row['split'] = 'validation' if row['group_id'] in validation else 'train'
    return rows


def quarantine_test_overlap(rows):
    """Keep test intact; exclude entire training tracks with exact test copies."""
    test_hashes = {r['pixel_sha256'] for r in rows if r['split'] == 'test'}
    groups = {r['group_id'] for r in rows if r['split'] != 'test' and r['pixel_sha256'] in test_hashes}
    for row in rows:
        if row['split'] != 'test' and row['group_id'] in groups:
            row['split'] = 'quarantine'
    return sorted(groups)


def crop_box(roi, width, height, padding=0.05, inclusive=True):
    x1, y1, x2, y2 = map(int, roi)
    x2 += int(inclusive)
    y2 += int(inclusive)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or not 0 <= padding <= 1:
        raise ValueError(f'Invalid ROI {roi} for {width}x{height}')
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
            min(width, math.ceil(x2+dx)), min(height, math.ceil(y2+dy))]


def prepare(root, config):
    root = Path(root)
    manifest = root / 'data/manifests/gtsrb_samples.csv'
    if manifest.exists():
        raise FileExistsError('Manifest already exists; use check-data, or a different --root for a new protocol')
    raw = root / 'data/raw/gtsrb'
    train_csv = sorted(raw.glob('GTSRB_Final_Training_Images/**/GT-*.csv'))
    test_csv = list(raw.glob('GTSRB_Final_Test_GT/**/GT-final_test.csv'))
    if len(train_csv) != 43 or len(test_csv) != 1:
        raise ValueError('Expected 43 training annotations and one official test annotation; run download')
    test_folder = raw / 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'
    rows = []
    seen_paths = set()
    for annotation in train_csv + test_csv:
        test = annotation == test_csv[0]
        with annotation.open(newline='') as stream:
            annotations = list(csv.DictReader(stream, delimiter=';'))
        for item in annotations:
            filename = item['Filename']
            if Path(filename).name != filename:
                raise ValueError('Annotation filename must be a basename')
            source = (test_folder if test else annotation.parent) / filename
            label = int(item['ClassId'])
            if not 0 <= label < 43 or (not test and int(annotation.parent.name) != label):
                raise ValueError(f'Invalid label: {source}')
            track_match = re.fullmatch(r'(\d{5})_(\d{5})\.ppm', filename)
            if not test and track_match is None:
                raise ValueError(f'Unverified train track filename: {filename}')
            source_id = f'test/{Path(filename).stem}' if test else f'train/{label:05d}/{Path(filename).stem}'
            with Image.open(source) as image:
                image.load()
                width, height = image.size
                pixels_sha = hashlib.sha256(image.convert('RGB').tobytes() + str(image.size).encode()).hexdigest()
            if (width, height) != (int(item['Width']), int(item['Height'])):
                raise ValueError(f'Annotation/image dimensions disagree: {source}')
            box = crop_box([item[k] for k in ['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']], width, height,
                           config['crop']['padding'], config['crop']['roi_end_inclusive'])
            if source in seen_paths:
                raise ValueError(f'Duplicate annotation: {source}')
            seen_paths.add(source)
            rows.append({'sample_id': 'gtsrb/' + source_id, 'dataset': 'gtsrb',
                         'source_path': source.relative_to(root).as_posix(), 'source_id': source_id,
                         'group_id': source_id if test else f'train/{label:05d}/{track_match[1]}',
                         'bbox': json.dumps(box), 'original_label': label, 'class_id': label,
                         'split': 'test' if test else 'train', 'width': width, 'height': height,
                         'source_sha256': digest(source), 'pixel_sha256': pixels_sha,
                         'crop_path': f'data/processed/gtsrb-final/{source_id}.png'})
    all_images = set(raw.glob('GTSRB_Final_Training_Images/**/*.ppm')) | set(test_folder.glob('*.ppm'))
    if seen_paths != all_images:
        raise ValueError(f'Image/annotation coverage mismatch: {len(seen_paths)} vs {len(all_images)}')
    quarantined = quarantine_test_overlap(rows)
    write_json(root / 'data/manifests/quarantine.json', {'reason': 'Training track contains exact pixel copies of official test images; official test kept intact',
               'group_ids': quarantined, 'sample_ids': [r['sample_id'] for r in rows if r['split'] == 'quarantine']})
    split_tracks(rows, config['split_seed'], config['validation_fraction'])
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[row['pixel_sha256']].append(row['sample_id'])
    duplicate_groups = [v for v in duplicates.values() if len(v) > 1]
    write_json(root / 'data/manifests/duplicate_audit.json', duplicate_groups)
    for i, row in enumerate(rows):
        with Image.open(root / row['source_path']) as image:
            crop = image.convert('RGB').crop(json.loads(row['bbox'])).resize((64, 64), Image.Resampling.BILINEAR)
            target = root / row['crop_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
        if i % 10000 == 0:
            print(f'Prepared {i}/{len(rows)} crops', flush=True)
    write_csv(manifest, rows)
    write_json(manifest.with_suffix('.json'), {'sha256': digest(manifest), 'crop': config['crop'],
               'bbox_convention': 'padded xyxy, exclusive right/bottom; original ROI inclusive by config',
               'training_archive': 'GTSRB_Final_Training_Images.zip',
               'split_seed': config['split_seed'], 'validation_fraction': config['validation_fraction'],
               'test_group_limitation': 'Official test has no verified tracks; source_id used'})
    audit(root)
    gallery(root, rows)
    return rows


def audit(root):
    root = Path(root)
    manifest = root / 'data/manifests/gtsrb_samples.csv'
    meta = json.loads(manifest.with_suffix('.json').read_text())
    if digest(manifest) != meta['sha256']:
        raise ValueError('Locked manifest was modified')
    rows = read_csv(manifest)
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample IDs')
    split_sets = {}
    counts = Counter()
    near_hashes, image_stats = defaultdict(list), []
    for split in ['train', 'validation', 'test']:
        subset = [r for r in rows if r['split'] == split]
        if {int(r['class_id']) for r in subset} != set(range(43)):
            raise ValueError(f'Missing/invalid classes in {split}')
        split_sets[split] = {r['group_id'] for r in subset}
    if any(split_sets[a] & split_sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Track leakage across splits')
    sources, hashes = {}, defaultdict(set)
    for row in rows:
        if row['source_id'] in sources:
            raise ValueError('Source duplicated in manifest')
        sources[row['source_id']] = row['split']
        source = root / row['source_path']
        if digest(source) != row['source_sha256']:
            raise ValueError(f'Source changed: {source}')
        x1, y1, x2, y2 = json.loads(row['bbox'])
        if not (0 <= x1 < x2 <= int(row['width']) and 0 <= y1 < y2 <= int(row['height'])):
            raise ValueError('Invalid crop box')
        crop = read_image(root / row['crop_path'])
        if crop.shape != (64, 64, 3):
            raise ValueError('Canonical image must be RGB 64x64')
        with Image.open(source) as original:
            small = np.asarray(original.convert('L').resize((9,8), Image.Resampling.BILINEAR))
        near_hash = np.packbits(small[:,1:] > small[:,:-1]).tobytes().hex()
        near_hashes[near_hash].append({'sample_id': row['sample_id'], 'split': row['split'], 'group_id': row['group_id']})
        image_stats.append({'sample_id': row['sample_id'], 'split': row['split'], 'class_id': row['class_id'],
                            'width': row['width'], 'height': row['height'], 'mean_rgb': float(crop.mean())})
        if row['split'] != 'quarantine':
            hashes[row['pixel_sha256']].add(row['split'])
        counts[(row['split'], int(row['class_id']))] += 1
    leaks = [key for key, value in hashes.items() if len(value) > 1]
    near_candidates = [v for v in near_hashes.values() if len({r['split'] for r in v}) > 1]
    write_json(root / 'data/manifests/near_duplicate_candidates.json', near_candidates)
    write_csv(root / 'data/manifests/image_statistics.csv', image_stats)
    write_json(root / 'data/manifests/audit.json', {'n_images': len(rows), 'n_quarantined': sum(r['split']=='quarantine' for r in rows), 'cross_split_duplicate_hashes': leaks,
               'n_tracks': {s: len(v) for s, v in split_sets.items()},
               'status': 'failed' if leaks else 'completed',
               'visual_roi_review': 'pending; inspect roi_gallery.jpg',
               'near_duplicates': 'Equal 64-bit dHash cross-split candidates saved for visual review; not a proof of duplication',
               'near_duplicate_candidate_groups': len(near_candidates)})
    write_csv(root / 'data/manifests/class_distribution.csv',
              [{'split': s, 'class_id': c, 'count': n} for (s, c), n in sorted(counts.items())])
    if leaks:
        raise ValueError('Exact pixel duplicates cross splits; see audit.json; do not silently remove test images')
    return rows


def gallery(root, rows):
    # Fixed selection: first train image of each class, then seven darkest remaining train crops.
    train = [r for r in rows if r['split'] == 'train']
    selected = [next(r for r in train if int(r['class_id']) == c) for c in range(43)]
    known = {r['sample_id'] for r in selected}
    selected += sorted([r for r in train if r['sample_id'] not in known],
                       key=lambda r: float(read_image(root / r['crop_path']).mean()))[:7]
    sheet = Image.new('RGB', (10 * 144, 5 * 166), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        with Image.open(root / row['source_path']) as original:
            original = original.convert('RGB')
            ImageDraw.Draw(original).rectangle(json.loads(row['bbox']), outline='red', width=1)
            scale = min(138/original.width, 138/original.height)
            original = original.resize((round(original.width*scale), round(original.height*scale)), Image.Resampling.NEAREST)
            x, y = (i % 10) * 144, (i // 10) * 166
            sheet.paste(original, (x, y))
            draw.text((x, y+140), f"class {row['class_id']}", fill='black')
    folder = root / 'data/manifests'
    sheet.save(folder / 'roi_gallery.jpg')
    write_json(folder / 'gallery_ids.json', [r['sample_id'] for r in selected])


def subset_rows(rows, split, per_class=None, seed=42):
    rows = [r for r in rows if r['split'] == split]
    if per_class is None:
        return rows
    if per_class < 1:
        raise ValueError('per_class must be positive')
    selected = []
    for label in range(43):
        members = [r for r in rows if int(r['class_id']) == label]
        # Select whole groups, possibly exceeding the requested image budget.
        groups = sorted({r['group_id'] for r in members}, key=lambda g: stable_seed(seed, g))
        chosen = []
        for group in groups:
            chosen.extend(r for r in members if r['group_id'] == group)
            if len(chosen) >= per_class:
                break
        selected.extend(chosen)
    if not selected:
        raise ValueError(f'Empty {split} split')
    return selected
