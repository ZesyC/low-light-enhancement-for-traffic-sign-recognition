"""Wait for the submitted Kaggle job, download its archive and verify all outputs."""
import json
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

from src.common import digest, read_csv, write_json
from scripts.verify_results import verify


KERNEL = 'sangtranminh211/gtsrb-resnet18-mixed-lowlight-20261012'


def extract(archive, output):
    with ZipFile(archive) as z:
        for member in z.infolist():
            p = Path(member.filename)
            if p.is_absolute() or '..' in p.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe result archive member')
        if z.testzip() is not None:
            raise ValueError('Result archive failed CRC verification')
        if any((output / member.filename).exists() for member in z.infolist() if not member.is_dir()):
            raise FileExistsError('Refusing to overwrite existing experiment results')
        z.extractall(output)


def main():
    root = Path.cwd()
    folder = root / 'outputs/resnet18_lowlight_kaggle'
    metadata = json.loads((folder / 'run.json').read_text())
    while True:
        status = subprocess.run(['kaggle', 'kernels', 'status', KERNEL], check=True,
                                capture_output=True, text=True, timeout=60).stdout.strip()
        logs = subprocess.run(['kaggle', 'kernels', 'logs', KERNEL], check=True,
                              capture_output=True, text=True, timeout=60).stdout
        (folder / 'latest.log').write_text(logs)
        metadata.update(status=status, pilot_completed='PILOT COMPLETE:' in logs,
                        training_completed='TRAIN COMPLETE:' in logs,
                        job_completed='COMPLETE: /kaggle/working/resnet18-lowlight-results.zip' in logs)
        write_json(folder / 'run.json', metadata)
        print(status, 'pilot:', metadata['pilot_completed'], 'train:', metadata['training_completed'], flush=True)
        if 'COMPLETE' in status:
            if not metadata['job_completed']:
                raise RuntimeError('Kaggle finished without the expected result archive; inspect latest.log')
            break
        if any(word in status for word in ['ERROR', 'CANCEL', 'FAIL']):
            raise RuntimeError('Kaggle job failed; saved execution log in latest.log')
        time.sleep(60)
    download = folder / 'download'
    subprocess.run(['kaggle', 'kernels', 'output', KERNEL, '-p', str(download),
                    '--file-pattern', r'^resnet18-lowlight-results\.zip$', '--page-size', '200'], check=True)
    archive = download / 'resnet18-lowlight-results.zip'
    extract(archive, root / 'outputs')
    for name in ['resnet18_lowlight_pilot/test', 'resnet18_lowlight/test_subset', 'resnet18_clean_replay/test_subset']:
        verify(root, root / 'outputs' / name)
    metadata.update(status='complete_and_verified', archive_sha256=digest(archive))
    write_json(folder / 'run.json', metadata)
    table = read_csv(root / 'outputs/resnet18_lowlight/test_subset/aggregate.csv')
    lines = ['# Mixed low-light: kết quả Kaggle đã kiểm chứng', '',
             'Đủ ba seed; hai classifier đánh giá trên cùng 430 ảnh test, 219 lượt mỗi classifier.', '',
             '| Mức (identity) | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) |', '|---|---:|---:|---:|']
    for r in table:
        if r['method'] in {'clean', 'identity'}:
            lines.append('| ' + r['condition'] + ' | ' + ' | '.join(
                f"{100*float(r[k+'_mean']):.2f} ± {100*float(r[k+'_std']):.2f}"
                for k in ['accuracy', 'macro_f1', 'weighted_f1']) + ' |')
    lines += ['', 'Delta paired theo condition/method nằm trong `resnet18_lowlight/paired_aggregate.csv`. '
              'Identity cô lập thay đổi training; các enhancer tune riêng bằng validation.', '',
              'Báo cáo và kiểm chứng: `resnet18_lowlight/test_subset/report.md`, `verification.json`. '
              'Đây là subset test; chưa đánh giá full official test cho mixed.']
    (folder / 'result_summary.md').write_text('\n'.join(lines) + '\n')
    readme = root / 'README.md'
    readme.write_text(readme.read_text().replace('Chưa có kết quả cuối của thí nghiệm mixed.',
        'Thí nghiệm mixed đã hoàn thành và kiểm chứng; xem `outputs/resnet18_lowlight_kaggle/result_summary.md` '
        'và báo cáo riêng trong `outputs/resnet18_lowlight/test_subset/`.'))
    print('DOWNLOADED AND VERIFIED:', archive, flush=True)


if __name__ == '__main__':
    main()
