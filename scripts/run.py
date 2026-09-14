"""Run all commands as python -m scripts.run COMMAND from the project root."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.common import device_for, environment, read_csv, read_image, save_array, synchronize, write_json
from src.data import audit, download, prepare, subset_rows
from src.degradation import degrade
from src.enhancement import Enhancer, METHODS, validation_grid
from src.experiment import evaluate, freeze, tune
from src.training import train


def smoke(root, config, device, output, count=43):
    root, output = Path(root), Path(output)
    if count < 1:
        raise ValueError('count must be positive')
    rows = read_csv(root / 'data/manifests/gtsrb_samples.csv')
    validation = [r for r in rows if r['split'] == 'validation']
    selected = [next(r for r in validation if int(r['class_id']) == c) for c in range(43)]
    selected += [r for r in validation if r not in selected]
    selected = selected[:count]
    output.mkdir(parents=True, exist_ok=True)
    status = {}
    for method in METHODS:
        try:
            enhancer = Enhancer(method, config['enhancers'][method], device)
            inputs = [degrade(read_image(root/r['crop_path']), **config['conditions']['medium'],
                              sample_id=r['sample_id'], noise_seed=config['noise_seeds'][0], condition='medium') for r in selected]
            enhancer(inputs[0], selected[0]['sample_id'])
            synchronize(device)
            start = time.perf_counter()
            results = [enhancer(x, r['sample_id']) for x, r in zip(inputs, selected)]
            synchronize(device)
            latency = 1000*(time.perf_counter()-start)/len(selected)
            for row, result in zip(selected, results):
                save_array(output/method/(row['sample_id'].replace('/', '_')+'.npy'), result)
            status[method] = {'status': 'completed', 'n_images': len(results), 'latency_ms': latency,
                              'metadata': enhancer.metadata}
        except Exception as error:
            status[method] = {'status': 'failed', 'error': str(error)}
        write_json(output/'smoke.json', {'environment': environment(), 'device': str(device), 'results': status,
                   'sample_ids': [r['sample_id'] for r in selected], 'split': 'validation', 'condition': 'medium'})
        print(method, status[method]['status'], flush=True)
    columns = ['clean', 'mild', 'medium', 'severe', *METHODS[1:]]
    sheet = Image.new('RGB', (len(columns)*112, min(5,len(selected))*132), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected[:5]):
        reference = read_image(root/row['crop_path'])
        for j, method in enumerate(columns):
            if method == 'clean':
                image = reference
            elif method in config['conditions']:
                image = degrade(reference, **config['conditions'][method], sample_id=row['sample_id'],
                                noise_seed=config['noise_seeds'][0], condition=method)
            elif status[method]['status'] == 'completed':
                image = read_image(output/method/(row['sample_id'].replace('/', '_')+'.npy'))
            else:
                draw.text((j*112, i*132+30), 'FAILED', fill='red')
                continue
            sheet.paste(Image.fromarray(np.rint(image*255).astype(np.uint8)).resize((104,104)), (j*112,i*132))
            draw.text((j*112,i*132+108), method, fill='black')
    sheet.save(output/'gallery.png')
    if any(v['status'] == 'failed' for v in status.values()):
        raise RuntimeError(f'Pretrained smoke failed; see {output}/smoke.json')


def main():
    parser = argparse.ArgumentParser(description='GTSRB low-light pipeline; RGB 64x64, fixed CNN protocol A')
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--device', default='auto', choices=['auto','cpu','mps','cuda'])
    parser.add_argument('--threads', type=int, default=2)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('download')
    sub.add_parser('prepare')
    sub.add_parser('check-data')
    sub.add_parser('check')
    p = sub.add_parser('smoke')
    p.add_argument('--count', type=int, default=43)
    p.add_argument('--output', default='outputs/smoke')
    p = sub.add_parser('budget')
    p.add_argument('--per-class', type=int)
    p.add_argument('--smoke-log', default='outputs/smoke/smoke.json')
    p.add_argument('--validation-per-class', type=int)
    p = sub.add_parser('train')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--epochs', type=int)
    p.add_argument('--per-class', type=int)
    p.add_argument('--output', default='outputs/train')
    p = sub.add_parser('tune')
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--per-class', type=int)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--output', default='outputs/validation')
    p = sub.add_parser('freeze')
    p.add_argument('--selection', default='outputs/validation/selection.json')
    p.add_argument('--per-class', type=int)
    p.add_argument('--subset-seed', type=int, default=42)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--output', default='outputs/protocol.json')
    p = sub.add_parser('evaluate')
    p.add_argument('--protocol', default='outputs/protocol.json')
    p.add_argument('--output', default='outputs/test')
    p = sub.add_parser('report')
    p.add_argument('--output', default='outputs/test')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if config['crop']['size'] != 64 or config['crop']['interpolation'] != 'bilinear':
        parser.error('This implementation locks crops to RGB 64x64 bilinear')
    device = device_for(args.device)
    if args.command == 'download':
        download(args.root)
    elif args.command == 'prepare':
        prepare(args.root,config)
    elif args.command == 'check-data':
        audit(args.root)
        print('Data checks passed')
    elif args.command == 'check':
        from checks.check_pipeline import main as check
        check()
    elif args.command == 'smoke':
        smoke(args.root,config,device,args.output,args.count)
    elif args.command == 'budget':
        import shutil
        rows = read_csv(args.root / 'data/manifests/gtsrb_samples.csv')
        n_test = len(subset_rows(rows, 'test', args.per_class))
        n_validation = len(subset_rows(rows, 'validation', args.validation_per_class))
        n_conditions = len(config['conditions'])*len(config['noise_seeds'])
        candidates = sum(len(validation_grid(m, config['enhancers'][m])) for m in METHODS)
        estimated = (n_test*(1+n_conditions*len(METHODS)) + n_validation*n_conditions*candidates)*(64*64*3*4+128)
        result = {'n_test': n_test, 'n_validation': n_validation,
                  'cnn_condition_runs': len(config['train_seeds'])*(1+n_conditions*len(METHODS)),
                  'estimated_float_cache_gib': estimated/2**30,
                  'free_disk_gib': shutil.disk_usage(args.root).free/2**30,
                  'note': 'Cache estimate excludes filesystem overhead, raw data and predictions; test and validation subsets are separately specified'}
        smoke_log = Path(args.smoke_log)
        if smoke_log.exists():
            measurements = json.loads(smoke_log.read_text())['results']
            if all(measurements.get(m,{}).get('status')=='completed' for m in METHODS):
                result['test_enhancer_compute_hours'] = n_test*n_conditions*sum(measurements[m]['latency_ms'] for m in METHODS)/3_600_000
                result['timing_note'] = 'Pilot device, no CNN/metrics/I/O; estimate only'
        print(json.dumps(result,indent=2))
    elif args.command == 'train':
        for seed in args.seeds or config['train_seeds']:
            train(args.root,config,seed,device,args.output,args.per_class,args.epochs)
    elif args.command == 'tune':
        tune(args.root,config,args.checkpoints,device,args.output,args.per_class,args.methods)
    elif args.command == 'freeze':
        freeze(args.root,args.selection,args.output,args.per_class,args.subset_seed,args.pilot)
    elif args.command == 'evaluate':
        evaluate(args.root,args.protocol,args.output,device)
    elif args.command == 'report':
        from src.reporting import report
        print(report(args.output, args.root))


if __name__ == '__main__':
    main()
