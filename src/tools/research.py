"""Run from the repository root: python -m src.tools.research --help."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('download')
    commands.add_parser('prepare')
    commands.add_parser('audit')
    for name in ['train', 'tune', 'freeze', 'evaluate', 'report']:
        p = commands.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
        if name not in ['report', 'freeze']:
            p.add_argument('--device', default='auto')
        if name in ['train', 'tune']:
            p.add_argument('--config', type=Path, default=Path('configs/resnet18.json'))
        if name == 'train':
            p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
        if name == 'tune':
            p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
            p.add_argument('--per-class', type=int, default=30)
        if name == 'freeze':
            p.add_argument('--selection', type=Path, required=True)
            p.add_argument('--per-class', type=int, help='Omit to freeze all 12,630 official test images')
        if name == 'evaluate':
            p.add_argument('--protocol', type=Path, required=True)
    p = commands.add_parser('errors')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('bundle')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command in ['errors', 'bundle']:
        from .research_evidence import analyze, bundle
        result = (analyze if args.command == 'errors' else bundle)(args.input, args.output)
    elif args.command in ['download', 'prepare', 'audit']:
        from src import data
        cfg = json.loads((args.root / 'configs/resnet18.json').read_text(encoding='utf-8'))
        result = data.prepare(args.root, cfg) if args.command == 'prepare' else getattr(data, args.command)(args.root)
    elif args.command == 'report':
        from src.reporting import report
        result = report(args.output, args.root)
    else:
        from src.common import device_for
        from src.experiment import tune, freeze, evaluate
        if args.command == 'train':
            from src.training import train
            cfg = json.loads(args.config.read_text(encoding='utf-8'))
            result = [str(train(args.root, cfg, seed, device_for(args.device), args.output)) for seed in args.seeds]
        elif args.command == 'tune':
            cfg = json.loads(args.config.read_text(encoding='utf-8'))
            result = tune(args.root, cfg, args.checkpoints, device_for(args.device), args.output, args.per_class)
        elif args.command == 'freeze':
            result = freeze(args.root, args.selection, args.output, args.per_class)
        else:
            result = evaluate(args.root, args.protocol, args.output, device_for(args.device))
    if result is not None:
        print(str(result) if not isinstance(result, (list, dict)) else f'Completed {args.command}')


if __name__ == '__main__':
    main()
