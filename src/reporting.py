"""Generate tables and figures from completed paired runs, never invented scores."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import pearsonr, spearmanr

from .common import config_id, read_csv, read_image, write_csv, write_json
from .metrics import paired_bootstrap, paired_rows


def report(output, root=Path('.')):
    output, root = Path(output), Path(root)
    protocol = json.loads((output / 'protocol.json').read_text())
    states = json.loads((output / 'runs.json').read_text())
    if len(states) != len(protocol['runs']) or any(r['status'] != 'completed' for r in states):
        raise ValueError('Incomplete run matrix; finish failed/planned runs before reporting')
    rows = read_csv(output / 'summary.csv')
    if {r['run_id'] for r in rows} != {config_id(r) for r in protocol['runs']} or len(rows) != len(states):
        raise ValueError('Summary does not match frozen run matrix')
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    expected_ids = set(protocol['sample_ids'])
    predictions = {}
    for row in rows:
        preds = read_csv(output / 'predictions' / f"{row['run_id']}.csv")
        if len(preds) != len(expected_ids) or {p['sample_id'] for p in preds} != expected_ids:
            raise ValueError('Run sample coverage differs from frozen subset')
        predictions[row['run_id']] = preds
    baselines = {(r['condition'], r['train_seed'], r['noise_seed']): r for r in rows if r['method'] == 'identity'}
    deltas = []
    for row in rows:
        if row['method'] == 'clean':
            continue
        base = baselines[(row['condition'], row['train_seed'], row['noise_seed'])]
        paired_rows(predictions[base['run_id']], predictions[row['run_id']])
        deltas.append({**row, **{'delta_'+k+'_pp': 100*(float(row[k])-float(base[k])) for k in ['accuracy', 'macro_f1']},
                       **{'delta_'+k: float(row[k])-float(base[k]) for k in ['psnr_mean', 'ssim_mean']}})
    write_csv(output / 'deltas.csv', deltas)
    metrics = ['accuracy', 'macro_f1', 'psnr_mean', 'ssim_mean', 'latency_ms']
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['condition'], row['method'], row['train_seed'])].append(row)
    per_training_seed = []
    for (condition, method, seed), members in grouped.items():
        per_training_seed.append({'condition': condition, 'method': method, 'train_seed': seed,
                                  **{k: float(np.mean([float(r[k]) for r in members])) for k in metrics}})
    write_csv(output / 'noise_averaged.csv', per_training_seed)
    # Preserve all seed combinations in summary.csv; this second table describes noise sensitivity.
    sensitivity = []
    for (condition, method, seed), members in grouped.items():
        sensitivity.append({'condition': condition, 'method': method, 'train_seed': seed,
                            **{k+'_noise_std': float(np.std([float(r[k]) for r in members], ddof=1)) if len(members)>1 else 0.0
                               for k in ['accuracy', 'macro_f1']}})
    write_csv(output / 'noise_sensitivity.csv', sensitivity)
    aggregate = []
    for condition, method in sorted({(r['condition'], r['method']) for r in rows}):
        members = [r for r in per_training_seed if r['condition'] == condition and r['method'] == method]
        entry = {'condition': condition, 'method': method, 'n_training_seeds': len(members)}
        for key in metrics:
            values = [r[key] for r in members]
            entry[key+'_mean'] = float(np.mean(values))
            entry[key+'_std'] = 0.0 if len(members) == 1 or all(np.isinf(v) for v in values) else float(np.std(values, ddof=1))
        aggregate.append(entry)
    write_csv(output / 'aggregate.csv', aggregate)
    correlations = []
    for condition in protocol['config']['conditions']:
        points = [r for r in aggregate if r['condition'] == condition]
        for quality in ['psnr_mean', 'ssim_mean']:
            for recognition in ['accuracy', 'macro_f1']:
                x, y = np.array([r[quality+'_mean'] for r in points]), np.array([r[recognition+'_mean'] for r in points])
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                defined = len(x) >= 3 and np.ptp(x)>0 and np.ptp(y)>0
                correlations.append({'condition': condition, 'quality': quality, 'recognition': recognition,
                                     'n': len(x), 'pearson': float(pearsonr(x,y).statistic) if defined else '',
                                     'spearman': float(spearmanr(x,y).statistic) if defined else '',
                                     'status': 'exploratory; dependent methods' if defined else 'undefined: constant or too few points'})
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(x, 100*y)
                for row in points:
                    if np.isfinite(row[quality+'_mean']):
                        ax.annotate(row['method'], (row[quality+'_mean'], 100*row[recognition+'_mean']), fontsize=8)
                ax.set(xlabel=quality, ylabel=recognition+' (%)', title=f'{condition}; method means, n={len(x)}')
                fig.tight_layout()
                fig.savefig(figures / f'scatter_{condition}_{quality}_{recognition}.png', dpi=140)
                plt.close(fig)
    write_csv(output / 'correlations.csv', correlations)
    for metric in ['accuracy', 'macro_f1']:
        fig, ax = plt.subplots(figsize=(9, 5))
        levels = list(protocol['config']['conditions'])
        for method in protocol['selection']['methods']:
            points = [next(r for r in aggregate if r['condition']==c and r['method']==method) for c in levels]
            ax.errorbar(levels, [100*r[metric+'_mean'] for r in points],
                        yerr=[100*r[metric+'_std'] for r in points], marker='o', label=method)
        ax.set(ylabel=metric+' (%)', title='Noise averaged first; error bars = training seed SD')
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f'{metric}_levels.png', dpi=140)
        plt.close(fig)
    bootstrap, failures = [], []
    selected = protocol['selection']['chosen_method']
    for condition in protocol['config']['conditions']:
        for checkpoint in protocol['checkpoints']:
            seed = str(checkpoint['seed'])
            a, b = [], []
            for row in rows:
                if row['condition'] == condition and row['train_seed'] == seed:
                    if row['method'] == 'identity':
                        a.extend(predictions[row['run_id']])
                    if row['method'] == selected:
                        b.extend(predictions[row['run_id']])
            bootstrap.append({'condition': condition, 'train_seed': seed, 'method': selected,
                              **paired_bootstrap(a,b,protocol['config']['bootstrap_repetitions'])})
            for before, after in paired_rows(a,b):
                old = before['prediction'] == before['class_id']
                new = after['prediction'] == after['class_id']
                category = 'helped' if new and not old else 'harmed' if old and not new else 'still_wrong' if not new else 'still_correct'
                failures.append({'condition': condition, 'train_seed': seed, 'category': category,
                                 'sample_id': after['sample_id'], 'noise_seed': after['noise_seed'],
                                 'class_id': after['class_id'], 'before': before['prediction'], 'after': after['prediction'],
                                 'before_path': before['output_path'], 'after_path': after['output_path']})
    write_json(output / 'bootstrap.json', bootstrap)
    write_csv(output / 'cases.csv', failures)
    # First stable sample ID per category; never cherry-pick visually attractive examples.
    examples = []
    for category in ['helped', 'harmed', 'still_wrong']:
        members = sorted([r for r in failures if r['category']==category], key=lambda r: (r['sample_id'], r['condition'], r['noise_seed']))
        examples.extend(members[:5])
    write_json(output / 'example_ids.json', examples)
    if examples:
        sheet = Image.new('RGB', (400, 150*len(examples)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, case in enumerate(examples):
            for j, key in enumerate(['before_path', 'after_path']):
                image = read_image(root / case[key])
                sheet.paste(Image.fromarray(np.rint(image*255).astype(np.uint8)).resize((112,112)), (j*140,i*150))
            draw.text((0,i*150+113), f"{case['category']}: {case['before']} -> {case['after']} (true {case['class_id']})", fill='black')
        sheet.save(figures / 'failure_examples.png')
    distribution = root / 'data/manifests/class_distribution.csv'
    if distribution.exists():
        counts = read_csv(distribution)
        fig, ax = plt.subplots(figsize=(12,4))
        for split in ['train','validation','test']:
            members = [r for r in counts if r['split']==split]
            ax.plot([int(r['class_id']) for r in members], [int(r['count']) for r in members], label=split)
        ax.set(xlabel='Class ID', ylabel='Images', title='GTSRB class distribution')
        ax.legend(); fig.tight_layout(); fig.savefig(figures/'class_distribution.png', dpi=140); plt.close(fig)
    for checkpoint in protocol['checkpoints']:
        learning = Path(checkpoint['path']).parent / f"learning_{checkpoint['seed']}.csv"
        if learning.exists():
            history = read_csv(learning)
            fig, axes = plt.subplots(1,3,figsize=(13,4))
            for ax, metric in zip(axes, ['loss','accuracy','macro_f1']):
                for split in ['train','validation']:
                    members = [r for r in history if r['split']==split]
                    ax.plot([int(r['epoch']) for r in members], [float(r[metric]) for r in members], label=split)
                ax.set(xlabel='Epoch', ylabel=metric); ax.legend()
            fig.tight_layout(); fig.savefig(figures/f"learning_{checkpoint['seed']}.png", dpi=140); plt.close(fig)
    for row in rows:
        if row['method'] in {'clean', selected} and row['noise_seed'] in {'', str(protocol['config']['noise_seeds'][0])}:
            score = json.loads((output / 'classes' / f"{row['run_id']}.json").read_text())
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.imshow(score['confusion_matrix'], cmap='Blues')
            ax.set(xlabel='Predicted class', ylabel='True class', title=f"{row['method']} {row['condition']} seed {row['train_seed']}")
            fig.tight_layout(); fig.savefig(figures / f"confusion_{row['run_id']}.png", dpi=130); plt.close(fig)
    lines = ['# GTSRB: báo cáo kết quả thực chạy', '',
             '**PILOT — không phải kết quả đánh giá cuối.**' if protocol['pilot'] else 'Giao thức A: CNN khóa trọng số.', '',
             f"Số mẫu mỗi điều kiện: {len(expected_ids)}; subset: `{protocol['subset_id']}`.",
             f"Pipeline chọn trước test bằng validation: **{selected}**.", '',
             'Noise seed được lấy trung bình trong mỗi training seed, sau đó báo mean ± sample SD qua training seed.', '',
             '| Mức | Phương pháp | Accuracy (%) | Macro-F1 (%) | PSNR (dB) | SSIM | ms/ảnh |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['condition']} | {r['method']} | {100*r['accuracy_mean']:.2f} ± {100*r['accuracy_std']:.2f} | {100*r['macro_f1_mean']:.2f} ± {100*r['macro_f1_std']:.2f} | {r['psnr_mean_mean']:.2f} | {r['ssim_mean_mean']:.4f} | {r['latency_ms_mean']:.2f} |")
    lines += ['', '## Phân tích và giới hạn', '',
              '- `deltas.csv`: chênh lệch so identity cùng mức và seed; nhận diện tính bằng điểm phần trăm.',
              '- `correlations.csv`: Pearson/Spearman theo mức; loại clean; ít điểm và các phương pháp phụ thuộc nhau.',
              '- `bootstrap.json`: paired bootstrap theo group/source, có điều kiện trên checkpoint.',
              '- `cases.csv`, `example_ids.json`: mẫu được giúp, làm hỏng, hoặc vẫn nhận sai; ID lựa chọn cố định.',
              '- Đây là thiếu sáng mô phỏng trên crop GTSRB; không suy thành hiệu năng camera ban đêm.',
              '- PSNR clean là vô hạn; không thay bằng số hữu hạn. Một training seed không ước lượng được biến thiên huấn luyện.', '',
              '![Macro-F1](figures/macro_f1_levels.png)', '', '![Accuracy](figures/accuracy_levels.png)']
    (output / 'report.md').write_text('\n'.join(lines)+'\n')
    return output / 'report.md'
