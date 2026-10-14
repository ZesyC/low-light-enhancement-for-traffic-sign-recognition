"""Verify supplementary Kaggle runs, summarize ablation, and create an offline demo."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from src.common import digest, read_csv, write_csv, write_json
from src.metrics import classification


def build(root=Path('.')):
    root = Path(root)
    output = root / 'outputs/research_evidence'
    protocol = json.loads((output / 'protocol.json').read_text())
    manifest = root / 'data/manifests/gtsrb_samples.csv'
    assert digest(manifest) == protocol['manifest_sha256']
    manifest_rows = read_csv(manifest)
    labels = {r['sample_id']: int(r['class_id']) for r in manifest_rows if r['split'] == 'test'}
    ids = set(protocol['sample_ids'])
    assert len(ids) == 430 and ids <= labels.keys()
    rows = read_csv(output / 'summary.csv')
    expected = {(v, m, str(n), str(s)) for v in protocol['variants'] for m in protocol['methods']
                for n in protocol['noise_seeds'] for s in protocol['train_seeds']}
    assert len(rows) == 288 and {(r['variant'], r['method'], r['noise_seed'], r['train_seed']) for r in rows} == expected
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        predictions = read_csv(output / 'predictions' / f"{row['run_id']}.csv")
        assert len(predictions) == 430 and {r['sample_id'] for r in predictions} == ids
        assert all(int(r['class_id']) == labels[r['sample_id']] and 0 <= int(r['prediction']) < 43
                   and 0 <= float(r['true_probability']) <= 1 for r in predictions)
        score = classification([int(r['class_id']) for r in predictions], [int(r['prediction']) for r in predictions])
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            assert math.isclose(float(row[metric]), score[metric], abs_tol=1e-12)
            grouped[row['variant'], row['method'], metric][row['train_seed']].append(score[metric])
    aggregate = []
    for variant, method in sorted({(r['variant'], r['method']) for r in rows}):
        result = {'variant': variant, 'method': method, 'n_training_seeds': 3}
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            means = [float(np.mean(v)) for v in grouped[variant, method, metric].values()]
            assert len(means) == 3 and all(len(v) == 3 for v in grouped[variant, method, metric].values())
            result.update({metric+'_mean': float(np.mean(means)), metric+'_std': float(np.std(means, ddof=1))})
        aggregate.append(result)
    write_csv(output / 'aggregate.csv', aggregate)
    points = {(r['variant'], r['method']): r for r in aggregate}
    lines = ['# Ablation suy giảm ở mức vừa', '',
             'Classifier clean ResNet-18 cố định, tham số enhancer chọn trước bằng validation. '
             '288 lượt trên cùng subset test 430 ảnh, ba seed CNN và ba noise seed. '
             'Thí nghiệm phụ này phân tích gamma và từng loại nhiễu, không dùng để đổi hyperparameter.', '',
             '| Phương pháp | Gamma | Gamma + Gaussian | Gamma + Poisson | Gamma + cả hai |',
             '|---|---:|---:|---:|---:|']
    for method in protocol['methods']:
        values = [f"{100*points[v, method]['macro_f1_mean']:.2f} ± {100*points[v, method]['macro_f1_std']:.2f}"
                  for v in protocol['variants']]
        lines.append('| '+method+' | '+' | '.join(values)+' |')
    identity = [points[v, 'identity']['macro_f1_mean'] for v in protocol['variants']]
    lines += ['', 'Đơn vị Macro-F1: %. Noise được trung bình trong mỗi training seed trước, '
              'sau đó tính mean ± sample SD giữa ba seed.', '',
              f'Identity giảm từ {100*identity[0]:.2f}% ở gamma-only xuống {100*identity[-1]:.2f}% '
              'khi thêm cả Poisson và Gaussian. Đây là ảnh hưởng trên mô phỏng, không suy ra camera đêm thực.', '',
              'Cùng condition `medium` và seed được truyền vào hàm suy giảm. Khi tắt Poisson, '
              'chuỗi RNG trước bước Gaussian thay đổi, nên các biến thể không dùng cùng trường Gaussian pixel tuyệt đối.', '',
              'Kết quả supplementary không thay thế bảng full-test 12.630 ảnh. Gamma-only không có nhiễu, '
              'ba noise seed tạo cùng ảnh và không được coi là ba phép đo độc lập.']
    (output / 'ablation_report.md').write_text('\n'.join(lines)+'\n')
    demo = output / 'demo'
    records = read_csv(demo / 'predictions.csv')
    validation = {r['sample_id']: int(r['class_id']) for r in manifest_rows if r['split'] == 'validation'}
    demo_ids = json.loads((output / 'demo_ids.json').read_text())
    assert len(set(demo_ids)) == 43 and {validation[k] for k in demo_ids} == set(range(43))
    assert len(records) == 43 * 25
    combinations = {(r['sample_id'], r['condition'], r['method']) for r in records}
    assert len(combinations) == len(records)
    expected_demo = {(i, 'clean', 'clean') for i in demo_ids} | {
        (i, c, m) for i in demo_ids for c in ['mild', 'medium', 'severe'] for m in protocol['methods']}
    assert combinations == expected_demo
    for r in records:
        assert int(r['class_id']) == validation[r['sample_id']]
        assert (demo / r['image']).resolve().is_relative_to(demo.resolve()) and (demo / r['image']).is_file()
        assert 0 <= float(r['true_probability']) <= 1 and 0 <= int(r['prediction']) < 43
    data = [{k: r[k] for k in ['sample_id', 'class_id', 'condition', 'method', 'prediction', 'true_probability', 'image']}
            for r in records]
    document = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GTSRB: demo tăng sáng và phân loại</title><style>
*{box-sizing:border-box}body{margin:32px auto;padding:0 24px;max-width:1120px;font:17px Arial,sans-serif;color:#172536;background:#fff}h1{font-size:30px;margin:0 0 12px}p{line-height:1.5;color:#526272}label{margin-right:24px}select{padding:8px;font:inherit;margin:8px}#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:24px}article{border-top:1px solid #cbd4df;padding:12px 0}h2{font-size:20px;margin:0 0 10px}img{width:144px;height:144px;image-rendering:pixelated;display:block}article p{font-size:16px;margin:10px 0 0}.correct{color:#22632e}.wrong{color:#a0271f}@media(max-width:620px){#grid{grid-template-columns:repeat(2,1fr)}body{padding:0 16px}}
</style><body><h1>Tăng sáng có giúp nhận đúng biển báo?</h1><p>Chọn một crop validation và mức tối để xem đủ bảy phương pháp với cùng ResNet-18 seed 11. Đây là demo 43 lớp, không phải bảng đánh giá cuối.</p>
<label>Biển báo <select id="sample" aria-label="Chọn lớp biển báo"></select></label><label>Mức tối <select id="condition" aria-label="Chọn mức tối"><option value="mild">Nhẹ</option><option value="medium" selected>Vừa</option><option value="severe">Nặng</option></select></label>
<p id="caption"></p><main id="grid"></main><script>const data=__DATA__;
const names={clean:'Crop gốc',identity:'Ảnh tối',gamma:'Gamma',he:'HE',clahe:'CLAHE',retinex:'MSR',zero_dce:'Zero-DCE',retinexformer:'Retinexformer',diffusion:'Diffusion'};
const sample=document.getElementById('sample'),condition=document.getElementById('condition'),grid=document.getElementById('grid');
data.filter(r=>r.method==='clean').sort((a,b)=>Number(a.class_id)-Number(b.class_id)).forEach(r=>{const o=document.createElement('option');o.value=r.sample_id;o.textContent='Class '+r.class_id;sample.append(o)});
function render(){grid.replaceChildren();const rows=data.filter(r=>r.sample_id===sample.value&&(r.condition===condition.value||r.method==='clean'));document.getElementById('caption').textContent=sample.value+' — nhãn thật: class '+rows[0].class_id;Object.keys(names).forEach(m=>{const r=rows.find(x=>x.method===m);if(!r)return;const a=document.createElement('article'),h=document.createElement('h2'),im=document.createElement('img'),p=document.createElement('p');h.textContent=names[m];im.src=r.image;im.alt=names[m]+' cho class '+r.class_id;p.textContent='Dự đoán: class '+r.prediction+' ('+(r.prediction===r.class_id?'đúng':'sai')+')';p.className=r.prediction===r.class_id?'correct':'wrong';a.append(h,im,p);grid.append(a)});}
sample.addEventListener('change',render);condition.addEventListener('change',render);render();</script></body></html>'''
    (demo / 'index.html').write_text(document.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')))
    write_json(output / 'verification.json', {
        'status': 'verified', 'ablation_runs': 288, 'images_per_run': 430,
        'predictions_checked': 288*430, 'demo_samples': 43, 'demo_images': len(records),
        'manifest_sha256': digest(manifest), 'protocol_sha256': digest(output / 'protocol.json')})
    print('VERIFIED:', output / 'verification.json')
    print('DEMO:', demo / 'index.html')


if __name__ == '__main__':
    build()
