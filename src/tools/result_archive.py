"""A reader-friendly ZIP: merged prediction tables, explicit scope, hashes, no input images."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .research_evidence import audited_runs, read_csv


def aggregate(summaries):
    metrics=['accuracy','macro_f1','weighted_f1','psnr_mean','ssim_mean','latency_ms']
    within=defaultdict(list)
    for r in summaries:
        within[r['condition'],r['method'],r['train_seed']].append(r)
    between=defaultdict(list)
    for (condition,method,_),rows in within.items():
        between[condition,method].append({k:float(np.mean([float(r[k]) for r in rows])) for k in metrics})
    result=[]
    for (condition,method),rows in sorted(between.items()):
        out={'condition':condition,'method':method,'n_training_seeds':len(rows)}
        for k in metrics:
            values=[r[k] for r in rows]
            out[k+'_mean']=float(np.mean(values))
            out[k+'_sd']=0.0 if len(values)==1 or all(np.isinf(values)) else float(np.std(values,ddof=1))
        result.append(out)
    return result


def pack(folder, output, analysis=None, source=None, notebook=None, supplemental=None, include_checkpoints=True, subset=None, extension=None):
    folder,output=Path(folder),Path(output)
    protocol,runs=audited_runs(folder)
    if output.exists():
        raise FileExistsError('Use a new archive path')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.partial.zip')
    if temporary.exists():
        temporary.unlink()
    metadata={'schema':'gtsrb-readable-results-v1','status':'completed_and_prediction_audited',
              'scope':protocol['subset_id'],'images_per_run':len(protocol['sample_ids']),
              'runs':len(runs),'prediction_rows':sum(len(p) for _,p,_ in runs),
              'aggregation':'Average noise within checkpoint, then mean and sample SD across training checkpoints.',
              'not_included':['GTSRB source images','author model weights','image caches','duplicate demo image galleries'],
              'cache_paths':'Any output_path column refers to regenerable temporary cache, not a file in this ZIP.'}
    def csv_bytes(rows):
        rows=list(rows)
        stream=io.StringIO(newline='')
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return stream.getvalue().encode('utf-8')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('READ_ME_FIRST.md',f'''# GTSRB - kết quả thực nghiệm

Trạng thái: hoàn tất, đã tính lại chỉ số từ dự đoán từng ảnh.
Phạm vi: {len(protocol['sample_ids']):,} ảnh mỗi lượt, {len(runs)} lượt.
{'Đây là full-test chính thức.' if protocol['subset_id']=='official_test' else 'Đây là subset, không phải full-test 12.630 ảnh.'}

## Đọc theo thứ tự
1. `01_summary/aggregate.csv`: bảng kết quả trung bình và độ lệch chuẩn.
2. `02_analysis/`: F1 từng lớp, các cặp nhầm, trường hợp được giúp/bị làm sai và biểu đồ.
3. `03_predictions/test.csv`: toàn bộ dự đoán gộp thành một bảng; lọc theo run_id, method, condition và seed.
4. `04_protocol/`: cấu hình khóa trước test, trạng thái lượt chạy và thông tin nguồn.
5. `05_training/`: checkpoint do lần chạy tạo ra và lịch sử huấn luyện, nếu có.
6. `06_reproduce/`: mã nguồn/notebook kèm theo, nếu có.
7. `07_supplemental/`: validation/ablation cũ hoặc bổ sung, nếu có; không trộn với kết quả test chính.
8. `08_processing_extension/`: nghiên cứu khử nhiễu và thứ tự xử lý, nếu đã chạy; tách rõ subset và full-test.

## Quy ước
Noise seed được lấy trung bình trong mỗi training seed trước khi tính mean và sample SD.
Các checkpoint dùng chung ảnh; không coi các seed là những ảnh độc lập.
Confusion matrix có hàng là nhãn thật, cột là lớp dự đoán (0-42).
`confusion_matrices.npz` gồm run_ids và matrices, đọc bằng numpy.load(..., allow_pickle=False).
SHA256.json kiểm tra tính toàn vẹn từng file, không tự chứng minh quyền dữ liệu hoặc tính đúng của nhãn.

## Những thứ không đưa vào ZIP
Không kèm ảnh GTSRB gốc, trọng số enhancer của tác giả hoặc cache trung gian.
Không kèm hàng nghìn ảnh demo lặp; giữ bảng ID/dự đoán để truy vết và tái tạo bằng mã nguồn.
Các đường dẫn output_path trong bảng là cache tạm có thể đã giải phóng, không phải file bị thiếu trong ZIP.
''')
        z.writestr('04_protocol/archive_info.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        summaries=[r for r,_,_ in runs]
        z.writestr('01_summary/aggregate.csv',csv_bytes(aggregate(summaries)))
        z.writestr('01_summary/runs.csv',csv_bytes(summaries))
        first_meta=next(iter(runs))[0]
        extra=['run_id','condition','method','train_seed']
        fields=extra+[k for k in runs[0][1][0] if k not in extra]
        with z.open('03_predictions/test.csv','w') as binary:
            stream=io.TextIOWrapper(binary,encoding='utf-8',newline='')
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for run,predictions,_ in runs:
                for row in predictions:
                    writer.writerow({**row,**{k:run[k] for k in extra}})
            stream.flush();stream.detach()
        matrices=io.BytesIO()
        np.savez_compressed(matrices,run_ids=np.asarray([r['run_id'] for r,_,_ in runs]),
                            matrices=np.asarray([s['confusion_matrix'] for _,_,s in runs],dtype=np.int32))
        z.writestr('02_analysis/confusion_matrices.npz',matrices.getvalue())
        for name in ['protocol.json','runs.json','environment.json','classifier_latency.json']:
            if (folder/name).is_file():z.write(folder/name,'04_protocol/'+name)
        for name in ['restore.json','checkpoint_hashes.json','upstream_sources.json','source_hashes.json']:
            if (folder.parent/name).is_file():z.write(folder.parent/name,'04_protocol/'+name)
        if analysis:
            analysis=Path(analysis)
            for name in ['per_class_aggregate.csv','transitions.csv','confusion_pairs.csv','examples.csv','audit.json']:
                if (analysis/name).is_file():z.write(analysis/name,'02_analysis/'+name)
            for p in sorted(analysis.glob('*.png')):
                z.write(p,'02_analysis/figures/'+p.name)
        train=folder.parent/'train'
        if train.is_dir():
            for p in sorted(train.iterdir()):
                if p.is_file() and (p.suffix in {'.json','.csv'} or (include_checkpoints and p.suffix=='.pt')):
                    z.write(p,'05_training/'+p.name)
        validation=folder.parent/'validation'
        if validation.is_dir():
            for p in sorted(validation.glob('*.json')):z.write(p,'04_protocol/validation/'+p.name)
            for p in sorted(validation.glob('*.csv')):z.write(p,'07_supplemental/validation/'+p.name)
        # Preserve supplemental predictions in one CSV per section rather than hundreds of files.
        sections=[]
        if (validation/'predictions').is_dir():sections.append(('validation',validation))
        if supplemental and (Path(supplemental)/'predictions').is_dir():sections.append(('ablation',Path(supplemental)))
        if subset and (Path(subset)/'predictions').is_dir():sections.append(('subset',Path(subset)))
        for label,section in sections:
            summary_path=section/('validation_runs.csv' if label=='validation' else 'summary.csv')
            index={}
            if summary_path.exists():
                from src.common import config_id
                for r in read_csv(summary_path):
                    if 'run_id' in r: rid=r['run_id']
                    else:
                        # Validation hashes are generated before CSV serialization.
                        raw=dict(r)
                        for k in ['noise_seed','train_seed','n_images']:raw[k]=int(raw[k])
                        for k in ['macro_f1','accuracy','weighted_f1']:raw[k]=float(raw[k])
                        rid=config_id(raw)
                    index[rid]=r
            paths=sorted((section/'predictions').glob('*.csv'))
            if not paths:continue
            first=read_csv(paths[0])[0]
            meta_fields=['run_id','variant','method','condition','train_seed']
            fields=meta_fields+[k for k in first if k not in meta_fields]
            with z.open('07_supplemental/'+label+'/predictions.csv','w') as binary:
                stream=io.TextIOWrapper(binary,encoding='utf-8',newline='')
                w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
                for p in paths:
                    meta=index.get(p.stem,{})
                    for row in read_csv(p):w.writerow({**row,**{k:meta.get(k,p.stem if k=='run_id' else '') for k in meta_fields}})
                stream.flush();stream.detach()
            if label in {'ablation','subset'}:
                for p in sorted(section.glob('*.json')):z.write(p,'07_supplemental/'+label+'/'+p.name)
                for p in sorted(section.glob('*.csv')):z.write(p,'07_supplemental/'+label+'/'+p.name)
                if (section/'demo/predictions.csv').exists():z.write(section/'demo/predictions.csv','07_supplemental/demo_predictions.csv')
        if extension:
            extension=Path(extension)
            for scope in ['subset','fulltest']:
                section=extension/scope
                if not (section/'summary.csv').exists():continue
                ep,eruns=audited_runs(section)
                prefix='08_processing_extension/'+scope+'/'
                z.writestr(prefix+'aggregate.csv',csv_bytes(aggregate([r for r,_,_ in eruns])))
                for name in ['protocol.json','runs.json','summary.csv','reuse.json']:
                    if (section/name).is_file():z.write(section/name,prefix+name)
                extra=['run_id','condition','method','train_seed']
                fields=extra+[k for k in eruns[0][1][0] if k not in extra]
                with z.open(prefix+'predictions.csv','w') as binary:
                    stream=io.TextIOWrapper(binary,encoding='utf-8',newline='')
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    for run,predictions,_ in eruns:
                        for row in predictions:writer.writerow({**row,**{k:run[k] for k in extra}})
                    stream.flush();stream.detach()
                matrices=io.BytesIO()
                np.savez_compressed(matrices,run_ids=np.asarray([r['run_id'] for r,_,_ in eruns]),
                    matrices=np.asarray([s['confusion_matrix'] for _,_,s in eruns],dtype=np.int32))
                z.writestr(prefix+'confusion_matrices.npz',matrices.getvalue())
                for name in ['per_class_aggregate.csv','transitions.csv','confusion_pairs.csv','examples.csv','order_comparisons.csv','audit.json']:
                    p=section/'error_analysis'/name
                    if p.is_file():z.write(p,prefix+'analysis/'+name)
            validation=extension/'validation'
            for p in sorted(validation.glob('*.json')):z.write(p,'08_processing_extension/validation/'+p.name)
            for p in sorted(validation.glob('*.csv')):z.write(p,'08_processing_extension/validation/'+p.name)
            paths=sorted((validation/'predictions').glob('*.csv'))
            if paths:
                fields=['run_id']+[k for k in read_csv(paths[0])[0] if k!='run_id']
                with z.open('08_processing_extension/validation/predictions.csv','w') as binary:
                    stream=io.TextIOWrapper(binary,encoding='utf-8',newline='')
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    for p in paths:
                        for row in read_csv(p):writer.writerow({**row,'run_id':p.stem})
                    stream.flush();stream.detach()
        if source:
            source=Path(source)
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as snapshot:
                for directory in ['src','configs']:
                    for p in sorted((source/directory).rglob('*')):
                        if p.is_file() and p.suffix in {'.py','.json'} and '__pycache__' not in p.parts:
                            # Only own implementation; upstream external/ is never bundled.
                            snapshot.write(p,p.relative_to(source).as_posix())
            z.writestr('06_reproduce/project_source.zip',buffer.getvalue())
        if notebook:z.write(notebook,'06_reproduce/notebook.ipynb')
    with zipfile.ZipFile(temporary,'a',compression=zipfile.ZIP_DEFLATED) as z:
        hashes={}
        for name in z.namelist():
            h=hashlib.sha256()
            with z.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            hashes[name]=h.hexdigest()
        z.writestr('SHA256.json',json.dumps(hashes,indent=2))
    temporary.replace(output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analysis',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--notebook',type=Path)
    p.add_argument('--supplemental',type=Path)
    p.add_argument('--subset',type=Path)
    a=p.parse_args()
    print(pack(a.input,a.output,a.analysis,a.source,a.notebook,a.supplemental,subset=a.subset))
