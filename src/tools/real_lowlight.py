"""Evaluate independently annotated real low-light crops without synthetic degradation.

python -m src.tools.real_lowlight --help
The manifest records human provenance assertions, not an automatic proof of real capture.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from src.common import digest, device_for, synchronize
from src.enhancement import Enhancer, provenance
from src.experiment import load_cnn, source_code_hash
from src.metrics import classification
from src.model import image_tensor


def load_manifest(path, root):
    required = {'sample_id','image_path','class_id','group_id','x1','y1','x2','y2','source_url','license','annotator','capture_kind'}
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Manifest requires {sorted(required)}')
        rows = list(reader)
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique, nonempty samples')
    samples, groups, seen_crops = [], {}, set()
    for r in rows:
        if any(not r[k].strip() for k in required) or r['capture_kind'] != 'real_lowlight':
            raise ValueError('Missing provenance or not declared real_lowlight')
        label = int(r['class_id'])
        if not 0 <= label < 43:
            raise ValueError('Only manually verified GTSRB classes 0..42 are accepted')
        box = tuple(int(r[k]) for k in ['x1','y1','x2','y2'])
        path_ = (Path(root)/r['image_path']).resolve()
        if not path_.is_relative_to(Path(root).resolve()):
            raise ValueError('Image must be inside the image root')
        sha = digest(path_)
        if (sha,box) in seen_crops:
            raise ValueError('Duplicate crop; do not inflate the sample count')
        seen_crops.add((sha,box))
        if sha in groups and groups[sha] != r['group_id']:
            raise ValueError('ROIs in the same source image must share a group ID')
        groups[sha] = r['group_id']
        with Image.open(path_) as im:
            x1,y1,x2,y2 = box
            if not 0 <= x1 < x2 <= im.width or not 0 <= y1 < y2 <= im.height:
                raise ValueError('Invalid exclusive xyxy ROI')
            # Matches crop_box: 5% padding, round outward, clamp to source bounds.
            from src.data import crop_box
            padded = crop_box(box,im.width,im.height,padding=0.05,inclusive=False)
            crop = im.convert('RGB').crop(padded).resize((64,64),Image.Resampling.BILINEAR)
        samples.append((dict(r,class_id=label,image_sha256=sha), np.asarray(crop,dtype=np.float32)/255))
    return samples


def evaluate_real(manifest, image_root, selection_path, checkpoints, output, device):
    selection = json.loads(Path(selection_path).read_text(encoding='utf-8'))
    if selection.get('pilot',True):
        raise ValueError('Use a completed, non-pilot validation selection')
    if selection['code_hash'] != source_code_hash():
        raise ValueError('Implementation differs from validation; restore the original source or rerun validation first')
    if {digest(p) for p in checkpoints} != set(selection['checkpoints'].values()) or len(checkpoints) != 3:
        raise ValueError('Supply exactly the original three selected classifier checkpoints')
    samples = load_manifest(manifest,image_root)
    methods = list(dict.fromkeys(['identity','clahe',selection['chosen_method']]))
    models = [load_cnn(p,selection['manifest_sha256'],device,selection['config']['cnn']) for p in checkpoints]
    if {s['seed'] for _,s in models} != {11,22,33}:
        raise ValueError('Require training seeds 11/22/33')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new real-lowlight output directory')
    output.mkdir(parents=True)
    protocol = {'status':'running','domain':'real_lowlight','selection_sha256':digest(selection_path),
                'implementation_hash':source_code_hash(),
                'manifest_sha256':digest(manifest),'methods':methods,'checkpoints':{str(p):digest(p) for p in checkpoints},
                'samples':[r for r,_ in samples], 'roi':'exclusive xyxy, pad 5%, bilinear RGB 64x64',
                'limits':'Human provenance assertions; no paired clean reference, no PSNR/SSIM; descriptive evaluation only'}
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    def write(name, rows):
        with (output/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    predictions, summaries = [], []
    for method in methods:
        params = selection['enhancers'][method]
        if provenance(method,params) != selection['enhancer_provenance'][method]:
            raise ValueError(f'Enhancer provenance changed: {method}')
        enhancer = Enhancer(method,params,device)
        enhancer(samples[0][1],samples[0][0]['sample_id'])  # warm-up, excluded from timing
        enhanced, latencies = [], []
        for row,image in samples:
            synchronize(device); start=time.perf_counter()
            result=enhancer(image,row['sample_id'])
            synchronize(device); latencies.append(1000*(time.perf_counter()-start))
            enhanced.append(result)
        for model,saved in models:
            labels, predicted = [], []
            for (row,_),image in zip(samples,enhanced):
                with torch.inference_mode():
                    probabilities = model(image_tensor(image).unsqueeze(0).to(device)).softmax(1)[0].cpu()
                pred=int(probabilities.argmax()); labels.append(row['class_id']); predicted.append(pred)
                predictions.append({'sample_id':row['sample_id'],'group_id':row['group_id'],'class_id':row['class_id'],
                                    'method':method,'train_seed':saved['seed'],'prediction':pred,'confidence':float(probabilities[pred])})
            score=classification(labels,predicted)
            observed=sorted(set(labels))
            summaries.append({'method':method,'train_seed':saved['seed'],'n_images':len(samples),
                              'n_groups':len({r['group_id'] for r,_ in samples}),'n_observed_classes':len(observed),
                              'accuracy':score['accuracy'],'macro_f1_43':score['macro_f1'],
                              'macro_f1_observed':float(np.mean([score['f1_per_class'][c] for c in observed])),
                              'weighted_f1':score['weighted_f1'],'enhancer_ms':float(np.mean(latencies))})
            write('predictions.csv',predictions); write('summary.csv',summaries)
    aggregate=[]
    for method in methods:
        rows=[r for r in summaries if r['method']==method]
        r={'method':method}
        for k in ['accuracy','macro_f1_43','macro_f1_observed','weighted_f1']:
            r[k+'_mean']=float(np.mean([v[k] for v in rows])); r[k+'_sd']=float(np.std([v[k] for v in rows],ddof=1))
        aggregate.append(r)
    write('aggregate.csv',aggregate)
    protocol['status']='completed'
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='auto')
    a=p.parse_args()
    print(evaluate_real(a.manifest,a.image_root,a.selection,a.checkpoints,a.output,device_for(a.device)))


if __name__=='__main__':
    main()
