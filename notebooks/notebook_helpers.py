"""Notebook presentation and orchestration; processing/training stay in cloned src."""
import io, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
from copy import deepcopy
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from IPython.display import display, Image
from src.common import read_csv, read_image, write_csv, write_json, digest, config_id
from src.data import download, prepare, subset_rows
from src.degradation import degrade
from src.enhancement import classical, Enhancer, METHODS
from src.experiment import cache_images, load_cnn, predict, source_code_hash
from src.metrics import image_metrics, classification


def show(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    plt.close(fig)


def setup(ROOT, BUNDLE):
    SOURCE = next(p.parent for p in Path('/kaggle/input').rglob('snapshot.json') if (p.parent/'external').is_dir())
    shutil.copytree(SOURCE/'external', ROOT/'external')
    missing = [p for m,p in [('einops','einops'), ('pytorch_msssim','pytorch-msssim==1.0.0')]
               if importlib.util.find_spec(m) is None]
    if missing: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
    print('Đã clone code dự án từ GitHub; không thay torch CUDA của Kaggle.')
    assert torch.cuda.is_available(), 'Bật GPU trước khi chạy.'
    torch.set_num_threads(2)
    CONFIG = json.loads((ROOT/'configs/resnet18.json').read_text())
    DEVICE = torch.device('cuda:0')
    OUT = ROOT/'outputs/notebook_A'
    print('GPU:', torch.cuda.get_device_name(0))
    return CONFIG, DEVICE, OUT


def prepare_data(ROOT, CONFIG, BUNDLE):
    raw = ROOT/'data/raw/gtsrb'
    raw.mkdir(parents=True)
    # Đuôi .bin giữ ZIP gốc khỏi bị Kaggle tự giải nén.
    for archive in (BUNDLE/'raw_archives').glob('*.zip.bin'):
        shutil.copy2(archive, raw/archive.name.removesuffix('.bin'))
    download(ROOT)
    prepare(ROOT, CONFIG)
    MANIFEST = ROOT/'data/manifests/gtsrb_samples.csv'
    ROWS = read_csv(MANIFEST)
    display(pd.DataFrame(ROWS).groupby('split').size().rename('Số ảnh').to_frame())
    validation = subset_rows(ROWS, 'validation', 1, 42)
    DEMO = [next(r for r in validation if int(r['class_id'])==c) for c in range(43)]
    TEST_ROWS = subset_rows(ROWS, 'test', 10, 42)
    assert len(TEST_ROWS)==430
    EXAMPLE = DEMO[1]
    REFERENCE = read_image(ROOT/EXAMPLE['crop_path'])
    write_json(ROOT/'outputs/predeclared_test_ids.json', [r['sample_id'] for r in TEST_ROWS])
    display(Image(filename=str(ROOT/EXAMPLE['crop_path']), width=192))
    return ROWS, TEST_ROWS, DEMO, EXAMPLE, REFERENCE


def show_degradation(REFERENCE, EXAMPLE, CONFIG):
    DARK = {c:degrade(REFERENCE, **p, sample_id=EXAMPLE['sample_id'], noise_seed=101, condition=c)
            for c,p in CONFIG['conditions'].items()}
    display(pd.DataFrame(CONFIG['conditions']).T)
    fig, axes = plt.subplots(2, 4, figsize=(12,5))
    quality = []
    for j,(name,image) in enumerate([('clean',REFERENCE), *DARK.items()]):
        axes[0,j].imshow(image); axes[0,j].set_title(name); axes[0,j].axis('off')
        axes[1,j].hist(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).ravel(), bins=40, range=(0,1))
        axes[1,j].set_xlabel('Cường độ pixel')
        quality.append({'Ảnh':name, **image_metrics(REFERENCE,image,CONFIG['ssim'])})
    show(fig)
    display(pd.DataFrame(quality).round(4))
    return DARK


def show_classical(REFERENCE, dark, CONFIG):
    images = {'identity':dark}
    for method in ['gamma','he','clahe','retinex']:
        images[method] = classical(dark, method, CONFIG['enhancers'][method])
    fig, axes = plt.subplots(2, 5, figsize=(14,5))
    quality = []
    for j,(method,image) in enumerate(images.items()):
        axes[0,j].imshow(image); axes[0,j].set_title(method); axes[0,j].axis('off')
        axes[1,j].hist(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).ravel(), bins=40, range=(0,1))
        axes[1,j].set_xlabel('Cường độ pixel')
        quality.append({'Phương pháp':method, **image_metrics(REFERENCE,image,CONFIG['ssim'])})
    show(fig)
    display(pd.DataFrame(quality).round(4))


def show_parameters(dark):
    fig,axes=plt.subplots(2,3,figsize=(9,5.5))
    for ax,gamma in zip(axes[0],[0.4,0.6,0.8]):
        image=classical(dark,'gamma',{'gamma':gamma})
        ax.imshow(image);ax.set_title(f'Gamma {gamma}');ax.axis('off')
    for ax,clip in zip(axes[1],[1.0,2.0,4.0]):
        image=classical(dark,'clahe',{'clip_limit':clip,'tile_grid':[8,8]})
        ax.imshow(image);ax.set_title(f'CLAHE clipLimit {clip}');ax.axis('off')
    show(fig)
    print('Gamma nhỏ nâng sáng mạnh hơn; CLAHE clipLimit lớn có thể làm nhiễu rõ hơn. Chọn tham số bằng validation.')


def show_training(OUT):
    validation_scores = [{'seed':s, **json.loads((OUT/'train'/f'validation_{s}.json').read_text())}
                         for s in [11, 22, 33]]
    display(pd.DataFrame(validation_scores)[['seed','accuracy','macro_f1','weighted_f1']].round(4))


def show_validation(selection):
    display(pd.DataFrame(selection['scores'])[['method','params','macro_f1','latency_ms']])
    print('Pipeline chọn trên validation:', selection['chosen_method'])


def show_enhancers(REFERENCE, dark, EXAMPLE, selection, DEVICE, CONFIG):
    all_images = {'clean':REFERENCE}
    quality = []
    for method in METHODS:
        enhancer = Enhancer(method, selection['enhancers'][method], DEVICE)
        image = enhancer(dark, EXAMPLE['sample_id'])
        all_images[method] = image
        quality.append({'method':method, **image_metrics(REFERENCE,image,CONFIG['ssim'])})
        del enhancer
    fig, axes = plt.subplots(3, 3, figsize=(9,9))
    for ax,(method,image) in zip(axes.flat,all_images.items()):
        ax.imshow(image); ax.set_title(method); ax.axis('off')
    show(fig)
    display(pd.DataFrame(quality).round(4))


def show_results(ROOT, OUT):
    from src.reporting import report
    from scripts.verify_results import verify
    report(OUT/'test_subset', ROOT)
    verification = verify(ROOT, OUT/'test_subset')
    assert verification['predictions_checked']==94170
    AGG = pd.read_csv(OUT/'test_subset/aggregate.csv')
    table = AGG[['condition','method']].copy()
    for metric in ['accuracy','macro_f1']:
        table[metric+' (%)'] = [f'{100*m:.2f} ± {100*s:.2f}' for m,s in zip(AGG[metric+'_mean'],AGG[metric+'_std'])]
    table['PSNR'] = AGG.psnr_mean_mean.round(2)
    table['SSIM'] = AGG.ssim_mean_mean.round(4)
    display(table)
    for name in ['macro_f1_levels.png','scatter_medium_ssim_mean_macro_f1.png','failure_examples.png','learning_11.png']:
        display(Image(filename=str(OUT/'test_subset/figures'/name)))
    print('Xem trường hợp được giúp/sai thêm: ảnh sáng hơn chưa chắc giữ tốt chi tiết để nhận diện.')


def run_ablation(ROOT, CONFIG, selection, CHECKPOINTS, DEVICE, TEST_ROWS):
    MANIFEST = ROOT/'data/manifests/gtsrb_samples.csv'
    selected_config = deepcopy(CONFIG)
    selected_config['enhancers'].update(selection['enhancers'])
    from scripts.run_research_evidence_kaggle import degradation_variants
    EVIDENCE=ROOT/'outputs/research_evidence'
    variants=degradation_variants(CONFIG['conditions']['medium'])
    models={s:load_cnn(p,digest(MANIFEST),DEVICE,CONFIG['cnn'])[0] for s,p in zip(CONFIG['train_seeds'],CHECKPOINTS)}
    write_json(EVIDENCE/'protocol.json',{'kind':'degradation_ablation','training':False,'tuning':False,
        'sample_ids':[r['sample_id'] for r in TEST_ROWS],'manifest_sha256':digest(MANIFEST),
        'source_code_hash':source_code_hash(),'variants':variants,'methods':list(METHODS),
        'enhancers':selection['enhancers'],'checkpoint_sha256':{p.name:digest(p) for p in CHECKPOINTS},
        'noise_seeds':CONFIG['noise_seeds'],'train_seeds':CONFIG['train_seeds'],
        'role':'supplementary processing analysis; no hyperparameter selection from test'})
    ablation=[]
    from src.common import config_id
    for variant,parameters in variants.items():
        cfg=deepcopy(selected_config);cfg['conditions']['medium']=parameters
        for noise in cfg['noise_seeds']:
            for method in METHODS:
                cached=cache_images(ROOT,TEST_ROWS,digest(MANIFEST),cfg,'medium',noise,method,cfg['enhancers'][method],DEVICE)
                for seed,model in models.items():
                    predictions=predict(ROOT,TEST_ROWS,cached,model,cfg,DEVICE)
                    score=classification([r['class_id'] for r in predictions],[r['prediction'] for r in predictions])
                    run={'variant':variant,'method':method,'noise_seed':noise,'train_seed':seed}
                    run_id=config_id(run)
                    write_csv(EVIDENCE/'predictions'/f'{run_id}.csv',predictions)
                    ablation.append({**run,'run_id':run_id,'n_images':len(predictions),
                        **{k:score[k] for k in ['accuracy','macro_f1','weighted_f1']}})
                    write_csv(EVIDENCE/'summary.csv',ablation)
                print(f'{len(ablation)}/288: {variant}, noise {noise}, {method}',flush=True)
    assert len(ablation)==288


def run_demo(ROOT, CONFIG, selection, CHECKPOINTS, DEVICE, DEMO):
    MANIFEST = ROOT/'data/manifests/gtsrb_samples.csv'
    EVIDENCE = ROOT/'outputs/research_evidence'
    selected_config = deepcopy(CONFIG)
    selected_config['enhancers'].update(selection['enhancers'])
    checkpoint = next(p for p in CHECKPOINTS if p.name=='cnn_11.pt')
    models = {11:load_cnn(checkpoint, digest(MANIFEST), DEVICE, CONFIG['cnn'])[0]}
    write_json(EVIDENCE/'demo_ids.json',[r['sample_id'] for r in DEMO])
    demo_records=[]
    for condition in ['clean',*CONFIG['conditions']]:
        for method in (['clean'] if condition=='clean' else METHODS):
            cached=cache_images(ROOT,DEMO,digest(MANIFEST),selected_config,condition,101,method,selected_config['enhancers'].get(method,{}),DEVICE)
            predictions=predict(ROOT,DEMO,cached,models[11],selected_config,DEVICE)
            for sample,record,prediction in zip(DEMO,cached,predictions):
                name=config_id({'sample_id':sample['sample_id'],'condition':condition,'method':method})+'.png'
                path=EVIDENCE/'demo/images'/name;path.parent.mkdir(parents=True,exist_ok=True)
                PILImage.fromarray(np.rint(read_image(ROOT/record['path'])*255).astype(np.uint8)).save(path)
                demo_records.append({'condition':condition,'method':method,'image':'images/'+name,**prediction})
    write_csv(EVIDENCE/'demo/predictions.csv',demo_records)
    from scripts.finalize_evidence import build as finalize_evidence
    finalize_evidence(ROOT)
    assert len(demo_records)==1075
    display(pd.read_csv(EVIDENCE/'aggregate.csv')[['variant','method','macro_f1_mean','macro_f1_std']].round(4))
    sample_id=DEMO[1]['sample_id']
    records=[r for r in demo_records if r['sample_id']==sample_id and (r['condition']=='medium' or r['method']=='clean')]
    fig,axes=plt.subplots(3,3,figsize=(10,10))
    for ax,r in zip(axes.flat,records):
        ax.imshow(PILImage.open(EVIDENCE/'demo'/r['image']))
        correct=int(r['prediction'])==int(r['class_id'])
        ax.set_title(f'{r["method"]}: class {r["prediction"]}\nNhãn thật {r["class_id"]} — {"đúng" if correct else "sai"}',color='#21653b' if correct else '#a32a2a')
        ax.axis('off')
    show(fig)


def export_results(ROOT, OUT, selection, CHECKPOINTS):
    MANIFEST = ROOT/'data/manifests/gtsrb_samples.csv'
    AGG = pd.read_csv(OUT/'test_subset/aggregate.csv')
    chosen = selection['chosen_method']
    points = {(r.condition,r.method):r for r in AGG.itertuples()}
    print('Pipeline chọn bằng validation:',chosen)
    for condition in ['mild', 'medium', 'severe']:
        gain = 100*(points[condition,chosen].macro_f1_mean-points[condition,'identity'].macro_f1_mean)
        print(f'{condition}: thay đổi Macro-F1 so ảnh tối {gain:+.2f} điểm phần trăm')
    from scripts.finalize_research import build as finalize_research
    finalize_research(OUT/'test_subset')
    write_json(ROOT/'outputs/notebook_run.json', {'status':'complete_and_verified', 'training':'fresh clean seeds 11/22/33',
        'main_runs':219,'main_images':430,'ablation_runs':288,'ablation_images':430,'demo_images':1075,
        'source_code_hash':source_code_hash(),'manifest_sha256':digest(MANIFEST),
        'github_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'checkpoints':{p.name:digest(p) for p in CHECKPOINTS}})
    archive = shutil.make_archive('/kaggle/working/gtsrb-notebook-results','zip',ROOT/'outputs')
    print('COMPLETE:',archive)
    return archive
