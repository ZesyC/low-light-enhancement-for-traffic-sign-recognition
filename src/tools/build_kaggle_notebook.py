"""Build a standalone public Kaggle notebook from an explicit source-file allowlist."""
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[2]
FILES=sorted((ROOT/'src').glob('*.py'))+[ROOT/'src/tools'/name for name in
    ['__init__.py','public_sources.py','fresh_run.py','research_evidence.py','result_archive.py',
     'checkpoint_restore.py','processing_pipeline.py','processing_study.py','baseline_input.py']]
FILES += [ROOT/'configs/resnet18.json',ROOT/'configs/public_sources.json',ROOT/'configs/processing_extension.json']
sources={p.relative_to(ROOT).as_posix():p.read_text(encoding='utf-8') for p in FILES}
cells=[]


def add(kind,source,hidden=False):
    cell={'cell_type':kind,'id':f'clean-{len(cells):02d}','metadata':{},
          'source':textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind=='code':
        cell.update(execution_count=None,outputs=[])
        compile(''.join(cell['source']),cell['id'],'exec')
    if hidden:cell['metadata']={'jupyter':{'source_hidden':True},'tags':['implementation']}
    cells.append(cell)


add('markdown','''
# Tăng cường ảnh thiếu sáng cho phân loại biển báo GTSRB

**Bài tập lớn môn Xử lý ảnh — Nhóm 2.** Notebook public chạy mới từ đầu: huấn luyện ba ResNet-18 seed 11/22/33 trên ảnh gốc, chọn tham số trên validation, đánh giá subset và full-test, rồi xuất kết quả có thể kiểm tra lại.

Không cần dataset private, file API, checkpoint GTSRB cũ hay Google Drive cá nhân. GTSRB tải trực tiếp từ [archive tác giả](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). Mã và trọng số enhancer tải trực tiếp từ [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE), [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer), [Wavelet diffusion](https://github.com/JianghaiSCU/Diffusion-Low-Light), khóa commit và SHA-256. Giữ giấy phép và ghi công gốc; không coi việc tải về là cấp phép lại. Zero-DCE chỉ dùng cho mục đích nghiên cứu/học thuật phi thương mại.

Đây là **lần chạy mới**, không phải kết quả cũ được sao chép. Chỉ gọi là hoàn tất khi RUN_STATUS.json ghi completed và gói kết quả đã qua kiểm tra dự đoán. Ảnh thiếu sáng là mô phỏng; không tuyên bố đã đánh giá camera đêm thực.
''')
add('markdown','''
## 1. Mã nguồn đi kèm

Đây là notebook chính duy nhất: toàn bộ bước chuẩn bị dữ liệu, train/khôi phục, chọn tham số, đánh giá và đóng gói đều nằm trong file này. Cell mã nguồn tự tạo các module Python và cấu hình trong thư mục tạm; không cần tải riêng src/configs hoặc chạy notebook khác. GTSRB, trọng số tác giả và ZIP kết quả dùng lại được tải theo giao thức đã ghi.

Cell bên dưới chứa bản chụp mã nguồn của chính dự án, đọc được và có hash từng file. Không cần clone một nhánh GitHub chưa được công bố. Mã, ảnh đầu vào và trọng số tác giả được lưu trong `/tmp/gtsrb-clean`, tách khỏi Output công khai. Chỉ các kết quả và checkpoint huấn luyện mới được xuất ra `/kaggle/working`.
''')
add('code','''
import os, sys, json, hashlib, subprocess, shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path('/tmp/gtsrb-clean')
PUBLIC = Path('/kaggle/working')
ROOT.mkdir(parents=True,exist_ok=True)
PUBLIC.mkdir(parents=True,exist_ok=True)
''')
payload='PROJECT_FILES = '+json.dumps(sources,ensure_ascii=False,indent=2)+'''
for relative, content in PROJECT_FILES.items():
    path = ROOT / relative
    assert path.resolve().is_relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8')
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
SOURCE_HASHES = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PROJECT_FILES}
print('Source snapshot:',len(SOURCE_HASHES),'files; no API credentials required.')
'''
add('code',payload,hidden=True)
add('markdown','''
## 2. Môi trường và cấu hình

Giữ bộ PyTorch/CUDA của Kaggle. Chỉ cài dependency còn thiếu. Nếu có hai GPU, các training seed và nhóm đánh giá được phân công độc lập giữa hai GPU; mỗi seed vẫn giữ batch size, optimizer và dữ liệu như cấu hình. Trọng số ba enhancer chỉ dùng suy luận.

Full-test tốn nhiều giờ. Notebook giải phóng cache sau khi cả ba classifier dùng xong một nhóm ảnh; không lưu toàn bộ cache vào Output. Không tự giảm số seed hoặc phương pháp khi sắp hết thời gian.
''')
add('code','''
import importlib.util, importlib.metadata
dependencies = {'einops':'einops','yaml':'PyYAML','pytorch_msssim':'pytorch-msssim','gdown':'gdown',
                'skimage':'scikit-image','sklearn':'scikit-learn','cv2':'opencv-python-headless'}
missing = [package for module,package in dependencies.items() if importlib.util.find_spec(module) is None]
try:
    importlib.metadata.version('opencv-python-headless')
except importlib.metadata.PackageNotFoundError:
    missing.append('opencv-python-headless')
if missing:
    subprocess.run([sys.executable,'-m','pip','install','-q',*dict.fromkeys(missing)],check=True)
import torch, numpy as np, pandas as pd
torch.set_num_threads(2)
assert torch.cuda.is_available(), 'Bật GPU và Internet trong Settings trước khi chạy.'
DEVICES = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
CONFIG = json.loads((ROOT/'configs/resnet18.json').read_text())
OUT = ROOT/'outputs/fresh'
OUT.mkdir(parents=True,exist_ok=True)
NOTEBOOK_URL = 'https://www.kaggle.com/code/kieuthithutrang/gtsrb-low-light-enhancement-for-traffic-signs'
from src.common import write_json, environment
def status(stage, **extra):
    record = {'stage':stage,'notebook_url':NOTEBOOK_URL,'fresh_training':True,**extra}
    write_json(PUBLIC/'RUN_STATUS.json',record)
    print('STAGE:',stage,flush=True)
status('environment_ready',devices=DEVICES)
print('PyTorch:',torch.__version__,'GPU:',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
''')
add('markdown','''
## 3. Tải và kiểm tra trọng số tác giả trước khi train

Kiểm tra tải được file, đúng checksum, đúng kiến trúc và suy luận được RGB 64×64. Nếu nguồn tác giả không truy cập được, dừng trước huấn luyện; không thay bằng trọng số khác hoặc lấy từ dataset private.
''')
add('code','''
from src.tools.public_sources import provision
from src.enhancement import Enhancer
UPSTREAM = provision(ROOT,ROOT/'configs/public_sources.json')
for method in ['zero_dce','retinexformer','diffusion']:
    enhancer = Enhancer(method,CONFIG['enhancers'][method],torch.device(DEVICES[0]))
    preview = enhancer(np.full((64,64,3),.15,dtype=np.float32),'preflight')
    assert preview.shape == (64,64,3) and np.isfinite(preview).all()
    print('Forward OK:',method,flush=True)
    del enhancer
torch.cuda.empty_cache()
write_json(OUT/'upstream_sources.json',UPSTREAM)
write_json(OUT/'source_hashes.json',SOURCE_HASHES)
status('upstream_verified')
''')
add('markdown','''
## 4. GTSRB chính thức và chia dữ liệu

Nhãn train nằm trong 43 file `GT-*.csv` ngay trong ZIP ảnh train; nhãn test tải bằng gói GT test riêng. Chia train/validation theo class+track, seed 42, cách ly track có ảnh trùng pixel với test; crop ROI padding 5%, RGB 64×64. Chọn trước ID subset test để không chọn ví dụ theo kết quả.
''')
add('code','''
from src.data import download, prepare, audit, subset_rows
from src.common import read_csv
download(ROOT)
prepare(ROOT,CONFIG)
audit(ROOT)
rows = read_csv(ROOT/'data/manifests/gtsrb_samples.csv')
display(pd.DataFrame(rows).groupby('split').size().rename('Số ảnh').to_frame())
assert sum(r['split']=='test' for r in rows) == 12630
assert sum(r['split']=='train' for r in rows) == 31350
assert sum(r['split']=='validation' for r in rows) == 7829
write_json(OUT/'predeclared_subset_ids.json',[r['sample_id'] for r in subset_rows(rows,'test',10,42)])
status('dataset_verified')
''')
add('markdown','''
## 5. Huấn luyện mới ResNet-18

Khởi tạo ImageNet, tinh chỉnh toàn mạng trên ảnh gốc; augmentation nhẹ, không dùng mô hình suy giảm test để huấn luyện. Ba seed 11/22/33, tối đa 20 epoch, dừng sớm; chọn checkpoint bằng Macro-F1 validation clean. Không đọc checkpoint GTSRB cũ.
''')
add('code','''
from src.tools.fresh_run import train_fresh
status('training',devices=DEVICES)
CHECKPOINTS = train_fresh(ROOT,CONFIG,OUT/'train',DEVICES)
from src.common import digest
write_json(OUT/'new_checkpoint_hashes.json',{p.name:digest(p) for p in CHECKPOINTS})
status('training_completed')
display(pd.DataFrame([json.loads((OUT/'train'/f'validation_{s}.json').read_text()) for s in [11,22,33]]))
''')
add('markdown','''
## 6. Chọn phương pháp trên validation

1.290 ảnh validation, 3 mức suy giảm và 3 noise seed, cùng ba classifier. Chọn cấu hình Gamma/CLAHE/MSR và pipeline bằng Macro-F1 validation; chưa xem test. Cache giải phóng sau khi ba checkpoint dùng xong.
''')
add('code','''
from src.experiment import tune, freeze
status('validation_selection')
selection = tune(ROOT,CONFIG,CHECKPOINTS,torch.device(DEVICES[0]),OUT/'validation',per_class=30,release_cache=True)
torch.cuda.empty_cache()
display(pd.DataFrame(selection['scores']))
print('Pipeline chọn trước test:',selection['chosen_method'])
status('validation_completed',selected_method=selection['chosen_method'])
''')
add('markdown','''
## 7. Subset 430 ảnh: kiểm tra giao thức và phân tích lỗi

219 lượt, đủ 43 lớp (10 ảnh/lớp). Tính lại metric từ dự đoán, F1 từng lớp, ma trận nhầm lẫn và các trường hợp được giúp/bị làm sai. Đây là subset; không gọi là full-test. Xuất gói subset trước để vẫn giữ được bằng chứng nếu phiên Kaggle bị ngắt trong các bước dài sau đó.
''')
add('code','''
from src.tools.fresh_run import parallel_evaluate, ablation
from src.tools.research_evidence import analyze
from src.tools.result_archive import pack, aggregate
subset_protocol = OUT/'protocol_subset.json'
freeze(ROOT,OUT/'validation/selection.json',subset_protocol,per_class=10)
status('subset_evaluation')
parallel_evaluate(ROOT,subset_protocol,OUT/'subset',DEVICES)
analyze(OUT/'subset',OUT/'subset/error_analysis')
display(pd.DataFrame(aggregate(read_csv(OUT/'subset/summary.csv')))[['condition','method','accuracy_mean','macro_f1_mean','macro_f1_sd']])
pack(OUT/'subset',PUBLIC/'gtsrb-subset-results.zip',analysis=OUT/'subset/error_analysis',source=ROOT)
status('subset_completed',prediction_rows=94170)
''')
add('markdown','''
## 8. Ablation trên subset

Tách Gamma-only, Gamma+Gaussian, Gamma+Poisson và Gamma+cả hai, giữ nguyên checkpoint và tham số đã chọn. Có 288 lượt. Không dùng kết quả ablation để chọn lại pipeline.
''')
add('code','''
status('ablation')
ablation(ROOT,CONFIG,selection,CHECKPOINTS,torch.device(DEVICES[0]),OUT/'ablation')
torch.cuda.empty_cache()
display(pd.DataFrame(read_csv(OUT/'ablation/summary.csv')).head())
status('ablation_completed')
''')
add('markdown','''
## 9. Full-test chính thức 12.630 ảnh

Khóa cùng selection rồi đánh giá đủ 219 lượt, tổng cộng 2.765.970 dự đoán. Các nhóm condition/noise/method độc lập được phân cho GPU khả dụng; cả ba checkpoint nhận cùng ảnh tăng cường trong mỗi nhóm. Cache được tái tạo đúng seed và giải phóng sau nhóm, giữ thông tin hash và dự đoán. Nếu chưa xong bước này, không coi bảng subset là full-test.
''')
add('code','''
full_protocol = OUT/'protocol_fulltest.json'
freeze(ROOT,OUT/'validation/selection.json',full_protocol)
status('fulltest_evaluation')
parallel_evaluate(ROOT,full_protocol,OUT/'fulltest',DEVICES)
analyze(OUT/'fulltest',OUT/'fulltest/error_analysis')
full_aggregate = pd.DataFrame(aggregate(read_csv(OUT/'fulltest/summary.csv')))
display(full_aggregate[['condition','method','accuracy_mean','macro_f1_mean','macro_f1_sd']])
status('fulltest_verified',runs=219,images_per_run=12630,prediction_rows=2765970)
''')
add('markdown','''
## 10. Xuất gói kết quả dễ đọc

Mở `READ_ME_FIRST.md` trong ZIP trước. Bảng dự đoán test gộp vào một CSV với run_id/method/condition/seed; confusion matrix gộp vào NPZ, có bản kê SHA-256. Validation, subset và ablation nằm riêng, không trộn với full-test. Không đóng gói ảnh gốc, cache hoặc trọng số enhancer tác giả.

Gói subset có checkpoint để khôi phục nếu phiên bị ngắt. Sau khi full-test hoàn tất, checkpoint chỉ nằm trong gói đầy đủ, tránh giữ hai bản trùng nhau. Kết quả ảnh đêm thực chưa được tạo vì chưa có dữ liệu thực gán nhãn tương thích.
''')
add('code','''
final_zip = PUBLIC/'gtsrb-notebook-results.zip'
pack(OUT/'fulltest',final_zip,analysis=OUT/'fulltest/error_analysis',source=ROOT,
     supplemental=OUT/'ablation',subset=OUT/'subset')
import zipfile
with zipfile.ZipFile(final_zip) as z:
    assert z.testzip() is None
    print('ZIP:',final_zip.name,'files:',len(z.namelist()),'MB:',round(final_zip.stat().st_size/1e6,1))
    print('Top-level:',sorted({n.split('/')[0] for n in z.namelist()}))
# Replace only our own explicitly named subset ZIP with a tables-only version.
tables_only = PUBLIC/'.gtsrb-subset-tables.zip'
pack(OUT/'subset',tables_only,analysis=OUT/'subset/error_analysis',include_checkpoints=False)
tables_only.replace(PUBLIC/'gtsrb-subset-results.zip')
status('completed',final_archive=final_zip.name,archive_sha256=digest(final_zip),
       fulltest_prediction_rows=2765970,real_lowlight_evaluated=False)
print('COMPLETE — tải gtsrb-notebook-results.zip ở Output của notebook.')
''')
add('markdown','''
## Tài liệu và phạm vi

- GTSRB: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, *The German Traffic Sign Recognition Benchmark: A multi-class classification competition*, IJCNN 2011.
- Zero-DCE: C. Guo et al., CVPR 2020.
- Retinexformer: Y. Cai et al., ICCV 2023.
- Wavelet-based Diffusion LLIE: H. Jiang et al., ACM TOG 2023.

Ảnh crop và suy giảm tổng hợp phục vụ nghiên cứu. PSNR/SSIM đo so ảnh gốc trên cùng crop; không thay thế hiệu năng phân loại. Độ lệch chuẩn qua ba checkpoint không phải khoảng tin cậy cho camera thực. Không có phát hiện biển báo toàn cảnh hay đánh giá ảnh đêm thực trong lần chạy này.
''')
from src.tools.notebook_extension import extend
extend(cells,add)
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
          'language_info':{'name':'python'},'gtsrb':{'default_train_mode':'reuse','private_inputs':False,
          'extension_run_ready':True,'baseline_input':'direct_public_notebook_output_zip',
          'source_files':len(sources)}},
          'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks/gtsrb-low-light-enhancement-for-traffic-signs.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(path)
