"""Assemble the extension in the existing single public notebook; never publish it."""
import textwrap


def extend(cells,add):
    def find(text):
        matches=[c for c in cells if text in ''.join(c['source']) and 'PROJECT_FILES =' not in ''.join(c['source'])]
        if len(matches)!=1:raise ValueError('Notebook section changed: '+text)
        return matches[0]
    def set_source(cell,text):
        source=textwrap.dedent(text).strip()
        if cell['cell_type']=='code':compile(source,'extension cell','exec')
        cell['source']=source.splitlines(keepends=True)
    def replace(marker,old,new):
        cell=find(marker);source=''.join(cell['source'])
        if old not in source:raise ValueError('Missing notebook replacement '+old)
        set_source(cell,source.replace(old,new))
    set_source(cells[0],'''
        # Xử lý ảnh thiếu sáng cho phân loại biển báo GTSRB

        **Bài tập lớn môn Xử lý ảnh — Nhóm 2.** Một notebook cho toàn bộ nghiên cứu: bảy phương pháp tăng sáng, ba bộ lọc khử nhiễu và hai thứ tự kết hợp. Chọn tham số trên validation, đánh giá subset và full-test, xuất bằng chứng có thể kiểm tra lại.

        `reuse` dùng ba checkpoint và kết quả đối chứng đã xác minh để chạy khảo sát khử nhiễu; `fresh` huấn luyện từ đầu. Không tự huấn luyện khi thiếu checkpoint. Kết quả tái sử dụng được ghi rõ nguồn. Chỉ trạng thái `completed` cùng gói kết quả qua kiểm tra mới xác nhận hoàn tất; không gọi subset là full-test hoặc coi ảnh tối mô phỏng là camera đêm thực.

        GTSRB tải từ [archive tác giả](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). Trọng số enhancer tải từ repo tác giả, khóa commit và SHA-256. Giữ giấy phép và ghi công; không coi việc tải là cấp phép lại. Zero-DCE chỉ dùng cho nghiên cứu/học thuật phi thương mại. Không cần API key trong notebook.
    ''')
    replace('OUT = ROOT/',"OUT = ROOT/'outputs/fresh'",'''TRAIN_MODE = 'reuse'  # 'fresh' để chạy toàn bộ từ đầu.
CHECKPOINT_ARCHIVE = None  # Optional: ZIP đã có trong phiên; None tải trực tiếp Output v1.
REQUIRE_BASELINE_REUSE = True  # Stop instead of rerunning completed baseline experiments.
RUN_ABLATION = TRAIN_MODE == 'fresh'
RUN_EXTENSION_FULL_TEST = True
assert TRAIN_MODE in {'fresh','reuse'}
if TRAIN_MODE == 'reuse':
    from src.tools.baseline_input import resolve_baseline
    CHECKPOINT_ARCHIVE = resolve_baseline(ROOT/'baseline_input',CHECKPOINT_ARCHIVE)
OUT = ROOT/('outputs/fresh' if TRAIN_MODE=='fresh' else 'outputs/reuse')''')
    replace("def status(","'fresh_training':True","'fresh_training':TRAIN_MODE=='fresh','extension_requested':True")
    set_source(find('## 5.'),'''
        ## 5. Khôi phục hoặc huấn luyện ba ResNet-18

        `reuse`: tải ZIP trực tiếp từ Output của phiên v1 của chính notebook, không tạo Dataset và không cần API key. Có thể điền CHECKPOINT_ARCHIVE để dùng ZIP đã có. Kiểm tra checksum gói gốc, sửa riêng đường dẫn supplemental bị trùng và đối chiếu toàn bộ manifest với gói đã kiểm chứng; không thay dự đoán hay checkpoint. Sau đó kiểm tra ba seed, manifest và cấu hình CNN; chỉ đọc checkpoint bằng weights_only, không chạy mã trong ZIP. Selection và dự đoán chỉ được dùng lại khi cùng mã, dữ liệu và checkpoint. `fresh`: ImageNet, tinh chỉnh ảnh gốc, seed 11/22/33, tối đa 20 epoch; chọn checkpoint bằng Macro-F1 validation clean. Sao lưu checkpoint ngay sau bước này.
    ''')
    set_source(find('from src.tools.fresh_run import train_fresh'),'''
        from src.tools.fresh_run import train_fresh
        from src.common import digest
        if TRAIN_MODE == 'fresh':
            status('training',devices=DEVICES)
            CHECKPOINTS = train_fresh(ROOT,CONFIG,OUT/'train',DEVICES)
        else:
            from src.tools.checkpoint_restore import restore
            CHECKPOINTS, RESTORED = restore(ROOT,CHECKPOINT_ARCHIVE,OUT,CONFIG)
            shutil.copy2(ROOT/'baseline_input/input_provenance.json',OUT/'train/input_provenance.json')
            print(RESTORED)
            if REQUIRE_BASELINE_REUSE:
                assert RESTORED['selection_reusable'] and set(RESTORED['baseline_scopes'])=={'subset','fulltest'} and RESTORED['ablation_reusable'], 'Baseline artifacts cannot be reused exactly; stopping before any baseline retraining or reevaluation.'
        write_json(OUT/'checkpoint_hashes.json',{p.name:digest(p) for p in CHECKPOINTS})
        import zipfile
        backup = PUBLIC/'gtsrb-checkpoints.zip'
        with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
            hashes={}
            for p in CHECKPOINTS:
                name='05_training/'+p.name;z.write(p,name);hashes[name]=digest(p)
            z.writestr('SHA256.json',json.dumps(hashes,indent=2))
        status('classifiers_ready',checkpoint_backup=backup.name)
    ''')
    replace('selection = tune(',"selection = tune(ROOT,CONFIG,CHECKPOINTS,torch.device(DEVICES[0]),OUT/'validation',per_class=30,release_cache=True)",
        """if (OUT/'validation/selection.json').exists():
    selection = json.loads((OUT/'validation/selection.json').read_text())
else:
    selection = tune(ROOT,CONFIG,CHECKPOINTS,torch.device(DEVICES[0]),OUT/'validation',per_class=30,release_cache=True)""")
    for scope,per_class in [('subset',',per_class=10'),('fulltest','')]:
        marker='subset_protocol = ' if scope=='subset' else 'full_protocol = '
        variable='subset_protocol' if scope=='subset' else 'full_protocol'
        cell=find(marker);source=''.join(cell['source'])
        source=source.replace('from src.tools.research_evidence import analyze','from src.tools.research_evidence import analyze, audited_runs')
        source=source.replace(f"freeze(ROOT,OUT/'validation/selection.json',{variable}{per_class})\n",'')
        source=source.replace(f"parallel_evaluate(ROOT,{variable},OUT/'{scope}',DEVICES)",f"""if (OUT/'{scope}/summary.csv').exists():
    audited_runs(OUT/'{scope}')
else:
    if not {variable}.exists():freeze(ROOT,OUT/'validation/selection.json',{variable}{per_class})
    parallel_evaluate(ROOT,{variable},OUT/'{scope}',DEVICES)""")
        source=source.replace(f"analyze(OUT/'{scope}',OUT/'{scope}/error_analysis')",f"if not (OUT/'{scope}/error_analysis/audit.json').exists():analyze(OUT/'{scope}',OUT/'{scope}/error_analysis')")
        if scope=='subset':source=source.replace("pack(OUT/'subset',PUBLIC/", "if not (PUBLIC/'gtsrb-subset-results.zip').exists():pack(OUT/'subset',PUBLIC/")
        if scope=='fulltest':source+='''
baseline_backup = PUBLIC/'gtsrb-baseline-results.zip'
if not baseline_backup.exists():
    pack(OUT/'fulltest',baseline_backup,analysis=OUT/'fulltest/error_analysis',source=ROOT,
         supplemental=OUT/'ablation',subset=OUT/'subset')
print('Baseline full-test đã được sao lưu trước khi chạy mở rộng:',baseline_backup.name)
'''
        set_source(cell,source)
    set_source(find("status('ablation')"),'''
        if RUN_ABLATION:
            status('ablation')
            ablation(ROOT,CONFIG,selection,CHECKPOINTS,torch.device(DEVICES[0]),OUT/'ablation')
            display(pd.DataFrame(read_csv(OUT/'ablation/summary.csv')).head())
        else:
            print('Dùng lại ablation đã kiểm tra.' if (OUT/'ablation/summary.csv').exists() else 'Ablation không được yêu cầu.')
        torch.cuda.empty_cache()
        status('ablation_completed' if RUN_ABLATION else 'ablation_not_requested')
    ''')
    # Add with the normal builder, then place these cells before final packaging.
    count=len(cells)
    add('markdown','''
        ## 10. Mở rộng: khử nhiễu và thứ tự xử lý

        Gaussian, bilateral và Non-local Means: hai cấu hình mỗi bộ lọc. Chọn độ mạnh trên 1.290 ảnh validation rồi **giữ cùng tham số** khi so hai thứ tự với Gamma và CLAHE. Tổng cộng 6 ứng viên độc lập và 12 chuỗi kết hợp cần đánh giá thêm trên validation. Không thay CNN, mức tối, noise seed hoặc test ID.

        Subset chạy mọi họ phương pháp. Validation chọn một cặp thứ tự theo điểm trung bình hai thứ tự, kèm đối chứng độc lập, bộ lọc tốt nhất và ứng viên tổng thể tốt nhất cho full-test. Khóa cả hai giao thức trước khi đọc test; không chọn theo subset test. NLM chuyển RGB sang uint8 BGR theo OpenCV và ghi lượng tử hóa vào provenance. Pipeline không đọc nhãn hay ảnh gốc; bộ đánh giá mới đọc để tính metric.
    ''')
    add('code','''
        from src.tools import processing_study as extension
        EXT = OUT/'processing_extension'
        status('extension_validation')
        extension_selection = extension.tune(ROOT,OUT/'validation/selection.json',CHECKPOINTS,
            ROOT/'configs/processing_extension.json',torch.device(DEVICES[0]),EXT/'validation')
        display(pd.DataFrame(extension_selection['scores'])[['method','category','macro_f1','latency_ms']])
        print('Full-test methods locked from validation:',extension_selection['fulltest_methods'])
        extension.freeze(ROOT,EXT/'validation/selection.json',EXT/'protocol_subset.json',per_class=10)
        extension.freeze(ROOT,EXT/'validation/selection.json',EXT/'protocol_fulltest.json',per_class=None)
        for scope in (['subset','fulltest'] if RUN_EXTENSION_FULL_TEST else ['subset']):
            status('extension_'+scope)
            extension.evaluate(ROOT,EXT/('protocol_'+scope+'.json'),EXT/scope,torch.device(DEVICES[0]),
                baseline_output=OUT/scope)
            if not (EXT/scope/'error_analysis/audit.json').exists():
                extension.analyze(EXT/scope,EXT/scope/'error_analysis')
            display(pd.DataFrame(aggregate(read_csv(EXT/scope/'summary.csv')))[
                ['condition','method','accuracy_mean','macro_f1_mean','macro_f1_sd','psnr_mean_mean','ssim_mean_mean']])
            if scope=='subset' and not (PUBLIC/'gtsrb-processing-subset-results.zip').exists():
                pack(EXT/scope,PUBLIC/'gtsrb-processing-subset-results.zip',
                     analysis=EXT/scope/'error_analysis',include_checkpoints=False)
        torch.cuda.empty_cache()
        status('extension_verified',extension_fulltest_completed=RUN_EXTENSION_FULL_TEST)
    ''')
    added=cells[count:];del cells[count:]
    position=cells.index(find('## 10. Xuất'))
    cells[position:position]=added
    replace('## 10. Xuất','## 10. Xuất','## 11. Xuất')
    replace('final_zip = PUBLIC/',"pack(OUT/'fulltest',final_zip,", "staged_zip = PUBLIC/'.gtsrb-final-export.zip'\nif staged_zip.exists():staged_zip.unlink()\npack(OUT/'fulltest',staged_zip,")
    replace('final_zip = PUBLIC/',"supplemental=OUT/'ablation',subset=OUT/'subset')", "supplemental=OUT/'ablation',subset=OUT/'subset',extension=EXT)\nstaged_zip.replace(final_zip)")
    replace('final_zip = PUBLIC/',"tables_only = PUBLIC/'.gtsrb-subset-tables.zip'", "tables_only = PUBLIC/'.gtsrb-subset-tables.zip'\nif tables_only.exists():tables_only.unlink()")
    replace('final_zip = PUBLIC/',"fulltest_prediction_rows=2765970,real_lowlight_evaluated=False)","fulltest_prediction_rows=2765970,real_lowlight_evaluated=False,extension_fulltest_completed=RUN_EXTENSION_FULL_TEST)")
    replace('final_zip = PUBLIC/',"status('completed',final_archive=", """# Remove our explicitly named interim backups only after final ZIP verification.
for name in ['gtsrb-checkpoints.zip','gtsrb-baseline-results.zip','gtsrb-processing-subset-results.zip']:
    path = PUBLIC/name
    if path.exists():path.unlink()
status('completed',final_archive=""")
    for i,cell in enumerate(cells):cell['id']=f'clean-{i:02d}'
