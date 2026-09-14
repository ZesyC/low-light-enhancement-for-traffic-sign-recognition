"""Add data-driven discussion and an offline HTML slide deck after completed evaluation."""
import argparse
import base64
import html
import json
from pathlib import Path


from src.common import read_csv


def table(headers, rows):
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in headers)
    body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(c))}</td>' for c in row)+'</tr>' for row in rows)
    compact = ' class="compact"' if len(rows) > 8 else ''
    return f'<table{compact}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def image(path, alt):
    path = Path(path)
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def build(output):
    output = Path(output)
    if not (output/'report.md').exists():
        raise ValueError('Run report on a completed evaluation first')
    protocol = json.loads((output/'protocol.json').read_text())
    if protocol['pilot'] or len(protocol['runs']) != 219:
        raise ValueError('This thesis deck requires the complete 219-run protocol, not pilot results')
    selection = protocol['selection']
    rows = read_csv(output/'aggregate.csv')
    points = {(r['condition'],r['method']):r for r in rows}
    levels = list(protocol['config']['conditions'])
    methods = selection['methods']
    selected = selection['chosen_method']
    clean = points['clean','clean']
    percent = lambda r,k: 100*float(r[k+'_mean'])
    mean = lambda values: sum(values)/len(values)
    f1 = {m:mean([percent(points[c,m],'macro_f1') for c in levels]) for m in methods}
    best_test = max(f1,key=f1.get)
    delta_selected = f1[selected]-f1['identity']
    tradeoffs = []
    for c in levels:
        baseline = points[c,'identity']
        for m in methods:
            p = points[c,m]
            if float(p['ssim_mean_mean']) > float(baseline['ssim_mean_mean']) and float(p['macro_f1_mean']) < float(baseline['macro_f1_mean']):
                tradeoffs.append((c,m, float(p['ssim_mean_mean'])-float(baseline['ssim_mean_mean']), percent(p,'macro_f1')-percent(baseline,'macro_f1')))
    discussion = ['# Thảo luận kết quả GTSRB', '',
        '## Phạm vi thực nghiệm', '',
        f"Giao thức A dùng ba checkpoint CNN cố định, {len(selection['sample_ids'])} crop validation để chọn tham số và {len(protocol['sample_ids'])} ảnh test chung. Test subset phân tầng 43 lớp, 10 ảnh/lớp. Không dùng các điểm test để thay pipeline đã chọn.", '',
        'Dataset nguồn có 39.209 ảnh final train và 12.630 ảnh final test. Một track train lớp 14 chứa 8 bản sao pixel của ảnh test. Pipeline giữ toàn bộ dữ liệu nguồn và cách ly 30 ảnh trong track khỏi train/validation. Tập học thực tế có 31.350 train và 7.829 validation.', '',
        '## RQ1: Ảnh hưởng của thiếu sáng', '',
        f"Trên cùng subset test, clean đạt Accuracy {percent(clean,'accuracy'):.2f}% và Macro-F1 {percent(clean,'macro_f1'):.2f}%. Các giá trị là trung bình qua training seed."]
    for c in levels:
        p=points[c,'identity']
        discussion.append(f"- {c}: identity đạt Accuracy {percent(p,'accuracy'):.2f}%, Macro-F1 {percent(p,'macro_f1'):.2f}%; F1 chênh {percent(p,'macro_f1')-percent(clean,'macro_f1'):+.2f} điểm phần trăm so clean.")
    discussion += ['', '## RQ2 và RQ5: Phương pháp và quy trình đã chọn', '',
        f"Validation chọn **{selected}** theo Macro-F1 trung bình, ưu tiên latency khi kém phương án cao nhất dưới 0,5 điểm phần trăm. Trên test subset, F1 trung bình ngang trọng số ba mức của pipeline này là **{f1[selected]:.2f}%**, chênh **{delta_selected:+.2f} điểm phần trăm** so identity.", '',
        f"Phương pháp có F1 trung bình test cao nhất trong bảng là {best_test} ({f1[best_test]:.2f}%). Đây là mô tả hậu nghiệm, không thay thế quyết định validation. Các cấu hình được khảo sát không chứng minh tối ưu trên toàn không gian tham số.", '',
        '## RQ3: Chất lượng ảnh và nhận diện', '']
    if tradeoffs:
        discussion += ['Các trường hợp SSIM tăng nhưng Macro-F1 giảm so identity cùng mức:', '', '| Mức | Phương pháp | Delta SSIM | Delta F1 (điểm %) |', '|---|---|---:|---:|']
        discussion += [f'| {c} | {m} | {ssim:+.4f} | {df1:+.2f} |' for c,m,ssim,df1 in tradeoffs]
    else:
        discussion += ['Không có hàng tổng hợp nào vừa tăng SSIM vừa giảm Macro-F1 so identity trong lần chạy này. Điều này không chứng minh hai chỉ số luôn đồng biến trên ảnh riêng lẻ hoặc tập dữ liệu khác.']
    discussion += ['', '## RQ4: Tương quan', '',
        'Xem `correlations.csv` cho Pearson/Spearman theo từng mức, mỗi phương pháp là một điểm, n=8. Clean bị loại vì PSNR vô hạn. Đây là phân tích khám phá với các phương pháp dùng chung ảnh, không coi 8 điểm là các thí nghiệm độc lập và không suy quan hệ nhân quả.', '',
        '## Độ bất định và giới hạn', '',
        '- Noise seed được trung bình trong từng training seed, sau đó tính mean và sample SD qua ba training seed. Các tổ hợp training/noise không phải 9 lần lặp độc lập.',
        '- `bootstrap.json` có CI 95% paired bootstrap 1.000 lần. Mỗi CI có điều kiện trên một checkpoint, dùng cùng source resampling cho ba noise seed. Không lấy trung bình các đầu mút CI để giả lập CI tổng.',
        '- Tuning dùng 1.290 ảnh từ 43 track validation. Số nhóm nhỏ làm quyết định tham số có thể nhạy với cách chọn track. Test 430 ảnh giúp giảm chi phí nhưng không đại diện điểm trên toàn official test.',
        '- Track official test không được cung cấp. Bootstrap theo source_id không xử lý được mọi tương quan giữa các ảnh cùng biển nhưng không có metadata nhóm.',
        '- Dữ liệu có thiếu sáng mô phỏng trên sRGB crop 64×64. Mô hình không phát hiện biển trong toàn cảnh và không chứng minh hoạt động trong mọi điều kiện camera đêm.',
        '- Pretrained enhancer học ở phân phối khác GTSRB. Các mô hình nhỏ/chữ số có thể bị làm mượt hoặc biến đổi, cần xem `cases.csv` và ảnh minh họa thay vì chỉ nhìn độ sáng.', '',
        '## Nguồn và tái lập', '',
        '- GTSRB: Stallkamp và cộng sự, Neural Networks 2012. https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf',
        '- Zero-DCE: Guo và cộng sự, CVPR 2020. https://github.com/Li-Chongyi/Zero-DCE',
        '- Retinexformer: Cai và cộng sự, ICCV 2023. https://github.com/caiyuanhao1998/Retinexformer',
        '- Wavelet Diffusion: Jiang và cộng sự, ACM TOG 2023. https://arxiv.org/abs/2306.00306',
        '- MSR: Jobson và cộng sự. https://ntrs.nasa.gov/api/citations/19990005051/downloads/19990005051.pdf',
        '- Toàn bộ cấu hình/hash nằm trong `protocol.json`, trạng thái trong `runs.json`, dự đoán từng ảnh trong `predictions/`. Lệnh chạy trong README của dự án.']
    (output/'discussion.md').write_text('\n'.join(discussion)+'\n')
    slides=[]
    def slide(title,content):
        slides.append(f'<section><h1>{html.escape(title)}</h1>{content}</section>')
    slide('Tăng cường thiếu sáng cho nhận diện biển báo',
          '<p class="lead">So sánh bảy phương pháp trên GTSRB với cùng CNN</p><p>Thí nghiệm tái lập trên crop RGB 64×64</p><p class="small">Kết quả trên subset test 430 ảnh, 43 lớp</p>')
    slide('Dữ liệu và chống rò rỉ',table(['Phần dữ liệu','Số ảnh'],[['Train dùng để học','31.350'],['Validation toàn bộ','7.829'],['Validation chọn tham số','1.290'],['Official test','12.630'],['Test dùng so sánh','430'],['Train cách ly do trùng test','30']])+'<p class="small">Split theo class + track. Giữ official test. Quarantine cả track chứa bản sao pixel.</p>')
    slide('Mô phỏng thiếu sáng', '<p class="formula">z = x<sup>γ</sup><br>p = Poisson(Pz) / P<br>y = clip(p + ε, 0, 1)</p>'+table(['Mức','Gamma','Poisson peak','Gaussian σ'],[[c,*(protocol['config']['conditions'][c][k] for k in ['gamma_dark','poisson_peak','gaussian_sigma'])] for c in levels]))
    slide('Các phương pháp so sánh',table(['Phương pháp','Cấu hình hoặc mô hình'],[['identity','Ảnh tối không xử lý'],['gamma',str(selection['enhancers']['gamma'])],['HE','Equalize Y trong YCrCb'],['CLAHE',str(selection['enhancers']['clahe'])],['MSR',str(selection['enhancers']['retinex'])],['Zero-DCE','Checkpoint tác giả, đường cong tăng cường'],['Retinexformer','LOL-v1, không dùng mean ground truth'],['Wavelet Diffusion','Metadata: LOLv2, EMA, sampler 10 bước']]))
    slide('CNN và giao thức cố định','<p>Conv 32, Conv 64, Conv 128<br>Adaptive average pooling và Linear 43 lớp</p><p>Adam, learning rate 0,001, batch 64<br>Tối đa 40 epoch, patience 7</p><p>Training seed 11, 22, 33<br>Noise seed 101, 202, 303</p><p class="small">Checkpoint chọn bằng validation gốc. Mỗi checkpoint nhận toàn bộ phương pháp. Tổng 219 lượt điều kiện trên cùng test subset.</p>')
    slide('Nhận diện theo mức suy giảm',image(output/'figures/macro_f1_levels.png','Macro-F1 theo mức')+'<p class="small">Trung bình noise trong training seed trước. Thanh sai số là SD qua training seed.</p>')
    for c in levels:
        values=[]
        for m in methods:
            p=points[c,m]
            values.append([m,f"{percent(p,'accuracy'):.2f}",f"{percent(p,'macro_f1'):.2f}",f"{float(p['psnr_mean_mean']):.2f}",f"{float(p['ssim_mean_mean']):.3f}"])
        slide(f'Kết quả mức {c}', table(['Phương pháp','Accuracy %','Macro-F1 %','PSNR dB','SSIM'],values))
    slide('SSIM và nhận diện ở mức vừa',image(output/'figures/scatter_medium_ssim_mean_macro_f1.png','Scatter SSIM và Macro-F1 mức vừa')+'<p class="small">Mỗi phương pháp là một điểm. n=8, các điểm dùng chung ảnh. Tương quan chỉ mang tính khám phá.</p>')
    score_rows = [[s['method'],f"{s['macro_f1']*100:.2f}",f"{s['latency_ms']:.2f}"] for s in selection['scores']]
    slide('Quy trình chọn bằng validation',table(['Phương pháp','Macro-F1 validation %','ms/ảnh'],score_rows)+f'<p>Đã chọn: <strong>{html.escape(selected)}</strong></p>')
    slide('Kết quả của quy trình đã chọn',f'<p class="lead">{html.escape(selected)}</p><p>Macro-F1 test trung bình ba mức: <strong>{f1[selected]:.2f}%</strong></p><p>So identity: <strong>{delta_selected:+.2f} điểm phần trăm</strong></p><p>Phương pháp có F1 test cao nhất: {html.escape(best_test)} ({f1[best_test]:.2f}%)</p><p class="small">Điểm test không được dùng để thay quyết định validation.</p>')
    intervals=json.loads((output/'bootstrap.json').read_text())
    slide('Paired bootstrap cho chênh lệch Macro-F1',table(['Mức','CNN seed','Delta điểm %','CI 95%'],[[r['condition'],r['train_seed'],f"{r['macro_f1']['delta']:+.2f}",f"[{r['macro_f1']['ci95'][0]:+.2f}, {r['macro_f1']['ci95'][1]:+.2f}]"] for r in intervals])+'<p class="small">1.000 lần lấy lại mẫu theo source. CI có điều kiện trên từng checkpoint. Không thay thế biến thiên huấn luyện.</p>')
    slide('Giới hạn kết luận','<p>Thiếu sáng mô phỏng trên sRGB 64×64</p><p>Tuning trên 43 track validation, test 430 ảnh</p><p>Không có track official test để bootstrap theo biển thật</p><p>Pretrained enhancer khác phân phối GTSRB</p><p class="small">Kết quả không chứng minh hiệu năng phát hiện biển toàn cảnh hoặc camera ngoài đường ban đêm.</p>')
    slide('Tái lập và tài liệu tham khảo','<p>README.md chứa lệnh chạy.<br>protocol.json khóa config, sample ID và checkpoint hash.<br>predictions/ lưu đầu ra từng ảnh.</p><p class="small">GTSRB: Stallkamp và cộng sự, Neural Networks 2012<br>Zero-DCE: Guo và cộng sự, CVPR 2020<br>Retinexformer: Cai và cộng sự, ICCV 2023<br>Wavelet Diffusion: Jiang và cộng sự, ACM TOG 2023</p><p class="small">Nguồn đầy đủ nằm trong discussion.md và repository tác giả.</p>')
    css='''*{box-sizing:border-box}body{margin:0;background:#e6e9ee;color:#172536;font-family:Arial,sans-serif}section{width:1280px;height:720px;margin:30px auto;background:white;padding:52px 68px;overflow:hidden;position:relative}h1{font-size:44px;line-height:1.15;margin:0 0 32px;color:#173e66;font-weight:700}p{font-size:28px;line-height:1.45;margin:24px 0}.lead{font-size:56px;max-width:1000px}.small{font-size:20px;color:#526272;line-height:1.4}table{width:100%;border-collapse:collapse;font-size:23px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #dce2e8}.compact td,.compact th{padding:7px 10px}th{color:#173e66}td:first-child{font-weight:600}img{display:block;max-width:100%;height:440px;object-fit:contain;margin:auto}.formula{font-size:30px;line-height:1.35;margin:12px 0 24px}nav{position:fixed;bottom:12px;right:18px;z-index:2}button{font:18px Arial;padding:10px 18px;cursor:pointer}footer{position:absolute;bottom:20px;right:32px;font-size:17px;color:#65778a}@media print{@page{size:13.333in 7.5in;margin:0}body{background:white}section{margin:0;page-break-after:always}nav{display:none}}'''
    numbered=[s.replace('</section>',f'<footer>{i+1} / {len(slides)}</footer></section>') for i,s in enumerate(slides)]
    js="""const slides=[...document.querySelectorAll('section')];let current=0;function go(delta){current=Math.max(0,Math.min(slides.length-1,current+delta));slides[current].scrollIntoView({behavior:'smooth'});}document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(-1)}});"""
    document='<!doctype html><html lang="vi"><meta charset="utf-8"><title>GTSRB low-light</title><style>'+css+'</style><body>'+''.join(numbered)+'<nav aria-label="Chuyển slide"><button onclick="go(-1)" aria-label="Slide trước">Trước</button><button onclick="go(1)" aria-label="Slide sau">Sau</button></nav><script>'+js+'</script></body></html>'
    (output/'slides.html').write_text(document)
    print(output/'discussion.md')
    print(output/'slides.html')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',nargs='?',default='outputs/test')
    build(parser.parse_args().output)
