"""Write the seven-section research report from verified artifacts only."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean

from src.common import read_csv


def table(rows):
    lines = ['| Mức | Phương pháp | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) |',
             '|---|---|---:|---:|---:|']
    for r in rows:
        cells = [f"{100*float(r[k+'_mean']):.2f} ± {100*float(r[k+'_std']):.2f}"
                 for k in ['accuracy', 'macro_f1', 'weighted_f1']]
        lines.append('| '+r['condition']+' | '+r['method']+' | '+' | '.join(cells)+' |')
    return lines


def build(root=Path('.')):
    root = Path(root)
    output = root / 'outputs/resnet18_A/test_full'
    verified = json.loads((output / 'verification.json').read_text())
    assert verified['status'] == 'verified' and verified['runs'] == 219 and verified['images_per_run'] == 12630
    protocol = json.loads((output / 'protocol.json').read_text())
    selected = protocol['selection']['chosen_method']
    aggregate = read_csv(output / 'aggregate.csv')
    points = {(r['condition'], r['method']): r for r in aggregate}
    mean_f1 = lambda c, m: 100*float(points[c, m]['macro_f1_mean'])
    lines = ['# Tăng cường ảnh thiếu sáng cho phân loại biển báo giao thông', '',
             'Báo cáo thực nghiệm GTSRB. Kết quả chính dùng classifier ResNet-18 train clean, '
             'ba seed và toàn bộ official test. Trọng tâm là so sánh xử lý ảnh với classifier cố định.', '',
             '## 1. Bài toán và câu hỏi nghiên cứu', '',
             'Đầu vào là crop biển báo RGB, đầu ra là một trong 43 lớp. Câu hỏi chính là ảnh tăng sáng có '
             'giúp classifier nhận đúng hơn so với cùng ảnh tối chưa xử lý hay không. Phạm vi này là phân loại crop, '
             'chưa bao gồm phát hiện vị trí biển báo trong toàn cảnh hoặc video.', '',
             'RQ1: thiếu sáng và nhiễu làm giảm nhận diện bao nhiêu? RQ2: enhancer nào tốt trong cấu hình đã khảo sát? '
             'RQ3: PSNR/SSIM cao có đi cùng nhận diện tốt không? RQ4: tương quan thay đổi theo mức tối như thế nào? '
             'RQ5: pipeline chọn bằng validation có lợi ích và chi phí gì trên test?', '',
             '## 2. Cơ sở xử lý ảnh và nghiên cứu liên quan', '',
             'Gamma điều chỉnh giá trị pixel bằng hàm lũy thừa. HE cân bằng histogram kênh độ sáng Y trong YCrCb. '
             'CLAHE xử lý histogram theo ô cục bộ và giới hạn tương phản. Bản MSR trong dự án lấy trung bình '
             'chênh lệch log giữa ảnh và ảnh làm mờ ở nhiều sigma, sau đó chuẩn hóa theo percentile 1–99. '
             'Đây là cài đặt MSR đơn giản, không có bước color restoration của MSRCR.', '',
             'Zero-DCE ước lượng đường cong tăng sáng bằng một mạng nhỏ, dùng các loss không cần ảnh tham chiếu khi huấn luyện. '
             '[Guo và cộng sự, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html).', '',
             'Retinexformer sử dụng transformer dựa trên mô hình Retinex để phục hồi ảnh thiếu sáng. '
             '[Cai và cộng sự, ICCV 2023](https://github.com/caiyuanhao1998/Retinexformer). '
             'Wavelet-based Diffusion LLIE kết hợp diffusion có điều kiện trong miền wavelet và phục hồi thành phần tần số cao. '
             '[Jiang và cộng sự, ACM TOG 2023](https://arxiv.org/abs/2306.00306). '
             'Dự án dùng weights tác giả, không train lại các enhancer trên GTSRB.', '',
             '## 3. Dữ liệu và giao thức', '',
             'GTSRB gồm 39.209 ảnh final training và 12.630 ảnh official test. '
             'Dự án chia train/validation theo class và track: 31.350 train, 7.829 validation. '
             'Có 8 bản sao pixel chéo train–test trong một track lớp 14, nên cách ly cả track gồm 30 ảnh. '
             '[Thông tin benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html). '
             'Số mẫu và cách ly được kiểm kê từ manifest/audit của dự án.', '',
             'Cắt ROI có padding cố định 5%, quy ước endpoint ghi trong manifest, resize RGB 64×64. '
             'Ảnh tối được tạo theo thứ tự gamma, Poisson, Gaussian rồi clip về [0,1].', '',
             '| Mức | Gamma làm tối | Poisson peak | Gaussian sigma |', '|---|---:|---:|---:|']
    for c, v in protocol['config']['conditions'].items():
        lines.append(f"| {c} | {v['gamma_dark']} | {v['poisson_peak']} | {v['gaussian_sigma']} |")
    lines += ['', 'Train seed 11/22/33, noise seed 101/202/303. Chọn tham số và phương pháp bằng '
              '1.290 ảnh validation trước test. Khóa checkpoint, config, sample ID, source hash và provenance. '
              'Mọi phương pháp dùng chung ID và ảnh tối cho từng điều kiện. Clean có ba lượt, không nhân bản theo noise seed.', '',
              'Noise được trung bình trong từng training seed trước, rồi tính mean và sample SD qua ba training seed. '
              'Các tổ hợp training/noise không phải chín lần huấn luyện độc lập.', '',
              '## 4. Cài đặt và kiểm chứng', '',
              'ResNet-18 khởi tạo từ ImageNet, fine-tune toàn mạng với AdamW, lr 1e-4, weight decay 1e-4, '
              'batch 32, tối đa 20 epoch. Early stopping dùng patience 5, min_delta 0.001. '
              'Classifier resize toàn crop lên 224×224 rồi normalize ImageNet đúng một lần. Enhancer và metric ảnh vẫn ở 64×64.', '',
              'Checkpoint clean tốt nhất ở epoch 12/10/3 cho seed 11/22/33. GPU là Tesla T4, '
              'torch 2.11.0+cu128 và torchvision 0.26.0+cu128 theo artifact môi trường. '
              'Load checkpoint offline không tải lại ImageNet weights.', '',
              'Archive đã kiểm CRC và SHA256. Đã đối chiếu ID/nhãn và tính lại ba metric nhận diện từ '
              '2.765.970 dự đoán. Tái tạo 43 crop validation từ raw cho pixel khớp chính xác với canonical PNG, '
              'và tái tạo cả ba mức suy giảm với float32 khớp chính xác. '
              '[Kiểm chứng test](outputs/resnet18_A/test_full/verification.json), '
              '[kiểm chứng raw replay](outputs/replay/verification.json).', '',
              '## 5. Kết quả thực nghiệm', '', '### 5.1. ResNet-18 clean, full official test', '']
    lines += table(aggregate)
    lines += ['', 'PSNR/SSIM và latency đầy đủ nằm trong '
              '[bảng tổng hợp](outputs/resnet18_A/test_full/aggregate.csv). PSNR của clean là vô hạn, SSIM là 1. '
              'Latency enhancer đo batch 1 với warm-up và đồng bộ GPU, gồm transfer/inference nhưng loại I/O và metric. '
              'Classifier đo riêng batch 32, khoảng 0,85–1,02 ms/ảnh cho ba seed. Không gọi latency enhancer là thời gian toàn pipeline.', '',
              '![Macro-F1 theo mức](outputs/resnet18_A/test_full/figures/macro_f1_levels.png)', '',
              f'Validation chọn **{selected}**. Macro-F1 của pipeline này ở mild/medium/severe lần lượt '
              + '/'.join(f'{mean_f1(c, selected):.2f}%' for c in ['mild','medium','severe'])+'. '
              'Delta so identity lần lượt '+ '/'.join(f"{mean_f1(c, selected)-mean_f1(c, 'identity'):+.2f}" for c in ['mild','medium','severe'])
              +' điểm phần trăm. Clean và mild đạt cả ba metric >90%, medium/severe chưa đạt.', '',
              '### 5.2. Ablation gamma và nhiễu', '']
    evidence = root / 'outputs/research_evidence'
    if (evidence / 'verification.json').exists():
        ev = json.loads((evidence / 'verification.json').read_text())
        assert ev['status'] == 'verified' and ev['ablation_runs'] == 288
        lines += (evidence / 'ablation_report.md').read_text().splitlines()[2:]
    else:
        lines.append('Job supplementary đang chạy, chưa dùng log làm số liệu báo cáo.')
    lines += ['', '## 6. Tương quan và phân tích lỗi', '',
              '| Mức | Chất lượng ảnh | Pearson với Macro-F1 | Spearman với Macro-F1 | n |',
              '|---|---|---:|---:|---:|']
    for r in read_csv(output / 'correlations.csv'):
        if r['recognition'] == 'macro_f1':
            lines.append(f"| {r['condition']} | {r['quality']} | {float(r['pearson']):.3f} | {float(r['spearman']):.3f} | {r['n']} |")
    lines += ['', 'Mỗi phương pháp là một điểm, n=8, các điểm phụ thuộc vì dùng chung ảnh. '
              'Các tương quan là phân tích khám phá, không chứng minh quan hệ nhân quả. '
              'Ví dụ medium: Zero-DCE có PSNR 19,98 dB nhưng Macro-F1 72,41%, '
              'so diffusion 14,77 dB và 83,87%. Metric pixel cao không tự bảo đảm nhận diện tốt.', '',
              '![SSIM và Macro-F1 mức vừa](outputs/resnet18_A/test_full/figures/scatter_medium_ssim_mean_macro_f1.png)', '',
              '| Mức | Được giúp | Bị làm sai | Vẫn sai | Vẫn đúng |', '|---|---:|---:|---:|---:|']
    counts = Counter((r['condition'], r['category']) for r in read_csv(output / 'cases.csv'))
    for condition in ['mild','medium','severe']:
        lines.append('| '+condition+' | '+' | '.join(str(counts[condition,k]) for k in ['helped','harmed','still_wrong','still_correct'])+' |')
    lines += ['', 'Số đếm trên chín tổ hợp training/noise mỗi mức, dùng để mô tả các lần dự đoán, '
              'không coi là số ảnh độc lập. Ví dụ chọn bằng sample ID cố định trong `example_ids.json`.', '',
              '![Ví dụ giúp, làm hỏng, vẫn sai](outputs/resnet18_A/test_full/figures/failure_examples.png)', '']
    class_scores, supports = defaultdict(list), {}
    for row in read_csv(output / 'per_class.csv'):
        if row['condition'] == 'severe' and row['method'] in {'identity', selected}:
            label = int(row['class_id'])
            class_scores[label, row['method']].append(float(row['f1']))
            assert label not in supports or supports[label] == int(row['support'])
            supports[label] = int(row['support'])
    assert len(supports) == 43 and all(len(v) == 9 for v in class_scores.values())
    for method in {'identity', selected}:
        assert math.isclose(mean(mean(class_scores[k, method]) for k in supports),
                            float(points['severe', method]['macro_f1_mean']), abs_tol=1e-12)
    lines += ['Năm lớp có F1 thấp nhất của pipeline đã chọn ở severe (mean qua ba noise và ba training seed):', '',
              f'| Class | Support test | F1 identity (%) | F1 {selected} (%) |', '|---|---:|---:|---:|']
    for label in sorted(supports, key=lambda k: mean(class_scores[k, selected]))[:5]:
        lines.append(f'| {label} | {supports[label]} | {100*mean(class_scores[label,"identity"]):.2f} | '
                     f'{100*mean(class_scores[label,selected]):.2f} |')
    mistakes = Counter()
    for row in read_csv(output / 'summary.csv'):
        if row['condition'] == 'severe' and row['method'] == selected:
            mistakes.update((int(p['class_id']), int(p['prediction'])) for p in
                            read_csv(output / 'predictions' / f'{row["run_id"]}.csv')
                            if p['class_id'] != p['prediction'])
    lines += ['', 'Năm cặp nhầm lẫn thường gặp nhất của pipeline ở severe:', '',
              '| Nhãn thật | Dự đoán sai | Số lần dự đoán |', '|---:|---:|---:|']
    lines += [f'| {a} | {b} | {n} |' for (a,b),n in mistakes.most_common(5)]
    lines += ['', 'Support là số ảnh test của lớp, không nhân theo seed. Số nhầm lẫn cộng qua chín '
              'tổ hợp training/noise và không phải số ảnh độc lập.', '',
              'Class 19 clean validation có F1=1, support 30 ở cả ba seed. Cần xem '
              '[per_class.csv](outputs/resnet18_A/test_full/per_class.csv) cho các mức tối. '
              'Paired bootstrap 1.000 lần trong `bootstrap.json` có điều kiện trên từng checkpoint. '
              'Official test thiếu track xác minh, nên bootstrap theo source_id chưa xử lý được mọi tương quan cùng biển.', '',
              '## 7. Quy trình đề xuất, hạn chế và hướng phát triển', '',
              f'Trong giao thức A đã khảo sát, dùng tham số {selected} đã chọn bằng validation và classifier clean cố định. '
              'Diffusion có lợi ích nhận diện nhưng chậm khoảng 226–230 ms/ảnh; CLAHE khoảng 0,4 ms/ảnh '
              'và có Macro-F1 mild cao hơn trong quan sát test. Không đổi pipeline validation theo phương pháp thắng test.', '',
              'Kết luận chỉ áp dụng cho GTSRB crop với thiếu sáng mô phỏng trên sRGB, chưa chứng minh camera đêm thực, '
              'mưa, lóa, nhòe chuyển động hoặc detection toàn cảnh. Thiếu sáng nặng có thể mất thông tin không phục hồi được. '
              'Các enhancer pretrained khác phân phối GTSRB. CNN cũ chỉ có bảng subset và thiếu artifact tương ứng '
              'trong workspace, nên không dùng bảng đó để kết luận paired cải thiện của ResNet full-test.', '',
              'Hướng tiếp theo là phân tích tác động của xử lý ảnh lên nhiễu, màu sắc và chi tiết biển báo, sau đó kiểm tra dữ liệu đêm thực. '
              'Mục tiêu >90% ở mọi mức là mục tiêu cần đo, không phải kết quả đã đạt.', '',
              '## Tài liệu và bàn giao', '',
              '- [Hướng dẫn chạy](README.md), [giao thức](PROJECT_GUIDE.md), [dữ liệu](DATASETS.md).',
              '- [Slides HTML offline](outputs/resnet18_A/test_full/slides.html), '
              '[demo validation 43 lớp](outputs/research_evidence/demo/index.html).',
              '- [GTSRB: Stallkamp và cộng sự, 2012](https://benchmark.ini.rub.de/gtsrb_news.html).',
              '- [Zero-DCE: Guo và cộng sự, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html).',
              '- [Retinexformer: Cai và cộng sự, ICCV 2023](https://github.com/caiyuanhao1998/Retinexformer).',
              '- [Wavelet Diffusion: Jiang và cộng sự, ACM TOG 2023](https://arxiv.org/abs/2306.00306).',
              '- [Multiscale Retinex: Jobson, Rahman và Woodell, 1997](https://ntrs.nasa.gov/api/citations/19990005051/downloads/19990005051.pdf).',
              '- Config, environment, predictions, protocol và hashes nằm cùng thư mục kết quả. '
              'Source snapshot gốc A và receipt archive nằm trong `outputs/resnet18_A/`.']
    destination = root / 'RESEARCH_REPORT.md'
    destination.write_text('\n'.join(lines)+'\n')
    print(destination)
    return destination


if __name__ == '__main__':
    build()
