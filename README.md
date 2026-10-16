# Tăng cường ảnh thiếu sáng cho phân loại biển báo giao thông

Bài tập lớn môn xử lý ảnh: so sánh bảy phương pháp tăng sáng trên GTSRB và kiểm chứng khả năng giữ thông tin biển báo bằng **ResNet-18 train clean, khóa checkpoint**. Đầu vào là crop biển báo, đầu ra là một trong 43 lớp; chưa có detector hoặc video.

Đọc [báo cáo nghiên cứu](RESEARCH_REPORT.md), mở [15 slide offline](outputs/resnet18_A/test_full/slides.html) hoặc [demo 43 lớp](outputs/research_evidence/demo/index.html). Giao thức và nguồn dữ liệu nằm trong [PROJECT_GUIDE.md](PROJECT_GUIDE.md), [DATASETS.md](DATASETS.md).

Nhánh `main` tập trung vào xử lý ảnh. Code và kết quả train thích nghi được giữ riêng trên `experiment/mixed-lowlight`. Lịch sử CNN nhỏ và các kế hoạch triển khai cũ có trong Git; không dùng bảng CNN cũ để so paired với ResNet full-test.

## Notebook pipeline hoàn chỉnh

[GTSRB_LLIE_Complete_Pipeline.ipynb](notebooks/GTSRB_LLIE_Complete_Pipeline.ipynb) trình bày bằng tiếng Việt từ raw archive, ROI/split, mô phỏng tối/nhiễu đến histogram, Gamma/HE/CLAHE/MSR, đánh giá chất lượng, ablation và demo. ResNet chỉ kiểm chứng nhận diện: train mới ba seed clean 11/22/33 rồi giữ cố định khi so phương pháp xử lý ảnh.

Notebook clone repo GitHub ở commit `82a7c2e`, rồi gọi code có sẵn. Bản nguồn gồm 22 cell như file mẫu nhưng chỉ **30 dòng code**; [notebook_helpers.py](notebooks/notebook_helpers.py) đi kèm gom phần setup, vẽ hình, kiểm chứng và vòng lặp ablation/demo. Thuật toán xử lý ảnh và training vẫn dùng `src/` từ repo clone. Kaggle version 4 đã hoàn tất train mới ba seed clean; [notebook có đầy đủ output](notebooks/GTSRB_LLIE_Complete_Pipeline_Executed.ipynb) đã tải về và kiểm chứng ngày 14/10/2026.

Khi chạy trên [Kaggle](https://www.kaggle.com/code/sangtranminh211/gtsrb-llie-complete-pipeline-20261014), bật GPU T4, Internet và hai dataset private `gtsrb-resnet18-audited-experiment-20261007`, `gtsrb-llie-notebook-inputs-20261014`; file helper phải có trong bundle đầu vào. Ma trận mới gồm 219 lượt × 430 ảnh test và ablation 288 lượt × 430 ảnh; bảng full-test cũ bên dưới thuộc run riêng.

Log thu kết quả nằm trong `outputs/notebook_kaggle/collector_v4.log`. Artifacts đã lưu riêng trong `outputs/notebook_kaggle/results_v4/`; receipt tại `outputs/notebook_kaggle/download_receipt.json` xác nhận tải và kiểm chứng thành công.

## Pipeline và dữ liệu

```text
Ảnh gốc → crop RGB 64×64 → gamma + Poisson + Gaussian
       → identity hoặc enhancer → PSNR/SSIM so ảnh gốc
       → ResNet-18 cố định → Accuracy/Macro-F1/Weighted-F1
```

ResNet nhận toàn crop resize lên 224×224 và normalize ImageNet đúng một lần. Enhancer và metric ảnh làm việc ở 64×64. Tham số enhancer và pipeline được chọn trên validation trước test.

| Split | Số ảnh | Vai trò |
|---|---:|---|
| Train | 31.350 | Train classifier trên ảnh gốc |
| Validation | 7.829 | Chọn checkpoint |
| Quarantine | 30 | Cách ly track chứa 8 ảnh trùng pixel với test |
| Official test | 12.630 | Đánh giá cuối |
| Validation chọn enhancer | 1.290 | Chọn tham số và phương pháp |
| Subset test/ablation | 430 | 10 ảnh mỗi lớp, tách bảng full-test |

Train/validation chia theo class + track với seed 42; official test giữ nguyên. Padding ROI 5%, giữ cả biển nhỏ. Manifest và checksum nằm trong `data/manifests/`.

## Các phép xử lý ảnh

| Tên trong code | Phương pháp |
|---|---|
| `identity` | Giữ nguyên ảnh tối, mốc so sánh |
| `gamma` | Biến đổi cường độ bằng hàm lũy thừa |
| `he` | Cân bằng histogram trên kênh độ sáng Y |
| `clahe` | Cân bằng histogram cục bộ, giới hạn tương phản |
| `retinex` | Multi-Scale Retinex giản lược |
| `zero_dce` | Zero-DCE pretrained |
| `retinexformer` | Retinexformer pretrained |
| `diffusion` | Wavelet-based Diffusion LLIE pretrained |

`clean` là ảnh gốc tham chiếu. Ba enhancer học sâu dùng weights tác giả, không train lại trên GTSRB.

| Mức | Gamma làm tối | Poisson peak | Gaussian sigma |
|---|---:|---:|---:|
| mild | 1,5 | 120 | 0,005 |
| medium | 2,2 | 60 | 0,010 |
| severe | 3,0 | 30 | 0,020 |

Thứ tự suy giảm: gamma → Poisson → Gaussian → clip [0,1]. Noise seed 101/202/303; training seed 11/22/33. Mọi enhancer nhận cùng ảnh tối cho cùng ID/mức/noise seed. Ma trận chính có 219 lượt: 3 clean + 3 training seed × 3 mức × 3 noise seed × 8 phương pháp. Trung bình noise trong mỗi training seed trước, rồi tính mean và sample SD qua ba training seed.

## Kết quả chính đã kiểm chứng

Validation clean: Accuracy **99,57% ± 0,21**, Macro-F1 **99,47% ± 0,31**, Weighted-F1 **99,57% ± 0,21**. Class 19 đạt F1=1 ở cả ba seed, support 30/seed. Checkpoint tốt nhất ở epoch 12/10/3; dừng train ở epoch 17/14/8 cho seed 11/22/33.

Full-test clean: Accuracy **99,15% ± 0,30**, Macro-F1 **98,90% ± 0,40**, Weighted-F1 **99,13% ± 0,32**.

Macro-F1 mean ± sample SD (%) trên **12.630 ảnh**, trung bình noise trong mỗi training seed trước:

| Phương pháp | Tối nhẹ | Tối vừa | Tối nặng |
|---|---:|---:|---:|
| identity | 92.00 ± 1.71 | 78.04 ± 1.79 | 59.70 ± 1.64 |
| gamma | 93.49 ± 1.25 | 79.77 ± 1.85 | 60.48 ± 2.59 |
| he | 92.31 ± 1.96 | 77.27 ± 3.65 | 56.98 ± 6.48 |
| clahe | 94.35 ± 1.12 | 82.80 ± 1.46 | 64.31 ± 3.47 |
| retinex | 82.70 ± 3.12 | 55.71 ± 5.31 | 34.72 ± 5.71 |
| zero_dce | 89.65 ± 2.37 | 72.41 ± 3.12 | 51.27 ± 4.60 |
| retinexformer | 82.85 ± 1.80 | 76.28 ± 2.31 | 56.67 ± 1.15 |
| diffusion | 93.45 ± 1.80 | 83.87 ± 2.97 | 64.88 ± 3.84 |

Validation chọn **diffusion** trước test. Trên full test, diffusion tăng Macro-F1 so identity lần lượt **1,45; 5,83; 5,18 điểm phần trăm** ở mild/medium/severe. CLAHE có F1 test mild cao hơn; đây là quan sát test, không đổi pipeline đã chọn. Diffusion khoảng 226–230 ms/ảnh trên Tesla T4, CLAHE khoảng 0,4 ms/ảnh; đây chỉ là latency enhancer.

Mục tiêu cả Accuracy/Macro-F1/Weighted-F1 >90% đạt với clean và pipeline đã chọn ở mild; chưa đạt ở medium/severe. Không so trực tiếp điểm full-test ResNet với bảng CNN subset cũ như cùng benchmark. Báo cáo đầy đủ: [report.md](outputs/resnet18_A/test_full/report.md), [aggregate.csv](outputs/resnet18_A/test_full/aggregate.csv). Archive và SHA256: `outputs/resnet18_A/download_receipt.json`.

## Ablation và demo

[Job ablation/demo](https://www.kaggle.com/code/sangtranminh211/gtsrb-resnet18-ablation-and-demo-20261012) đã hoàn tất và kiểm chứng 288 lượt × 430 ảnh. Bốn biến thể ở mức vừa là gamma-only, gamma + Gaussian, gamma + Poisson và gamma + cả hai loại nhiễu. Giữ classifier clean và tham số enhancer đã khóa; đây là phân tích phụ trên subset.

MSR đạt Macro-F1 98,20% với gamma-only, giảm còn 50,97% khi thêm cả hai loại nhiễu. Xem [bảng ablation](outputs/research_evidence/ablation_report.md) và [kiểm chứng](outputs/research_evidence/verification.json).

[Demo offline](outputs/research_evidence/demo/index.html) có 43 lớp và 1.075 ảnh, dùng dự đoán thật của classifier clean seed 11 trên validation. Giữ thư mục `images/` cạnh `index.html`. Đây là minh họa, không thay bảng đánh giá test. [Raw replay](outputs/replay/verification.json) đã đối chiếu chính xác 43 crop và ba mức suy giảm từ ảnh nguồn.

## Cấu trúc source

```text
configs/resnet18.json       Cấu hình thí nghiệm chính; CLI mặc định
src/                       Dữ liệu, suy giảm, enhancer, classifier, metric và báo cáo
scripts/run.py             CLI chuẩn bị/train/tune/freeze/evaluate/report
scripts/setup_external.py  Mã và weights tác giả đã pin
scripts/*kaggle.py          Các job GPU có thể chạy lại
scripts/verify_results.py   Kiểm hash/ID/nhãn và tính lại metric
scripts/finalize_*.py       Tổng hợp ablation/demo, discussion và slide
scripts/write_research_report.py  Tạo báo cáo chính từ artifacts đã kiểm chứng
checks/                    Kiểm tra kỹ thuật và workflow
checks/fixtures/cnn.json    CNN nhỏ dùng cho test nhanh, không phải cấu hình bài báo
```

`data/`, `external/`, `outputs/` và `.venv/` không đưa vào Git. Clone repo không tự có dataset, checkpoint hoặc ảnh demo. `requirements-lock.txt` giữ môi trường local đã ghi nhận; không cài bộ torch macOS đè lên CUDA của Kaggle.

## Cài đặt và chuẩn bị

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.setup_external
python -m scripts.run download
python -m scripts.run prepare
python -m scripts.run check-data
```

`prepare` bỏ qua khi manifest đã tồn tại. Đối chiếu provenance tại `external/sources.json` và `data/manifests/gtsrb_sources.json`. Khi cần rà dữ liệu gần trùng, chạy `python -m scripts.review_near_duplicates`. `python -m scripts.summarize_degradation` tổng hợp các cache validation identity đã hoàn tất; cần có cache đủ ba mức và ba noise seed.

## Kiểm tra và tái lập

```bash
python -m checks.check_pipeline
python -m checks.check_resnet18
python -m checks.check_workflow
python -m checks.check_full_test_staging
python -m checks.check_finalization
python -m checks.check_raw_replay
python -m checks.check_notebook
python -m scripts.verify_results outputs/resnet18_A/test_full
```

Các test training/workflow local dùng dữ liệu tổng hợp trong thư mục tạm, không phải kết quả nghiên cứu. Training và các thí nghiệm thật chạy trên Kaggle:

- [ResNet-A v3](https://www.kaggle.com/code/sangtranminh211/gtsrb-resnet18-protocol-a-20261007): train ba seed và subset 430 ảnh.
- [Full-test v4](https://www.kaggle.com/code/sangtranminh211/gtsrb-resnet18-full-test-12630): chỉ evaluate checkpoint/tham số đã khóa trên 12.630 ảnh.

CLI mặc định đọc `configs/resnet18.json`. Khi tạo một thí nghiệm clean mới trên môi trường GPU, dùng output riêng, theo thứ tự:

```bash
python -m scripts.run --device cuda train --seeds 11 22 33 --output outputs/resnet18_new/train
python -m scripts.run --device cuda tune --checkpoints outputs/resnet18_new/train/cnn_11.pt outputs/resnet18_new/train/cnn_22.pt outputs/resnet18_new/train/cnn_33.pt --per-class 30 --output outputs/resnet18_new/validation
python -m scripts.run freeze --selection outputs/resnet18_new/validation/selection.json --output outputs/resnet18_new/protocol.json
python -m scripts.run --device cuda evaluate --protocol outputs/resnet18_new/protocol.json --output outputs/resnet18_new/test
python -m scripts.run report --output outputs/resnet18_new/test
```

Không thay code/config/checkpoint giữa tune, freeze và evaluate. Source hiện tại giữ hash của snapshot ResNet-A đã đánh giá; snapshot gốc còn ở `outputs/resnet18_A/source_snapshot/`. Không ghi đè artifacts đã khóa để thử cấu hình mới.

Tạo lại tài liệu từ bộ kết quả đã kiểm chứng:

```bash
python -m scripts.finalize_evidence
python -m scripts.finalize_research outputs/resnet18_A/test_full
python -m scripts.write_research_report
```

## Đọc artifacts

| Đường dẫn | Nội dung |
|---|---|
| `outputs/resnet18_A/train/` | Ba checkpoint, log và validation |
| `outputs/resnet18_A/validation/selection.json` | Tham số/pipeline chọn trước test |
| `outputs/resnet18_A/test_full/protocol.json` | Config, ID và hashes đã khóa |
| `outputs/resnet18_A/test_full/aggregate.csv` | Mean/SD cho các metric và latency |
| `outputs/resnet18_A/test_full/predictions/` | Dự đoán từng ảnh |
| `outputs/resnet18_A/test_full/figures/` | Biểu đồ, confusion matrix, ví dụ lỗi |
| `outputs/resnet18_A/test_full/verification.json` | Kiểm chứng 219 lượt × 12.630 ảnh |
| `outputs/resnet18_A/download_receipt.json` | Archive và SHA256 |
| `outputs/research_evidence/` | Ablation và demo validation |

Kết luận hiện áp dụng cho thiếu sáng mô phỏng trên crop GTSRB, chưa chứng minh camera đêm, mưa, lóa hoặc motion blur. Chất lượng pixel cao không tự bảo đảm nhận diện tốt. Nguồn dataset, mã và weights có điều kiện sử dụng riêng; đối chiếu trước khi phân phối lại.
