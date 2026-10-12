# Tăng cường ảnh thiếu sáng cho phân loại biển báo giao thông

**Ảnh sáng hơn có giúp mô hình nhận đúng biển báo hơn không?** Dự án này tạo ảnh thiếu sáng từ bộ GTSRB, xử lý bằng bảy phương pháp tăng sáng, rồi so sánh khả năng phân loại bằng cùng một CNN đã huấn luyện.

Đây là dự án thực nghiệm xử lý ảnh và học sâu phục vụ bài tập lớn. Đầu vào là **ảnh đã cắt vùng biển báo**, đầu ra là một trong **43 lớp biển báo**. Phạm vi hiện tại là phân loại crop biển báo; chưa có chức năng tìm vị trí biển báo trong ảnh đường phố hoặc video.

## 1. Cách hệ thống hoạt động

```text
GTSRB → cắt vùng biển báo → resize RGB 64 × 64
                              │
              ┌───────────────┴───────────────────┐
              │                                   │
      Train / validation                         Test
              │                                   │
   Huấn luyện CNN với 3 seed             Tạo ảnh tối và nhiễu
              │                                   │
    Khóa trọng số từng CNN               Giữ nguyên / tăng sáng
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                     Phân loại bằng CNN đã khóa
                                │
                Đo nhận diện, chất lượng ảnh, thời gian
```

CNN học từ ảnh gốc có augmentation. Khi đánh giá tăng sáng, trọng số CNN được giữ cố định: chỉ thay phương pháp xử lý ảnh đầu vào. Cách này giúp đo tác động của bước tăng sáng trong cùng điều kiện phân loại, gọi là **giao thức A** của dự án. Ảnh gốc cũng được đánh giá làm mốc `clean`.

## 2. Dữ liệu dùng để train và đánh giá

Dự án sử dụng **GTSRB — German Traffic Sign Recognition Benchmark**, gồm 43 lớp biển báo.

| Phần dữ liệu | Số ảnh | Mục đích |
|---|---:|---|
| Train | **31.350** | Học trọng số CNN |
| Validation | **7.829** | Chọn checkpoint CNN tốt nhất |
| Cách ly (`quarantine`) | **30** | Tránh rò rỉ dữ liệu sang test |
| Official test | **12.630** | Tập test đầy đủ có sẵn |
| Subset validation đã sử dụng | **1.290** | Chọn tham số và phương pháp tăng sáng |
| Subset test đã đánh giá | **430** | 10 ảnh mỗi lớp, dùng chung cho mọi phương pháp |

Hai subset ở cuối được lấy từ validation và test tương ứng. **430 ảnh là số ảnh test đã đánh giá, không phải số ảnh train.**

Ba seed huấn luyện `11, 22, 33` dùng cùng 31.350 ảnh train, khác các yếu tố ngẫu nhiên khi huấn luyện. Không cộng ba lần chạy thành ba tập dữ liệu độc lập.

Quy trình chuẩn bị dữ liệu:

- Dùng final training gồm 39.209 ảnh; không nhầm với bản `GTSRB-Training_fixed.zip` gồm 26.640 ảnh của giai đoạn thi cũ.
- Tách train/validation theo `(class_id, track_id)`, seed chia tập `42`, để ảnh cùng chuỗi chụp không nằm ở cả hai tập.
- Phát hiện 8 ảnh của một track lớp 14 trùng pixel với test; cách ly cả track 30 ảnh.
- Cắt ROI biển báo, thêm padding 5% mỗi phía và resize về RGB `64 × 64`.
- Lưu manifest, danh sách ID, thống kê lớp và kết quả audit để đối chiếu.

Chi tiết tại [DATASETS.md](DATASETS.md).

## 3. Mô hình CNN

Đầu vào có kích thước `[batch, 3, 64, 64]`; đầu ra là logits `[batch, 43]`.

```text
Conv 3→32   → BatchNorm → ReLU → MaxPool
Conv 32→64  → BatchNorm → ReLU → MaxPool
Conv 64→128 → BatchNorm → ReLU
Adaptive Average Pooling → Flatten → Dropout(0.4) → Linear 128→43
```

| Thiết lập | Giá trị |
|---|---|
| Optimizer / loss | Adam / Cross-entropy |
| Learning rate / batch size | 0,001 / 64 |
| Epoch tối đa / patience | 40 / 7 |
| Chọn checkpoint | Validation Macro-F1 cao nhất |
| Pixel đầu vào | RGB trong `[0, 1]` |
| Augmentation khi train | Xoay ±10°, tịnh tiến tối đa 5%, ColorJitter, RandomErasing |

BatchNorm hỗ trợ ổn định quá trình học. Dropout và augmentation giúp giảm overfitting. Validation không áp dụng augmentation ngẫu nhiên của tập train.

## 4. Các phương pháp tăng sáng

| Tên trong code | Phương pháp | Vai trò |
|---|---|---|
| `clean` | Ảnh gốc | Mốc khi chưa tạo thiếu sáng |
| `identity` | Giữ nguyên ảnh tối | Mốc đo lợi ích tăng sáng |
| `gamma` | Gamma correction | Điều chỉnh độ sáng bằng hàm lũy thừa |
| `he` | Histogram Equalization | Cân bằng histogram kênh độ sáng |
| `clahe` | CLAHE | Cân bằng histogram cục bộ, giới hạn tương phản |
| `retinex` | Multi-Scale Retinex | Xử lý độ sáng ở nhiều thang đo |
| `zero_dce` | Zero-DCE | Mô hình tăng sáng pretrained |
| `retinexformer` | Retinexformer | Mô hình tăng sáng pretrained |
| `diffusion` | Wavelet-based Diffusion LLIE | Mô hình diffusion pretrained |

Có **7 phương pháp tăng sáng**, cộng với `identity` và mốc `clean`. Các enhancer học sâu dùng checkpoint có sẵn, không được huấn luyện lại trên GTSRB trong pipeline hiện tại. Zero-DCE được chọn làm baseline học sâu chính để đối chiếu trong dự án.

Ảnh thiếu sáng được tạo theo thứ tự **gamma làm tối → nhiễu Poisson → nhiễu Gaussian → giới hạn về `[0, 1]`**.

| Mức | Gamma làm tối | Poisson peak | Gaussian sigma |
|---|---:|---:|---:|
| Nhẹ (`mild`) | 1,5 | 120 | 0,005 |
| Vừa (`medium`) | 2,2 | 60 | 0,010 |
| Nặng (`severe`) | 3,0 | 30 | 0,020 |

Ba seed nhiễu là `101, 202, 303`. Mọi enhancer nhận cùng ảnh tối cho cùng ID, mức và seed nhiễu. Ảnh gốc chỉ làm tham chiếu đánh giá; enhancer nhận ảnh tối.

## 5. Kết quả thực nghiệm đã có

Bộ artifact hiện có nằm tại `outputs/low-light-outputs-A-subset/`. Đã hoàn thành **219/219 lượt đánh giá**, đủ mọi phương pháp, ba seed CNN, ba mức tối và ba seed nhiễu:

```text
3 lượt clean + 3 seed CNN × 3 mức tối × 3 seed nhiễu × 8 phương pháp = 219
```

Mỗi lượt dùng cùng 430 ảnh test. Một lượt là một cấu hình đánh giá, không phải một lần train mới.

### CNN trên dữ liệu gốc

Cả ba seed hoàn thành 40 epoch. Bảng dưới đo trên **7.829 ảnh validation** tại checkpoint tốt nhất:

| Seed | Epoch tốt nhất | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 11 | 37 | 92,11% | 87,01% |
| 22 | 39 | 92,81% | 88,57% |
| 33 | 37 | 92,43% | 87,14% |
| Trung bình | — | **92,45%** | **87,57%** |

### Phân loại sau tăng sáng

Macro-F1 trung bình (%) trên **subset test 430 ảnh**, lấy từ `test_subset/aggregate.csv`:

| Phương pháp | Tối nhẹ | Tối vừa | Tối nặng |
|---|---:|---:|---:|
| Identity | 41,80 | 19,48 | 7,23 |
| Gamma | **58,92** | **35,18** | **18,74** |
| HE | 29,49 | 16,64 | 7,71 |
| CLAHE | 38,28 | 19,81 | 7,55 |
| Multi-Scale Retinex | 36,79 | 12,84 | 8,97 |
| Zero-DCE | 49,77 | 28,95 | 16,38 |
| Retinexformer | 45,40 | 28,13 | 12,67 |
| Diffusion | 45,16 | 29,07 | 13,95 |

Trên ảnh clean của cùng subset test, CNN đạt **Accuracy 84,81% ± 1,65** và **Macro-F1 83,79% ± 1,65**; độ lệch chuẩn tính bằng điểm phần trăm giữa ba seed CNN.

**Gamma được chọn bằng validation trước khi đánh giá test** và có Macro-F1 cao nhất trong các phương pháp tăng sáng trên subset này. So với identity, Gamma tăng khoảng **17,12; 15,70; 11,51 điểm phần trăm** ở ba mức tối. Tuy nhiên, nhận diện vẫn thấp hơn ảnh clean, đặc biệt ở mức tối nặng.

Kết luận này chỉ áp dụng cho dữ liệu và giao thức đang xét; không chứng minh Gamma luôn tốt hơn mô hình học sâu trên mọi ảnh thiếu sáng.

### Hiểu các chỉ số

- **Accuracy:** tỷ lệ ảnh được phân loại đúng.
- **Macro-F1:** trung bình F1 của 43 lớp với trọng số bằng nhau, giúp phản ánh cả các lớp ít ảnh.
- **PSNR / SSIM:** độ gần với ảnh gốc về chất lượng ảnh; điểm ảnh cao không tự bảo đảm nhận diện tốt.
- **Latency:** thời gian xử lý mỗi ảnh, phụ thuộc thiết bị và môi trường chạy.

Kết quả được trung bình qua seed nhiễu trong từng seed CNN trước, rồi tính mean và độ lệch chuẩn giữa ba seed CNN. Chín tổ hợp CNN/nhiễu không được coi là chín lần huấn luyện độc lập.

## 6. Cấu trúc dự án

```text
configs/default.json       Cấu hình CNN, dữ liệu, nhiễu và enhancer
src/
  data.py                  Download, split theo track, crop và audit
  model.py                 Kiến trúc CNN
  training.py              Augmentation, train và checkpoint
  degradation.py           Tạo thiếu sáng và nhiễu
  enhancement.py           Phương pháp tăng sáng và adapter pretrained
  experiment.py            Cache, chọn tham số, khóa giao thức, evaluate
  metrics.py               Metric và bootstrap
  reporting.py             Bảng, biểu đồ và báo cáo
scripts/run.py             CLI chung của pipeline
scripts/setup_external.py  Tải mã và checkpoint enhancer
checks/                    Kiểm tra pipeline và workflow
```

Các thư mục sau được tạo khi chạy hoặc nhận bộ artifact:

| Thư mục | Nội dung |
|---|---|
| `data/` | Dữ liệu gốc, crop và manifest |
| `external/` | Mã và weights của tác giả enhancer |
| `outputs/` | Checkpoint CNN, dự đoán, metric và báo cáo |

Cả ba thư mục nằm trong `.gitignore`. **Clone repo không tự có dữ liệu, checkpoint và kết quả.** Các bảng trong README giúp người đọc GitHub xem kết quả chính mà không cần tải artifact.

## 7. Cài đặt

Cần Python, Git, mạng để tải dữ liệu/weights và dung lượng lưu cache. Chạy lệnh từ **thư mục gốc repository**.

`requirements.txt` ghim phiên bản; `requirements-lock.txt` lưu môi trường tham chiếu. Tài liệu cũ ghi nhận Python 3.14/macOS arm64/CPU; môi trường Windows và Colab có thể cần phiên bản thư viện tương thích riêng. Nếu pip báo không tìm thấy bản ghim, cần đối chiếu Python, hệ điều hành và PyTorch phù hợp, đồng thời ghi lại thay đổi môi trường. Việc cập nhật README không bao gồm kiểm thử cài đặt mới.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trong các lệnh tiếp theo, thay `python` bằng `.\.venv\Scripts\python.exe` nếu chưa activate môi trường. Cách này không cần thay đổi execution policy của PowerShell.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Chuẩn bị dữ liệu và enhancer

```bash
python -m scripts.setup_external
python -m scripts.run download
python -m scripts.run prepare
python -m scripts.run check-data
```

Các lệnh lần lượt tải mã/checkpoint enhancer, tải GTSRB, tạo crop/manifest và kiểm tra dữ liệu. Setup lưu nguồn và SHA256 tại `external/sources.json`. `prepare` bỏ qua khi manifest đã tồn tại.

## 8. Chạy thí nghiệm

### Kiểm tra và ước lượng trước khi chạy dài

```bash
python -m scripts.run check
python -m checks.check_workflow
python -m scripts.run --device cpu smoke --count 43
python -m scripts.run budget --per-class 10 --validation-per-class 30
```

- `check`: kiểm tra các thành phần tính toán và CNN.
- `check_workflow`: chạy workflow trên dữ liệu giả trong thư mục tạm; không tạo số liệu thực nghiệm.
- `smoke`: chạy enhancer trên ảnh validation thật và tạo gallery để kiểm tra trực quan.
- `budget`: ước lượng dung lượng cache và thời gian từ log smoke nếu có.

### Train → chọn tham số → khóa cấu hình → test → báo cáo

Chuỗi lệnh sau tạo thí nghiệm ở quy mô subset như kết quả trong README:

```bash
python -m scripts.run train --seeds 11 22 33 --output outputs/train
python -m scripts.run tune --per-class 30 --checkpoints outputs/train/cnn_11.pt outputs/train/cnn_22.pt outputs/train/cnn_33.pt --output outputs/validation
python -m scripts.run freeze --selection outputs/validation/selection.json --per-class 10 --output outputs/protocol_subset.json
python -m scripts.run evaluate --protocol outputs/protocol_subset.json --output outputs/test_subset
python -m scripts.run report --output outputs/test_subset
```

| Bước | Công việc |
|---|---|
| `train` | Train CNN, lưu checkpoint tốt nhất và learning curves |
| `tune` | Chọn tham số enhancer và phương pháp bằng validation |
| `freeze` | Khóa ID test, cấu hình, hash checkpoint và nguồn mã |
| `evaluate` | Chạy điều kiện đã khóa, lưu dự đoán và metric |
| `report` | Tổng hợp bảng, biểu đồ, ví dụ lỗi và bootstrap |

Thiết bị mặc định `auto` ưu tiên CUDA, sau đó MPS, rồi CPU. Cờ chung như `--device`, `--root`, `--config`, `--threads` đứng **trước tên lệnh**:

```bash
python -m scripts.run --device cuda train --seeds 11 22 33
```

Lưu ý khi chạy lại:

- `train` chỉ bỏ qua checkpoint đã hoàn thành, khớp hash, seed, manifest, cấu hình, số epoch yêu cầu và ID train/validation. Metadata cũ thiếu thông tin xác minh hoặc lượt train dang dở sẽ bị từ chối; dùng `--output` mới, không ghi đè baseline.
- `freeze` không ghi đè file giao thức đã tồn tại; dùng tên mới cho thí nghiệm mới.
- `--per-class` giữ nguyên nhóm ảnh nên số ảnh train/validation có thể vượt yêu cầu; đối chiếu ID đã lưu.
- Giữ nguyên code, cấu hình và checkpoint giữa freeze và evaluate.
- Muốn đánh giá toàn tập, bỏ `--per-class` tương ứng ở tune/freeze, dùng output mới và ước lượng tài nguyên trước.
- Chọn tham số trên validation; không điều chỉnh theo điểm test rồi coi đó là đánh giá độc lập.

### ResNet-18 pretrained (cấu hình riêng)

`configs/resnet18.json` dùng ImageNet weights, fine-tune toàn bộ mạng với AdamW
(lr `1e-4`, weight decay `1e-4`, batch 32), tối đa 20 epoch. Early stopping theo
validation Macro-F1: patience 5, min_delta 0.001; reset patience khi tăng hơn 0.001
so với mốc cải thiện gần nhất. Checkpoint vẫn lưu Macro-F1 cao nhất, kể cả cải thiện nhỏ.
Crop/cache/enhancer và PSNR/SSIM vẫn
ở 64×64; model resize toàn crop lên 224×224 rồi normalize ImageNet đúng một lần.
Không dùng center crop. `configs/default.json` và state-dict CNN cũ vẫn chạy được;
load checkpoint ResNet không cần mạng. Chỉ train mới cần tải weights lần đầu.

```bash
python -m scripts.run check
python -m checks.check_resnet18
python -m checks.check_workflow

# Pilot kỹ thuật; dùng output mới nếu đường dẫn này đã có kết quả.
python -m scripts.run --config configs/resnet18.json train --seeds 11 --epochs 2 --per-class 10 --output outputs/resnet18_pilot/train
python -m scripts.run --config configs/resnet18.json tune --checkpoints outputs/resnet18_pilot/train/cnn_11.pt --per-class 2 --methods identity gamma --output outputs/resnet18_pilot/validation
python -m scripts.run freeze --selection outputs/resnet18_pilot/validation/selection.json --per-class 2 --pilot --output outputs/resnet18_pilot/protocol.json
python -m scripts.run evaluate --protocol outputs/resnet18_pilot/protocol.json --output outputs/resnet18_pilot/test
python -m scripts.run report --output outputs/resnet18_pilot/test

# Thực nghiệm đầy đủ: xem validation seed 11 trước khi chốt LR cho cả ba seed.
python -m scripts.run --config configs/resnet18.json train --seeds 11 --output outputs/resnet18_A/train
python -m scripts.run --config configs/resnet18.json train --seeds 11 22 33 --output outputs/resnet18_A/train
python -m scripts.run --config configs/resnet18.json tune --checkpoints outputs/resnet18_A/train/cnn_11.pt outputs/resnet18_A/train/cnn_22.pt outputs/resnet18_A/train/cnn_33.pt --per-class 30 --output outputs/resnet18_A/validation
python -m scripts.run freeze --selection outputs/resnet18_A/validation/selection.json --per-class 10 --subset-seed 42 --output outputs/resnet18_A/protocol.json
python -m scripts.run evaluate --protocol outputs/resnet18_A/protocol.json --output outputs/resnet18_A/test_subset
python -m scripts.run report --output outputs/resnet18_A/test_subset
```

`check_resnet18` kiểm tra preprocessing, gradient backbone/head, round-trip offline,
checkpoint cũ, từ chối cấu hình sai và skip an toàn. `check_workflow` chạy cả hai kiến
trúc, kiểm tra cache/replay, Weighted-F1 từ báo cáo cũ và checkpoint bị sửa.

Khi tải weights trên Python macOS gặp lỗi CA, dùng bundle đã cài, không tắt TLS:

```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

Báo cáo bổ sung Weighted-F1 trong `summary.csv`, `noise_averaged.csv`, `aggregate.csv`
và F1/support từng lớp ở `per_class.csv` (gồm class 19). Báo cáo cũ thiếu Weighted-F1
được tính lại từ predictions trong bộ nhớ, không sửa summary gốc. `latency_ms` chỉ đo
enhancer; `classifier_latency.json` đo resize + normalization + classifier riêng với
warm-up và đồng bộ thiết bị, ghi batch, crop resolution và device. Thời gian epoch gồm
train + validation; RSS là peak process, CUDA là peak allocated, MPS là driver allocated
ở cuối epoch (không phải peak VRAM). Windows không có số RSS qua stdlib này.

Audit ngày 07/10/2026 xác nhận 31.350 train / 7.829 validation / 12.630 test, đủ 43 lớp,
quarantine 30 ảnh; subset 430 ảnh seed 42 khớp `outputs/test_subset_plan.json`.
Artifact `outputs/low-light-outputs-A-subset/` dùng cho các bảng cũ chưa có trong workspace.
Hai checkpoint tại `outputs/train/` có validation Macro-F1 lần lượt 66,64% và 63,74%,
không phải checkpoint đã xác minh cho số liệu README. State-dict của chúng còn dùng CNN trước BatchNorm, khác kiến trúc baseline trong code trước đợt nâng cấp này; strict loader từ chối đúng. Tương thích ngược được kiểm chứng với CNN baseline hiện tại, không tự đoán lại kiến trúc lịch sử từ keys. Chưa dùng các checkpoint lịch sử để tuyên bố mức cải thiện.

Pilot local đã hoàn thành ngày 07/10/2026: seed 11, 2 epoch, MPS, batch 32,
1.290 ảnh train + 1.290 validation (giữ nguyên track nên vượt ngân sách 10 ảnh/lớp).
Validation Accuracy **38,37%**, Macro-F1/Weighted-F1 **34,46%**; class 19 F1=0,
support=30. Đây chỉ là kiểm chứng kỹ thuật, **chưa đạt mục tiêu 90%**. Tune chọn gamma
bằng validation; test pilot đủ **19/19 lượt**, 86 ảnh/lượt, clean Accuracy **30,23%**,
Macro-F1/Weighted-F1 **26,68%**. Không dùng điểm test pilot để chọn LR/loss cho lượt đầy đủ.

Epoch pilot mất 35,68 và 25,18 giây trên máy 8 GiB RAM; RSS peak ghi nhận khoảng 331 MiB,
MPS driver allocated cuối epoch khoảng 2,11 GiB. Đây không phải benchmark độc lập vì có
check CPU chạy đồng thời. Classifier + preprocessing đo riêng **3,64 ms/ảnh**, batch 32,
2 warm-up + 5 batch đo. Artifacts: `outputs/resnet18_pilot/test/report.md`,
`outputs/resnet18_pilot/train/learning_11.csv`, `outputs/resnet18_pilot/test/classifier_latency.json`.

### Chạy trên Kaggle

`scripts/run_resnet18_kaggle.py` chép snapshot private do Kaggle giải nén, xác minh hash, chạy check và
smoke enhancer trên GPU, rồi train đầy đủ. Giữ torch/torchvision có CUDA của Kaggle;
không cài bộ requirements dành cho macOS đè lên. Nếu seed 11 chưa đạt cả ba metric
validation >90%, thử thêm LR `3e-5` và `3e-4`, chọn theo Macro-F1 validation. Tái sử dụng
seed 11 hợp lệ, train 22/33, lưu `validation_summary.json`, rồi tune/freeze/evaluate
đủ 219 lượt trên subset 430 ảnh đã chốt. Training ảnh tối và full official test chưa
nằm trong job này; baseline so sánh paired vẫn cần artifacts tương ứng.

Job private [GTSRB ResNet18 Protocol A 20261007](https://www.kaggle.com/code/sangtranminh211/gtsrb-resnet18-protocol-a-20261007), v3, đã train đủ seed 11/22/33 và đánh giá 219/219 lượt trên subset 430 ảnh.
[Job full-test](https://www.kaggle.com/code/sangtranminh211/gtsrb-resnet18-full-test-12630), v4, đã hoàn thành 219/219 lượt trên toàn bộ 12.630 ảnh official test, dùng lại checkpoint và tham số đã khóa; không train hoặc tune lại.

Ngày 12/10/2026 đã tải archive đầy đủ, kiểm CRC và hash của ba checkpoint, manifest, mã nguồn. Đã đối chiếu ID/nhãn và tính lại Accuracy, Macro-F1, Weighted-F1 từ **2.765.970 dự đoán**, kiểm mean/SD giữa ba seed. Kết quả kiểm chứng: `outputs/resnet18_A/test_full/verification.json`.

### Kết quả ResNet-18 trên full official test

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

### Thí nghiệm tiếp theo: train hỗn hợp clean/low-light trên Kaggle

`configs/resnet18_lowlight.json` giữ nguyên ResNet-18, optimizer, split, mức tối và augmentation; train 50% clean, 50% còn lại chia đều mild/medium/severe. Noise seed train/validation riêng (`4101`, `9101`), tách bộ seed evaluation (`101`, `202`, `303`). Chọn checkpoint theo Macro-F1 validation trung bình ngang trọng số clean và ba mức tối; yêu cầu Macro-F1 validation clean >= **98,4741%** (mốc ResNet-A clean validation trừ 1 điểm phần trăm). Nếu không có checkpoint đạt, ghi thất bại và không đánh giá test.

`scripts/run_resnet18_lowlight_kaggle.py` chạy pilot GPU 2 epoch trước, tune/freeze/test pilot 19 lượt × 86 ảnh trong output riêng; kết quả pilot không dùng chọn hyperparameter. Sau khi pilot qua, train đủ ba seed, tune enhancer bằng validation, đánh giá 219 lượt trên subset 430 ảnh. Replay classifier clean với checkpoint gốc trên cùng ID để so sánh paired. `identity` cô lập thay đổi training regime; các enhancer được tune riêng theo validation nên delta có thể gồm thay đổi tham số enhancer. Nhãn giao thức mới là `mixed_lowlight`, không gọi là giao thức A clean.

Artifacts mới: `outputs/resnet18_lowlight/`; pilot: `outputs/resnet18_lowlight_pilot/`; baseline replay: `outputs/resnet18_clean_replay/`. Job private: [GTSRB ResNet18 Mixed Lowlight 20261012](https://www.kaggle.com/code/sangtranminh211/gtsrb-resnet18-mixed-lowlight-20261012). Chưa có kết quả cuối của thí nghiệm mixed. Full-test cho mixed là bước sau, với cấu hình khóa và output riêng.

```bash
python -m checks.check_lowlight
python -m scripts.verify_results outputs/resnet18_A/test_full
kaggle kernels status sangtranminh211/gtsrb-resnet18-mixed-lowlight-20261012
kaggle kernels logs -f sangtranminh211/gtsrb-resnet18-mixed-lowlight-20261012
kaggle kernels output sangtranminh211/gtsrb-resnet18-mixed-lowlight-20261012 -p outputs/resnet18_lowlight_kaggle/download --file-pattern '^resnet18-lowlight-results\.zip$'
```

## 9. Đọc kết quả ở đâu?

| File sau khi chạy | Nội dung |
|---|---|
| `outputs/train/train_<seed>.json` | Trạng thái, môi trường và ID dữ liệu train |
| `outputs/train/cnn_<seed>.pt` | Checkpoint CNN tốt nhất |
| `outputs/train/learning_<seed>.csv` | Loss, Accuracy, Macro-F1, Weighted-F1, thời gian epoch và bộ nhớ |
| `outputs/validation/selection.json` | Tham số và phương pháp đã chọn |
| `outputs/protocol_subset.json` | Giao thức đã khóa |
| `outputs/test_subset/report.md` | Báo cáo dễ đọc |
| `outputs/test_subset/summary.csv` | Kết quả từng lượt |
| `outputs/test_subset/aggregate.csv` | Trung bình và độ lệch chuẩn |
| `outputs/test_subset/predictions/` | Dự đoán từng ảnh |
| `outputs/test_subset/figures/` | Biểu đồ, confusion matrix và ví dụ ảnh |
| `outputs/test_subset/bootstrap.json` | Khoảng tin cậy có điều kiện trên checkpoint |

Bộ artifact dùng cho bảng kết quả README được lưu riêng tại `outputs/low-light-outputs-A-subset/`, gồm `train/`, `validation/`, `test_subset/` và manifest. Khi chuyển artifact giữa Colab và máy cá nhân, đường dẫn tuyệt đối trong metadata có thể còn trỏ tới môi trường cũ; cần đối chiếu trước khi chạy tiếp pipeline.

## 10. Giới hạn và hướng mở rộng

- CNN cũ được báo trên subset 430 ảnh; ResNet-18 giao thức A đã đánh giá đủ 12.630 ảnh official test. Hai bảng có phạm vi khác nhau.
- Thiếu sáng được mô phỏng; chưa đại diện đầy đủ cho camera ban đêm, mưa, lóa đèn và nhòe chuyển động.
- Hệ thống xử lý crop biển báo, chưa triển khai detector hoặc video thời gian thực.
- CNN cũ có class 19 F1=0 trong validation; ResNet-A đạt F1=1 ở class này trên clean validation. Cần xem per-class riêng theo mức tối.
- Đây là thực nghiệm so sánh phương pháp có sẵn; chưa đề xuất thuật toán tăng sáng mới.

Các hướng mở rộng gồm training hỗn hợp ảnh sáng/tối, phân tích từng lớp và kiểm tra trên ảnh thiếu sáng thực. Mỗi thay đổi cần được ghi nhận thành thí nghiệm riêng để so sánh rõ ràng.

## 11. Tài liệu liên quan

- [PROJECT_GUIDE.md](PROJECT_GUIDE.md): giao thức và yêu cầu nghiên cứu.
- [DATASETS.md](DATASETS.md): nguồn dữ liệu và kiểm tra rò rỉ.
- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): quá trình cải thiện CNN và số liệu từng bước.
- [plan.md](plan.md): kế hoạch công việc; đối chiếu checkbox với artifact thực tế.

Nguồn mã tác giả của Zero-DCE, Retinexformer và Diffusion-Low-Light được khai báo trong `scripts/setup_external.py`. Dữ liệu, mã và checkpoint có điều kiện sử dụng riêng; đối chiếu giấy phép từng nguồn khi tái sử dụng hoặc phân phối lại.
