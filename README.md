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

- `train` bỏ qua seed nếu `cnn_<seed>.pt` đã tồn tại. Checkpoint tồn tại chưa chứng minh lần train trước hoàn thành: kiểm tra `train_<seed>.json`. Muốn train lại, chọn `--output` mới.
- `freeze` không ghi đè file giao thức đã tồn tại; dùng tên mới cho thí nghiệm mới.
- `--per-class` giữ nguyên nhóm ảnh nên số ảnh train/validation có thể vượt yêu cầu; đối chiếu ID đã lưu.
- Giữ nguyên code, cấu hình và checkpoint giữa freeze và evaluate.
- Muốn đánh giá toàn tập, bỏ `--per-class` tương ứng ở tune/freeze, dùng output mới và ước lượng tài nguyên trước.
- Chọn tham số trên validation; không điều chỉnh theo điểm test rồi coi đó là đánh giá độc lập.

## 9. Đọc kết quả ở đâu?

| File sau khi chạy | Nội dung |
|---|---|
| `outputs/train/train_<seed>.json` | Trạng thái, môi trường và ID dữ liệu train |
| `outputs/train/cnn_<seed>.pt` | Checkpoint CNN tốt nhất |
| `outputs/train/learning_<seed>.csv` | Loss, Accuracy, Macro-F1 từng epoch |
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

- Đánh giá tăng sáng hiện tại dùng **430 ảnh test**, chưa chạy toàn bộ 12.630 ảnh official test.
- Thiếu sáng được mô phỏng; chưa đại diện đầy đủ cho camera ban đêm, mưa, lóa đèn và nhòe chuyển động.
- Hệ thống xử lý crop biển báo, chưa triển khai detector hoặc video thời gian thực.
- Class 19 vẫn có F1 bằng 0 trong kết quả validation CNN đã ghi nhận; điểm trung bình chưa mô tả hết các lớp yếu.
- Đây là thực nghiệm so sánh phương pháp có sẵn; chưa đề xuất thuật toán tăng sáng mới.

Các hướng mở rộng gồm đánh giá toàn test, xử lý mất cân bằng lớp và kiểm tra trên ảnh thiếu sáng thực. Mỗi thay đổi cần được ghi nhận thành thí nghiệm riêng để so sánh rõ ràng.

## 11. Tài liệu liên quan

- [PROJECT_GUIDE.md](PROJECT_GUIDE.md): giao thức và yêu cầu nghiên cứu.
- [DATASETS.md](DATASETS.md): nguồn dữ liệu và kiểm tra rò rỉ.
- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): quá trình cải thiện CNN và số liệu từng bước.
- [plan.md](plan.md): kế hoạch công việc; đối chiếu checkbox với artifact thực tế.

Nguồn mã tác giả của Zero-DCE, Retinexformer và Diffusion-Low-Light được khai báo trong `scripts/setup_external.py`. Dữ liệu, mã và checkpoint có điều kiện sử dụng riêng; đối chiếu giấy phép từng nguồn khi tái sử dụng hoặc phân phối lại.
