# Kế hoạch nâng cấp classifier sang ResNet-18

Ngày lập: 07/10/2026. Nhánh: `resnet18-migration-plan`.
Mốc xuất phát: `b750ed5` trên `main` cục bộ; chưa fetch để xác nhận trạng thái remote.
Trạng thái cập nhật 12/10/2026: ResNet-A đã train đủ ba seed, hoàn thành subset và full official test; artifacts đã tải và kiểm chứng. Đang triển khai thí nghiệm mixed low-light riêng trên Kaggle.

## 1. Mục tiêu và giới hạn

Thay classifier CNN nhỏ bằng ResNet-18 pretrained ImageNet, fine-tune cho 43 lớp GTSRB. Giữ CNN nhỏ làm baseline có thể chạy lại.

Giả định “các chỉ số trên 90%” nghĩa là Accuracy, Macro-F1 và Weighted-F1 tổng thể, không phải F1 của từng lớp. PSNR có đơn vị dB; SSIM không thuộc mục tiêu phần trăm này.

| Phạm vi | Tiêu chí mục tiêu | Cách kết luận |
|---|---|---|
| Validation clean, 7.829 ảnh | Cả ba metric trung bình 3 seed > 0,90 | Dùng chọn cấu hình, không thay thế kết quả test |
| Test clean | Cả ba metric trung bình 3 seed > 0,90 | Báo mean ± SD và từng seed; nếu chỉ mean đạt, không nói mọi seed đạt |
| Test tối nhẹ/vừa/nặng | Đo từng mức; mong muốn cả ba metric > 0,90 ở mỗi mức | Mục tiêu mở rộng, chưa có căn cứ bảo đảm đạt |
| Từng lớp | Báo support, F1, confusion matrix, chú ý class 19 | Không giấu lớp yếu sau metric tổng thể |

Không đổi split, seed, độ tối hoặc chọn ảnh test dễ hơn để đạt ngưỡng. Không diễn giải >90% trung bình là >90% ở mọi lớp/method/mức tối. Nếu chưa đạt, báo kết quả thật và chỉ điều chỉnh theo validation.

## 2. Hiện trạng đã kiểm tra

- `src/model.py`: `make_cnn()` dựng CNN 3 khối convolution, global pooling và Linear 128→43; `image_tensor()` chuyển HWC sang CHW, không normalize.
- `src/training.py`: train gọi trực tiếp `make_cnn()`; chỉ chấp nhận `normalization=none`; Adam lr=0,001; chọn checkpoint theo validation Macro-F1. Checkpoint đã lưu `cnn_config` nhưng loader chưa dùng để chọn kiến trúc.
- `src/experiment.py`: `load_cnn()` luôn dựng CNN cũ; `predict()` đưa batch ảnh cache thẳng vào model. `tune → freeze → evaluate` khóa manifest, code và checkpoint hashes.
- `src/metrics.py`: đã tính đủ Accuracy, Macro-F1, Weighted-F1 và confusion matrix; không cần viết lại công thức.
- `src/experiment.py` và `src/reporting.py`: bảng summary/aggregate chưa xuất Weighted-F1 dù JSON từng lớp có giá trị này.
- `scripts/run.py`: khóa crop 64×64; CLI đã có `--config`, `--output`, `--device`, `--seeds`; tận dụng, không tạo CLI mới.
- `checks/check_pipeline.py`, `checks/check_workflow.py`: đều gọi `make_cnn()`; phải giữ đường chạy mặc định cũ.
- `torchvision` đã có trong dependencies; không thêm thư viện model mới.

Số liệu tham chiếu trong README: validation clean Accuracy 92,45%, Macro-F1 87,57%; test clean subset 430 ảnh Accuracy 84,81%, Macro-F1 83,79%. Tài liệu cải tiến ghi Weighted-F1 validation 92,17% và F1 class 19 bằng 0.

Bộ `outputs/low-light-outputs-A-subset/` không hiện diện trong workspace lúc kiểm tra. Trước khi kết luận mức cải thiện, cần lấy lại artifacts tương ứng hoặc chạy baseline trên cùng tập ID; không xem bảng README là kiểm chứng lại checkpoint hiện có.

## 3. Thiết kế được chọn

### 3.1. Luồng ảnh và model

```text
Crop RGB 64×64 trong [0,1]
  → augmentation lúc train / enhancer lúc đánh giá
  → resize bilinear 224×224, antialias=True
  → ImageNet normalization
  → ResNet-18
  → Linear 512→43
  → logits [B,43]
```

- Giữ nguyên crop, dữ liệu cache, degradation, enhancer và PSNR/SSIM ở 64×64.
- Chỉ resize trước classifier. Resize không tạo thêm thông tin ảnh.
- Normalize mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`; áp dụng đúng một lần, sau augmentation/enhancement.
- Dùng resize toàn crop; không dùng center crop mặc định có thể cắt thêm phần biển báo. Đây là lựa chọn fine-tuning cần được ghi nhận, không khẳng định trùng hoàn toàn preprocessing pretrained.
- Đặt preprocessing cố định trong model ResNet để train, validation, tune và test đi qua cùng một đường. Dùng buffer cho mean/std để theo đúng device.
- Mở rộng `make_cnn()` bằng cấu hình tùy chọn và cờ khởi tạo pretrained. Không truyền cấu hình vẫn trả CNN cũ với state-dict keys giữ nguyên.
- Khi bắt đầu train mới: dùng `ResNet18_Weights.DEFAULT`, thay `fc` bằng `Linear(512,43)` rồi fine-tune toàn bộ mạng.
- Khi load checkpoint: dựng kiến trúc với `weights=None`, sau đó load state dict strict; không tải lại ImageNet weights khi inference.
- Checkpoint cũ thiếu trường architecture được hiểu là CNN cũ. Architecture/normalization không hợp lệ phải báo lỗi, không tự chuyển model.

Nguồn API: [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Pretrained là điểm khởi đầu, không phải bằng chứng đạt 90% trên dự án này.

### 3.2. Cấu hình train khởi điểm

Tạo `configs/resnet18.json` theo cấu trúc hiện có; giữ `configs/default.json` làm baseline.

| Trường trong `cnn` | Giá trị dự kiến |
|---|---|
| architecture | `resnet18` |
| weights | `IMAGENET1K_V1` — ghi tên cụ thể thay vì chỉ DEFAULT |
| input_size | 224 |
| normalization | `imagenet` |
| optimizer | `adamw` |
| learning_rate | 0,0001 |
| weight_decay | 0,0001 |
| batch_size | 32 |
| epochs / patience | 20 / 5 |
| min_delta (validation Macro-F1) | 0.001 |

Giữ augmentation hiện tại ở lần thử đầu để giảm số yếu tố thay đổi. Chưa thêm scheduler, freeze/unfreeze nhiều giai đoạn, focal loss, sampler hay ensemble. Các trường mới phải có mặc định tương thích CNN cũ: Adam và không thêm weight decay ngoài hành vi hiện tại.

Nếu hết bộ nhớ, giảm batch về 16, ghi cấu hình thực tế và dùng đồng nhất cho các seed. Không tự hứa thời gian train: đo một epoch trên thiết bị thực rồi ước lượng. Giữ deterministic settings; nếu thiết bị có toán tử không hỗ trợ thì báo rõ hoặc chọn thiết bị khác, không âm thầm tắt.

### 3.3. Checkpoint và provenance

- Lưu architecture, input size, normalization, weights khởi tạo, optimizer và cấu hình train trong `cnn_config`; tiếp tục lưu manifest hash, epoch, seed, environment.
- Kiểm tra cấu hình classifier của checkpoint khớp cấu hình tune; không trộn checkpoint CNN nhỏ với ResNet trong một run matrix.
- Tách toàn bộ output vào `outputs/resnet18_A/`; pilot vào `outputs/resnet18_pilot/`. Giữ tên `cnn_11.pt`… để tận dụng CLI.
- Logic hiện tại bỏ qua train nếu checkpoint đã tồn tại: bổ sung kiểm tra metadata/config/manifest trước khi skip; nếu không khớp thì yêu cầu output mới, không ghi đè.
- Không sửa protocol JSON đã khóa hoặc giả mạo hash. Sau khi hoàn tất code, chạy lại selection/freeze cho thí nghiệm mới.

## 4. Các bước triển khai và kiểm chứng

### Bước 0 — Chốt baseline và dữ liệu

1. Kiểm tra manifest/audit; xác nhận split theo track, quarantine và đủ 43 lớp.
2. Kiểm tra checkpoint hiện có và metadata; khôi phục baseline artifacts nếu có.
3. Chốt ID subset 430 ảnh giống baseline; nếu không xác nhận được ID thì ghi giới hạn hoặc chạy lại baseline.
4. Ghi thiết bị, bộ nhớ, phiên bản torch/torchvision và khả năng tải weights.

Đạt khi có baseline truy xuất được hoặc ghi rõ chưa có; không suy số liệu mới từ tài liệu cũ.

### Bước 1 — Tích hợp ResNet và preprocessing

Sửa tối thiểu `src/model.py`, `src/training.py`, `src/experiment.py`; thêm cấu hình riêng. Kiểm tra:

- CNN cũ và ResNet nhận `[2,3,64,64]`, trả logits hữu hạn `[2,43]`.
- Mean/std và resize thực sự được áp dụng một lần; eval cùng một ảnh qua đường dataset và đường predict cho logits khớp trong dung sai.
- Một train step có loss/gradient hữu hạn, cập nhật được backbone và head.
- Save/load round-trip giữ logits trong dung sai khi eval; load không cần mạng.
- Load checkpoint CNN cũ vẫn chạy được; sai architecture/config/manifest bị từ chối rõ ràng.

Đặt kiểm tra mới trong một file `checks/check_resnet18.py` dùng assert và thư mục tạm; dùng weights=None cho kiểm tra offline. Việc tải pretrained thật chỉ kiểm chứng riêng trong pilot.

### Bước 2 — Hoàn thiện metric và kiểm tra workflow

- Đưa `weighted_f1` có sẵn vào history, summary, noise-averaged và aggregate; báo mean ± SD.
- Đọc báo cáo cũ thiếu Weighted-F1: tính lại từ predictions bằng `classification()` nếu cần; không gán 0 hoặc tự bịa số. Không thay đổi artifact gốc chỉ để bổ sung cột.
- Giữ F1 từng lớp/confusion matrix; kiểm tra class 19 và các lớp có support thấp.
- Mở rộng workflow check cho ResNet với dữ liệu giả và weights=None; giữ kiểm tra replay cache và phát hiện checkpoint bị sửa.
- PSNR/SSIM cùng ảnh enhanced không được đổi chỉ vì thay classifier.
- `latency_ms` hiện là thời gian enhancer; không gọi nó là thời gian toàn pipeline. Đo riêng preprocessing+classifier với warm-up và đồng bộ thiết bị, ghi batch/resolution/device.

Đạt khi kiểm tra baseline lẫn ResNet qua được và báo cáo phân biệt đúng metric ảnh, nhận diện, thời gian.

### Bước 3 — Pilot kỹ thuật

- Seed 11, 2 epoch, tối đa 10 ảnh/lớp train và validation; chỉ kiểm tra kỹ thuật, không dùng để tuyên bố chất lượng.
- Chạy tune/freeze/evaluate với identity và gamma, gắn `--pilot`, 2 ảnh test/lớp để kiểm tra toàn luồng. Pilot không dùng đánh giá này để chọn hyperparameter.
- Đo RAM/VRAM, thời gian/epoch, classifier latency; kiểm tra lỗi tải weights/device trước lượt dài.

Đạt khi chạy end-to-end, artifact tách biệt và không có NaN/lỗi shape. Không yêu cầu metric cao từ pilot 2 epoch.

### Bước 4 — Train clean đầy đủ và chọn cấu hình trên validation

1. Train seed 11 trên toàn bộ train với cấu hình khởi điểm.
2. Xem learning curve, Macro-F1, F1 class 19 và confusion matrix trên validation.
3. Nếu chưa đạt, thử tối đa hai LR bổ sung `3e-5` và `3e-4` với cùng seed/split; giữ output riêng. Chọn bằng validation Macro-F1.
4. Nếu vẫn có lớp F1=0, kiểm tra nhãn/support/nhầm lẫn trước; chỉ thử weighted cross-entropy từ thống kê train khi có bằng chứng cần. Ghi thành ablation riêng, không đồng thời thêm sampler.
5. Chốt cấu hình rồi chạy đủ seed 11,22,33. Nếu lượt seed 11 đã đúng cấu hình cuối thì tái sử dụng artifact hợp lệ.

Không đọc test để chọn LR/loss/epoch. Nếu validation vẫn không đạt mục tiêu, báo trạng thái chưa đạt thay vì tiếp tục tuning không giới hạn.

### Bước 5 — Đánh giá giao thức A với classifier mới

- Tune enhancer trên validation cho ResNet; không mặc định gamma vẫn thắng.
- Khóa cấu hình và chạy cùng subset test 430 ảnh, đủ 219 lượt như baseline: 3 clean + 3 CNN seed × 3 mức × 3 noise seed × 8 method.
- So sánh paired cùng ID, condition, noise seed; báo cả baseline và ResNet, chênh lệch bằng điểm phần trăm.
- Trung bình noise seed trong mỗi CNN seed trước, rồi mean ± SD giữa ba CNN seed; không coi 9 tổ hợp là 9 lần train độc lập.
- Báo riêng clean và từng mức tối, đủ mọi method; không lấy method thắng trên test làm pipeline được chọn.
- Khi đủ ngân sách, đánh giá official test 12.630 ảnh theo protocol riêng với cấu hình đã khóa. Để so sánh full-test trực tiếp phải có baseline trên cùng tập; không so full-test mới với subset cũ như cùng benchmark.

Thử nghiệm này so sánh cả gói thay đổi: kiến trúc, pretraining, preprocessing và optimizer. Không quy toàn bộ mức tăng riêng cho residual connection.

### Bước 6 — Thí nghiệm ảnh tối riêng nếu cần

Chỉ triển khai sau khi có kết quả A. Mục đích: tăng khả năng nhận diện ảnh tối, khác câu hỏi tăng sáng có giúp classifier train clean không.

- Đề xuất ban đầu: 50% mẫu clean; 50% còn lại chia đều mild/medium/severe, tạo từ ảnh train bằng degradation hiện có.
- Dùng RNG train tái lập được, tách seed khỏi bộ evaluation `101,202,303`; validation tối dùng bộ seed validation cố định riêng. Tuyệt đối không tạo dữ liệu train từ validation/test.
- Chọn checkpoint theo Macro-F1 validation trung bình trên clean + ba mức tối với trọng số bằng nhau; ghi thêm guardrail clean không giảm quá 1 điểm phần trăm so với ResNet-A.
- Xuất `outputs/resnet18_lowlight/` riêng; lưu training regime trong metadata/protocol/report. Hiện code hard-code nhãn A ở nhiều chỗ: phải sửa đúng các đường freeze/evaluate/report trước khi dùng nhãn thí nghiệm mới.
- So sánh clean/mild/medium/severe của ResNet-A và ResNet train mixed; không gọi kết quả này là cải thiện do enhancer đơn thuần.

Đây là giai đoạn tiếp theo có điều kiện, không ghép vào lần triển khai ResNet-A đầu tiên. >90% ở severe vẫn cần đo; ảnh mất chi tiết do gamma/noise/clipping có thể không còn đủ thông tin.

## 5. Danh sách file dự kiến

| File | Trách nhiệm thay đổi |
|---|---|
| `configs/resnet18.json` — mới | Cấu hình thí nghiệm riêng |
| `src/model.py` | Chọn 2 kiến trúc, preprocessing ResNet dùng chung |
| `src/training.py` | Khởi tạo pretrained, optimizer theo config, metadata và skip an toàn |
| `src/experiment.py` | Load đúng architecture, validate config, xuất Weighted-F1 |
| `src/reporting.py` | Tổng hợp Weighted-F1 và hiển thị danh tính classifier |
| `checks/check_resnet18.py` — mới | Kiểm tra model, preprocessing, checkpoint và lỗi cấu hình |
| `checks/check_workflow.py` | Kiểm chứng end-to-end ResNet, giữ coverage CNN cũ |
| `scripts/run.py` | Chỉ cập nhật help nếu cần; tận dụng tham số sẵn có |
| `README.md` | Hướng dẫn lệnh, kết quả và giới hạn sau khi đã chạy thật |

Không sửa công thức metric, enhancer, crop, manifest hay dependencies nếu không có lỗi liên quan. Chưa tạo registry/factory tổng quát hay hệ thống plugin model.

## 6. Lệnh thực hiện sau khi hoàn tất code

Các lệnh dưới là kế hoạch, chưa chạy. `configs/resnet18.json` và check mới chưa tồn tại tại thời điểm lập tài liệu. Chạy trong thư mục repo, dùng Python của môi trường đã cài dependencies.

```bash
cd /Users/minhsang/low-light-enhancement-for-traffic-sign-recognition
source .venv/bin/activate

# Audit dữ liệu và kiểm tra hai kiến trúc.
python -m scripts.run check-data
python -m scripts.run check
python -m checks.check_resnet18
python -m checks.check_workflow

# Pilot train: đầu ra riêng, không ghi đè checkpoint baseline.
python -m scripts.run --config configs/resnet18.json --device auto train --seeds 11 --epochs 2 --per-class 10 --output outputs/resnet18_pilot/train
python -m scripts.run --config configs/resnet18.json --device auto tune --checkpoints outputs/resnet18_pilot/train/cnn_11.pt --per-class 2 --methods identity gamma --output outputs/resnet18_pilot/validation
python -m scripts.run freeze --selection outputs/resnet18_pilot/validation/selection.json --per-class 2 --pilot --output outputs/resnet18_pilot/protocol.json
python -m scripts.run --device auto evaluate --protocol outputs/resnet18_pilot/protocol.json --output outputs/resnet18_pilot/test
python -m scripts.run report --output outputs/resnet18_pilot/test

# Sau khi đã chọn cấu hình bằng validation: train đủ ba seed.
python -m scripts.run --config configs/resnet18.json --device auto train --seeds 11 22 33 --output outputs/resnet18_A/train

# Chọn tham số enhancer trên validation 30 ảnh/lớp.
python -m scripts.run --config configs/resnet18.json --device auto tune --checkpoints outputs/resnet18_A/train/cnn_11.pt outputs/resnet18_A/train/cnn_22.pt outputs/resnet18_A/train/cnn_33.pt --per-class 30 --output outputs/resnet18_A/validation

# Chỉ dùng subset-seed 42 nếu đã xác nhận ID khớp baseline.
python -m scripts.run freeze --selection outputs/resnet18_A/validation/selection.json --per-class 10 --subset-seed 42 --output outputs/resnet18_A/protocol.json
python -m scripts.run --device auto evaluate --protocol outputs/resnet18_A/protocol.json --output outputs/resnet18_A/test_subset
python -m scripts.run report --output outputs/resnet18_A/test_subset
```

Không thay code giữa tune/freeze/evaluate vì source hash đã khóa. Nếu cần sửa, dùng output mới và tạo lại selection/protocol hợp lệ. Lệnh `budget` hiện ước lượng cache/enhancer, không thay thế benchmark thời gian train ResNet.

## 7. Tiêu chí hoàn thành và bàn giao

- [x] CNN cũ vẫn load/chạy được; default config giữ hành vi cũ.
- [x] ResNet load pretrained khi train mới; checkpoint inference hoạt động offline.
- [x] Train và inference dùng cùng preprocessing; có kiểm tra runnable.
- [x] Ba seed hoàn thành, checkpoint đúng epoch validation Macro-F1 tốt nhất.
- [x] Run matrix hoàn chỉnh, đủ ID và không bỏ ảnh lỗi.
- [x] Accuracy/Macro-F1/Weighted-F1 có mean ± SD, từng seed và từng mức tối.
- [x] Báo cáo class 19, confusion matrix và latency classifier riêng.
- [x] Kết luận đạt/chưa đạt >90% theo từng phạm vi; không dùng mục tiêu làm số liệu.
- [x] Artifacts baseline được giữ nguyên; README trỏ đúng kết quả mới.

Hoàn thành tích hợp kỹ thuật không đồng nghĩa đạt mục tiêu thực nghiệm. Nếu metric chưa đạt, bàn giao kết quả và nguyên nhân quan sát được. Quay về baseline bằng `configs/default.json` và checkpoint cũ trong output riêng; không xóa kết quả hoặc dùng lệnh Git phá hủy.

## 8. Nhật ký triển khai 07/10/2026

- Đã tích hợp hai kiến trúc, preprocessing trong model, Adam/AdamW theo config,
  kiểm tra metadata trước khi skip và load checkpoint đúng kiến trúc offline.
- Đã bổ sung Weighted-F1, per-class CSV và benchmark classifier tách enhancer.
- Audit, check pipeline, check ResNet và check workflow cho cả hai kiến trúc đã qua.
  Workflow đối chiếu ảnh/PSNR/SSIM bằng cùng enhancer parameters; gamma được tune riêng
  có thể khác giữa hai classifier nên không dùng hai gamma đã chọn khác nhau để kiểm tra bất biến.
- Pilot pretrained MPS hoàn tất 2 epoch, tune/freeze/evaluate/report đủ 19 lượt × 86 ảnh.
  `--per-class` giữ track nên train/validation thực tế 1.290 ảnh mỗi split, không phải 430.
  Không sửa split hoặc hàm subset để ép số ảnh. Metric và tài nguyên thực nằm trong README.
- Subset 430 ảnh khớp `outputs/test_subset_plan.json`; artifacts baseline của bảng README
  chưa có, hai checkpoint local không khớp metric tham chiếu và dùng state-dict trước BatchNorm,
  vốn không khớp CNN ở commit trước migration. Strict loader từ chối; không tự đổi kiến trúc.
  Tương thích baseline được kiểm chứng bằng CNN hiện tại trong các kiểm tra offline.
  Chưa kết luận mức cải thiện.
- Theo yêu cầu tiếp theo của người dùng, chuyển train đầy đủ sang Kaggle;
  `scripts/run_resnet18_kaggle.py` chạy chọn LR trên validation → đủ ba seed → giao thức A.
  Bộ snapshot private và metadata ở `outputs/resnet18_kaggle/`.
- Các ô về ba seed/219 lượt/mean ± SD còn chờ kết quả Kaggle; chưa tuyên bố đạt 90%.
  Thí nghiệm mixed low-light và official full-test vẫn là bước sau có điều kiện.

- Kaggle v2 đã qua kiểm tra GPU Tesla T4 và smoke cả 8 enhancer. Log thực seed 11,
  epoch 1: validation loss 0,0387, Macro-F1 0,9768; đây chưa phải kết quả cuối ba seed/test.

- Theo yêu cầu người dùng, dừng Kaggle v2 và đổi cấu hình thành tối đa 20 epoch,
  patience 5, min_delta 0.001. Checkpoint lưu Macro-F1 cao nhất; patience chỉ reset
  khi vượt mốc cải thiện gần nhất hơn 0.001. Chưa khởi chạy lại.

## 9. Cập nhật 12/10/2026 — artifacts và thí nghiệm tiếp theo

- Kaggle ResNet-A v3 và full-test v4 đều COMPLETE. Tải archive 312 MiB, kiểm CRC, SHA256 checkpoint, manifest và source hash gốc; lưu source snapshot cũ trước sửa code.
- Full-test đủ 219 lượt × 12.630 ảnh. Đối chiếu toàn bộ 2.765.970 predictions và tính lại ba metric nhận diện, mean/SD. Xem `outputs/resnet18_A/test_full/verification.json`.
- Clean test: Accuracy 99,15%, Macro-F1 98,90%, Weighted-F1 99,13%. Diffusion được chọn bằng validation; Macro-F1 mild/medium/severe 93,45/83,87/64,88%. Chưa đạt 90% ở medium/severe.
- Thực hiện bước 6 thành `configs/resnet18_lowlight.json`, training 50% clean + 50% low-light. Seed train/validation noise riêng, checkpoint chọn bằng trung bình bốn mức validation, guardrail clean Macro-F1 >= 98,4741%. Thí nghiệm này đo training regime riêng, không quy mức tăng cho enhancer.
- Pilot và train đầy đủ chạy trên Kaggle; output/protocol mới, không sửa checkpoint hoặc protocol A đã khóa. So sánh paired với classifier clean replay trên cùng subset; delta identity cô lập training, enhancer tune riêng được ghi rõ.
- Ban đầu thư mục gốc không có quyền ghi nên dùng bản làm việc riêng. Sau khi người dùng cấp quyền, ngày 12/10/2026 đã đồng bộ code và artifacts về `/Users/minhsang/low-light-enhancement-for-traffic-sign-recognition`; collector Kaggle tiếp tục tại thư mục gốc. Bản trước đồng bộ được lưu trong `outputs/sync_backup/20261012_111952/`.
- Chưa có kết quả mixed để kết luận chất lượng; tiếp tục theo trạng thái job và artifacts thực tế.
