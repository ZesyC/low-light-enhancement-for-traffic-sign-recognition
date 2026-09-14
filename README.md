# Tăng cường thiếu sáng cho nhận diện biển báo

Pipeline Python cho GTSRB: crop ROI → thiếu sáng mô phỏng → 7 enhancer → cùng CNN 43 lớp → PSNR/SSIM, Accuracy/Macro-F1, tương quan và paired bootstrap. Áp dụng giao thức A trong [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Chạy ngay trong workspace này

```bash
source .venv/bin/activate
python -m scripts.run check
python -m checks.check_workflow
python -m scripts.run --device cpu smoke --count 43
```

- `check`: công thức nhiễu, split/ROI, phương pháp cổ điển, metric, pairing, bootstrap và CNN backward.
- `check_workflow`: dữ liệu giả **trong thư mục tạm**, kiểm tra tune → freeze → evaluate → report, cache và checkpoint bị sửa. Không tạo điểm nghiên cứu giả.
- `smoke`: chạy identity và đủ 7 enhancer trên 43 crop validation thật, ghi `outputs/smoke/smoke.json` và `gallery.png`. Cột enhancer trong gallery nhận cùng ảnh tối mức vừa.

Mọi lệnh chạy từ thư mục gốc dự án. Các cờ chung (`--device`, `--root`, `--config`, `--threads`) đứng **trước** tên lệnh.

## Cài trên máy mới

Đã kiểm tra Python 3.14, macOS arm64, CPU. Chọn bản PyTorch phù hợp phần cứng nếu chạy CUDA; ghi lại môi trường mới trước thí nghiệm.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.setup_external
python -m scripts.run download
python -m scripts.run prepare
python -m scripts.run check-data
```

`setup_external` tải đúng commit mã tác giả và các weights cần dùng, lưu SHA256 tại `external/sources.json`; không sửa repository ngoài. `requirements-lock.txt` ghi toàn bộ phiên bản đã cài. `prepare` không ghi đè manifest đã tồn tại. Khi chạy lại dữ liệu đã chuẩn bị, dùng `check-data`.

Nguồn train mặc định là **GTSRB_Final_Training_Images.zip**, 39.209 ảnh; giữ đủ 12.630 ảnh official test. Metadata [archive của nhóm tác giả](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) phân biệt bản này với `GTSRB-Training_fixed.zip` (26.640 ảnh train cuộc thi cũ). Đã sửa nhầm lẫn từ tài liệu ban đầu; các artifact `*_training_fixed` chỉ lưu pilot cũ.

Nguồn pretrained: [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE), [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer), [Wavelet Diffusion](https://github.com/JianghaiSCU/Diffusion-Low-Light). Giữ điều kiện sử dụng của tác giả. Nguồn ảnh không được suy giấy phép từ Torchvision. Trang benchmark gốc trả 502 lúc kiểm tra; chưa xác nhận quyền tái phân phối ảnh.

## Giao thức và kiểm tra dữ liệu

- 8 ảnh của một track lớp 14 trong final train trùng pixel với test. Pipeline cách ly cả track (30 ảnh) bằng `split=quarantine`, lưu `quarantine.json`; không xóa ảnh hoặc loại test. Train/validation tách theo `(class_id, track_id)`, seed 42, khoảng 20% track cho validation; giữ đủ 43 lớp và toàn bộ official test.
- ROI pixel gốc được diễn giải inclusive, padding 5% mỗi phía, lưu bbox crop theo xyxy exclusive. Crop RGB 64×64, bilinear, PNG 8-bit canonical. `roi_gallery.jpg` gồm 43 lớp và 7 ảnh tối; cần rà bằng mắt trước khi khóa nghiên cứu.
- `audit.json`, `class_distribution.csv`, `image_statistics.csv` ghi kiểm kê thực; checksum pixel chéo split phải không trùng. `near_duplicate_candidates.json` chỉ là ứng viên dHash để rà, không tự loại ảnh.
- Suy giảm trên RGB float32: gamma → Poisson theo tín hiệu → Gaussian → clip. Seed từ SHA256 của noise seed + sample ID + mức; đầu vào giống nhau giữa mọi enhancer.
- Output/cache `.npy` float32, không lượng tử hóa thêm. Cache chứa ID, hash file, tham số, phiên bản mã, checkpoint, thời gian; dùng lại giữa các CNN seed.
- HE/CLAHE xử lý Y trong YCrCb (bước này có lượng tử hóa 8-bit). MSR dùng epsilon 1e-6, percentile 1–99 chung RGB và trả đầu vào khi khoảng percentile gần 0.
- Adapter pretrained chỉ nhận ảnh tối, không nhận nhãn/ground truth. Retinexformer dùng config LOL-v1; diffusion dùng EMA và implicit sampler tác giả, eta=0, 10 bước, seed mỗi ảnh cố định. Pad replicate đến bội số 4/32 rồi unpad; ở 64×64 không cần pad.
- CNN đúng 3 convolution theo tài liệu, không augmentation hoặc normalization ngoài chia 255. Adam; early stopping theo Macro-F1 validation; weights-only checkpoint.

## Pilot và ước lượng tài nguyên

```bash
python -m scripts.run --device cpu train --seeds 11 --epochs 2 --per-class 1 --output outputs/pilot/train
python -m scripts.run --device cpu smoke --count 43
python -m scripts.run budget
python -m scripts.run budget --per-class 10 --validation-per-class 30
```

`--per-class` chọn **nguyên nhóm**, nên ở train/validation một track có thể có khoảng 30 ảnh dù yêu cầu 1. Pilot không phải kết quả tổng quát hóa. Không dùng checkpoint `outputs/pilot_training_fixed/` với manifest final mới.

`budget` ước lượng cache float cho validation grid và test, cùng thời gian enhancer từ pilot. Cache toàn bộ có thể vượt dung lượng laptop; lệnh không tự chạy hoặc xóa cache. `freeze --per-class 10` khóa subset test chung cho tất cả phương pháp, vẫn đủ lớp; Để giới hạn validation một cách tái lập, dùng `tune --per-class 30`; toàn bộ phương pháp/CNN dùng cùng các nhóm validation này và selection lưu ID đầy đủ. Đây là hạn chế phạm vi chọn tham số, không phải đánh giá trên toàn validation. Có thể đặt dữ liệu/cache trên ổ khác bằng `--root /duong/dan/thu-muc` cho toàn bộ lệnh dữ liệu/train/tune/evaluate/report. Đường dẫn repo và checkpoint enhancer trong config được tính từ thư mục chạy lệnh.

## Thí nghiệm đầy đủ

Chỉ chạy sau khi rà dữ liệu/gallery, kiểm tra pretrained và ngân sách. Các lệnh sau chạy đủ seed/phương pháp trên **subset validation và subset test đã định trước**; bỏ hai cờ `--per-class` nếu đủ ngân sách cho toàn bộ tập. **Chưa chạy đầy đủ trong lần triển khai**:

```bash
python -m scripts.run train
python -m scripts.run tune --per-class 30 --checkpoints outputs/train/cnn_11.pt outputs/train/cnn_22.pt outputs/train/cnn_33.pt
python -m scripts.run freeze --per-class 10
python -m scripts.run evaluate
python -m scripts.run report
```

- `train`: ba seed 11/22/33, tối đa 40 epoch, patience 7; ghi learning curves CSV, checkpoint tốt nhất, confusion matrix validation và môi trường.
- `tune`: gamma 3 giá trị, CLAHE 6 cấu hình, MSR 2 cấu hình; các phương pháp khác dùng pretrained cố định. Trung bình ngang trọng số qua mức/noise/CNN seed. Chọn cấu hình riêng từng phương pháp theo F1; chọn pipeline chung trong vùng <0,5 điểm phần trăm so tốt nhất thì ưu tiên latency. Identity luôn được xét.
- `freeze`: khóa selection, sample ID test, config, SHA256 manifest/checkpoint, commit và hash mã enhancer. Đủ 3 seed và 8 phương pháp tạo **219 lượt điều kiện CNN**. Chưa có selection validation thì không chạy test. `--pilot` là nhãn bắt buộc nếu dùng thử nghiệm rút gọn.
- `evaluate`: dùng snapshot đã khóa, lưu predictions từng ảnh, xác suất lớp thật, metric ảnh, per-class F1/confusion, trạng thái planned/running/completed/failed. Lỗi dừng rõ; chạy lại cùng lệnh dùng cache đã kiểm chứng.
- `report`: kiểm tra đủ lượt và đúng tập ID rồi tạo Markdown, CSV, hình đường/scatter/confusion/learning, ví dụ lỗi, tương quan riêng từng mức và paired bootstrap 1.000 lần theo nhóm/source. Mean noise trong training seed trước, rồi SD giữa training seed; PSNR clean giữ vô hạn, loại clean khỏi tương quan.

Để dùng subset test đã định trước:

```bash
python -m scripts.run freeze --per-class 10 --output outputs/protocol_subset.json
python -m scripts.run evaluate --protocol outputs/protocol_subset.json --output outputs/test_subset
python -m scripts.run report --output outputs/test_subset
```

Không đổi code/config/checkpoint giữa freeze và evaluate. Nếu thay sau khi đã xem test, phải ghi ảnh hưởng đến độc lập đánh giá; không gọi test đó là tập chưa quan sát.

## File chính

| File | Trách nhiệm |
|---|---|
| `configs/default.json` | Giao thức mặc định, CNN, suy giảm, enhancer, metric |
| `src/data.py` | Download, split track, crop, manifest và audit |
| `src/degradation.py` | Thiếu sáng và nhiễu tái lập |
| `src/enhancement.py` | 4 enhancer cổ điển + 3 adapter tác giả |
| `src/model.py`, `src/training.py` | CNN và huấn luyện |
| `src/experiment.py` | Cache, validation search, freeze và evaluate |
| `src/metrics.py`, `src/reporting.py` | Metric, bootstrap, tổng hợp và hình |
| `scripts/run.py` | CLI chung |
| `checks/` | Kiểm tra toán học và workflow độc lập |

Phần chưa thực hiện: huấn luyện đầy đủ 3 CNN, chọn pipeline bằng validation đầy đủ, 219 lượt test, kết luận nghiên cứu và slide bảo vệ. Ablation và thí nghiệm B là mở rộng phụ, chưa đưa vào CLI. Không tạo số liệu hoặc kết luận để lấp phần chưa chạy.
