# Tăng cường ảnh thiếu sáng cho phân loại biển báo giao thông

Bài tập lớn môn Xử lý ảnh khảo sát ảnh hưởng của **bảy phương pháp tăng cường thiếu sáng** đến khả năng phân loại 43 lớp biển báo trên GTSRB. ResNet-18 được huấn luyện trên ảnh gốc rồi khóa trọng số để so sánh các phương pháp trên cùng đầu vào.

Đầu vào là ảnh crop một biển báo; kết quả chỉ phản ánh **thiếu sáng mô phỏng**, chưa đánh giá phát hiện biển báo trong toàn cảnh hoặc camera đêm thực tế.

## Dữ liệu

Sử dụng **German Traffic Sign Recognition Benchmark (GTSRB)**, giữ đủ 43 lớp (ID 0–42). Nguồn tải là [archive của nhóm tác giả trên ERDA](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html), gồm `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip` và `GTSRB_Final_Test_GT.zip`.

| Tập | Số ảnh | Vai trò |
|---|---:|---|
| Train | 31.350 | Huấn luyện classifier trên ảnh gốc |
| Validation | 7.829 | Chọn checkpoint và cấu hình tăng cường |
| Quarantine | 30 | Cách ly toàn bộ track chứa 8 ảnh trùng pixel với test |
| Official test | 12.630 | Đánh giá cuối, giữ nguyên tập gốc |

Train/validation chia theo **class + track**, seed 42, trước khi sinh ảnh suy giảm. Crop theo ROI có padding 5%, resize RGB về 64×64; giữ cả biển nhỏ. Tập chọn enhancer gồm 1.290 ảnh validation; subset test/ablation gồm 430 ảnh (10 ảnh/lớp), được báo riêng với full-test.

## Pipeline và phương pháp

```text
Ảnh gốc → crop RGB 64×64 → gamma làm tối → nhiễu Poisson + Gaussian
       → tăng cường → PSNR/SSIM so với ảnh gốc
                   → ResNet-18 cố định → Accuracy / Macro-F1 / Weighted-F1
```

So sánh **Gamma, HE, CLAHE, Multi-Scale Retinex giản lược, Zero-DCE, Retinexformer và Wavelet-based Diffusion LLIE** với đối chứng `identity` (ảnh tối không xử lý). Ba phương pháp học sâu dùng trọng số pretrained của tác giả.

| Mức suy giảm | Gamma | Poisson peak | Gaussian sigma |
|---|---:|---:|---:|
| Nhẹ | 1,5 | 120 | 0,005 |
| Vừa | 2,2 | 60 | 0,010 |
| Nặng | 3,0 | 30 | 0,020 |

Mỗi phương pháp nhận cùng ảnh tối theo ID, mức và noise seed (101/202/303). ResNet-18 resize crop lên 224×224, normalize ImageNet và dùng ba training seed 11/22/33. Tham số enhancer và pipeline được chọn trên validation trước test. Ma trận chính gồm 219 lượt đánh giá; kết quả tổng hợp noise seed trong mỗi training seed trước, sau đó báo mean ± sample SD qua ba training seed.

## Kết quả chính

Trên **12.630 ảnh full-test**, ResNet-18 đạt Accuracy **99,22% ± 0,28** và Macro-F1 **98,94% ± 0,32** với ảnh gốc. Sau khảo sát tăng sáng, khử nhiễu và hai thứ tự kết hợp, validation chọn **Gaussian 5×5, sigma 0,8 rồi Gamma 0,8**. Macro-F1 (%) của các phương án đại diện:

| Phương pháp | Tối nhẹ | Tối vừa | Tối nặng |
|---|---:|---:|---:|
| Identity | 90,45 ± 3,38 | 76,07 ± 4,69 | 58,37 ± 7,77 |
| Diffusion | 93,63 ± 2,13 | 81,57 ± 4,98 | 62,46 ± 5,56 |
| Gaussian | 95,30 ± 1,93 | 81,85 ± 2,61 | 68,60 ± 2,74 |
| Gaussian rồi Gamma | 96,70 ± 1,16 | 85,54 ± 2,49 | 71,56 ± 2,01 |
| CLAHE rồi Gaussian | 96,57 ± 1,21 | 85,26 ± 2,60 | 71,77 ± 2,30 |

Chuỗi được chọn tăng **6,25 / 9,47 / 13,19 điểm phần trăm** so Identity. Không chọn lại theo ô test tốt nhất. Gaussian đơn lẻ ở mức nặng có PSNR/SSIM thấp hơn diffusion nhưng F1 cao hơn; chỉ số chất lượng ảnh không thay thế metric nhận dạng. Kết luận giới hạn ở ROI và thiếu sáng mô phỏng.

Số liệu, tương quan và biên bản kiểm tra ở [evidence/final](report/evidence/final/); báo cáo trình bày đầy đủ 8 phương pháp tăng sáng/đối chứng, 15 cấu hình khử nhiễu/kết hợp và giới hạn suy luận.

## Chạy thí nghiệm

Dùng một [notebook chính](notebooks/gtsrb-low-light-enhancement-for-traffic-signs.ipynb); [bản public Kaggle](https://www.kaggle.com/code/kieuthithutrang/gtsrb-low-light-enhancement-for-traffic-signs) phiên 2 đã hoàn tất phần mở rộng (khoảng 1 giờ 33 phút). Notebook dùng trực tiếp ZIP Output phiên 1, giữ ba checkpoint và kết quả baseline đã kiểm chứng; không tạo Dataset trung gian. Lỗi chuyển file lúc xuất ZIP đã được sửa. Notebook tải GTSRB từ tác giả; nguồn, commit và SHA-256 của ba enhancer được khóa trong [public_sources.json](configs/public_sources.json). Giữ giấy phép và ghi công của từng nguồn.

Notebook chứa sẵn 22 tệp mã nguồn/cấu hình và tự tạo chúng trong thư mục tạm khi chạy; không cần chạy notebook khác. Bản notebook đã chạy trước đây được giữ riêng trong `tmp/archived-notebooks/` để đối chiếu local, không đưa lên GitHub.

1. Bật GPU và Internet. Chạy cell thiết lập để bổ sung thư viện thiếu, giữ PyTorch/CUDA có sẵn. `requirements.txt` chứa phiên bản từ môi trường Python 3.14/macOS arm64 cũ, chưa kiểm tra lại trên Windows.
2. Chọn `TRAIN_MODE='reuse'`: notebook tự tải ZIP Output từ phiên `v1` của chính notebook qua API công khai, không tạo Dataset và không cần key. Có thể điền `CHECKPOINT_ARCHIVE` nếu ZIP đã có trong phiên. Kiểm tra checksum gói gốc, nội dung sau chuẩn hóa, seed, manifest và cấu hình; không khớp thì dừng, không tự train. Chọn `fresh` nếu muốn huấn luyện lại toàn bộ.
3. Chạy lần lượt kiểm tra enhancer → chuẩn bị dữ liệu → khôi phục/train CNN → chọn trên validation → subset → full-test → mở rộng → xuất ZIP. Dự đoán cũ chỉ dùng lại khi khớp mã, dữ liệu, cấu hình và checkpoint; nếu selection cũ không còn khớp thì chọn lại trên validation bằng CNN đã có.

Phần mở rộng bổ sung Gaussian, bilateral, NLM và hai thứ tự kết hợp với Gamma/CLAHE. Chọn độ mạnh bộ lọc trên validation rồi giữ giống nhau cho hai thứ tự; khóa phương pháp full-test trước khi đọc kết quả subset. Cấu hình nằm trong [processing_extension.json](configs/processing_extension.json). `RUN_EXTENSION_FULL_TEST=False` chỉ chạy subset mở rộng; không được gọi là full-test. Giữ nguyên CNN, mức suy giảm và seed.

Output chính là `gtsrb-notebook-results.zip`: mở `READ_ME_FIRST.md`, xem bảng tổng hợp, dự đoán CSV, ma trận NPZ và `SHA256.json`; phần mở rộng ở `08_processing_extension/`. ZIP không kèm ảnh nguồn, cache hoặc weights enhancer. Notebook xuất bản sao checkpoint và kết quả theo từng giai đoạn; tải chúng trước khi kết thúc phiên. Thời gian đối chứng dùng lại từ phiên khác chỉ mang tính mô tả.

## Phân tích và tái lập

Đã kiểm tra SHA từng file, độ phủ mẫu/nhãn/seed, tính lại Accuracy/Macro-F1/Weighted-F1 cho **5.847.990 dòng dự đoán** ở bốn phạm vi baseline/mở rộng × subset/full-test. Con số bao gồm đối chứng dùng lại, không phải số ảnh độc lập. Confusion matrix khớp ở 954 lượt có ma trận lưu trong ZIP; subset baseline được tính lại từ CSV. Ba checkpoint nhất quán giữa các phạm vi. Biên bản hiện tại ở [final/audit.json](report/evidence/final/audit.json). Chưa đánh giá ảnh đêm thực.


Từ thư mục gốc, có thể sinh lại notebook, xem lệnh thí nghiệm hoặc phân tích kết quả đã giải nén:

```powershell
python -m src.tools.build_kaggle_notebook
python -m src.tools.research --help
python -m src.tools.research errors --input outputs/recovered/notebook_A/test_subset --output outputs/recovered/notebook_A/test_subset/error_analysis_new
python -m src.tools.result_archive --help
python -m src.tools.verified_results --archive outputs/kaggle-results/gtsrb-notebook-results.zip --output outputs/verified-report --require-extension
```

Phân tích yêu cầu `protocol.json`, `runs.json`, `summary.csv`, `predictions/` và thư mục đầu ra mới; kiểm tra độ phủ/nhãn/metric trước khi xuất F1 từng lớp, confusion matrix và chuyển đúng/sai. Không cần train lại để phân tích CSV đã lưu. `order_comparisons.csv` của phần mở rộng ghi chênh lệch hai thứ tự: `post_minus_pre` dương nghĩa là tăng sáng rồi khử nhiễu có chỉ số cao hơn.

Khảo sát ảnh đêm thực dùng lệnh `python -m src.tools.real_lowlight --help`. Khi có ảnh, tạo `data/real_lowlight/manifest.csv` với dòng tiêu đề dưới đây rồi điền một dòng cho mỗi mẫu; hiện chưa có dữ liệu hoặc kết quả thực.

```csv
sample_id,image_path,class_id,group_id,x1,y1,x2,y2,source_url,license,annotator,capture_kind
```

Truyền đường dẫn đó qua `--manifest`, đặt `--image-root` là thư mục ảnh nguồn; `image_path` trong CSV tính từ thư mục này. Chỉ dùng ảnh chụp thật, nhãn GTSRB do người xác minh, có nguồn/quyền sử dụng; tọa độ ROI là xyxy với biên cuối exclusive, chưa padding, `group_id` chung cho cùng biển/chuỗi và `capture_kind=real_lowlight`. Không có ảnh tham chiếu ghép cặp thì không tính PSNR/SSIM; báo số lớp quan sát khi đọc F1.

## Mã nguồn và báo cáo

| Đường dẫn | Nội dung |
|---|---|
| [src/](src/) | Xử lý ảnh, dữ liệu, CNN và metric; lệnh chạy/đóng gói ở [src/tools/](src/tools/) |
| [configs/](configs/) | Cấu hình thí nghiệm và nguồn enhancer |
| [tests/](tests/) | Kiểm tra pipeline và tính toàn vẹn kết quả |
| [Notebook chính](notebooks/gtsrb-low-light-enhancement-for-traffic-signs.ipynb) | Toàn bộ quy trình Kaggle, chế độ fresh/reuse |
| [Báo cáo](report/bao_cao.pdf) · [Slide](report/thuyet_trinh.pdf) | PDF cuối; mã LaTeX, hình và bảng nguồn trong [report/](report/README.md) |

Biên dịch PDF bằng `powershell -File report/latex/build.ps1`; tạo lại hình/bảng từ ZIP đã kiểm chứng theo [hướng dẫn báo cáo](report/README.md). Hướng dẫn font và compiler nằm trong [report/README.md](report/README.md).

Dữ liệu, checkpoint, weights bên ngoài, kết quả thô, cache và bản nháp lưu local ở `data/`, `external/`, `outputs/`, `tmp/`; Git giữ mã, cấu hình, notebook và bằng chứng gọn. Lockfile macOS cũ cũng chỉ lưu local.
