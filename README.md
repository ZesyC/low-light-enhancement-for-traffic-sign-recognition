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

Trên full-test **12.630 ảnh**, ResNet-18 đạt Accuracy **99,15% ± 0,30** và Macro-F1 **98,90% ± 0,40** với ảnh gốc. Bảng dưới là Macro-F1 (%) cho đối chứng, CLAHE và pipeline diffusion đã chọn trên validation:

| Phương pháp | Tối nhẹ | Tối vừa | Tối nặng |
|---|---:|---:|---:|
| Identity | 92,00 ± 1,71 | 78,04 ± 1,79 | 59,70 ± 1,64 |
| CLAHE | 94,35 ± 1,12 | 82,80 ± 1,46 | 64,31 ± 3,47 |
| Diffusion | 93,45 ± 1,80 | 83,87 ± 2,97 | 64,88 ± 3,84 |

Diffusion tăng Macro-F1 so với identity **1,45 / 5,83 / 5,18 điểm phần trăm** ở ba mức. CLAHE có chi phí thấp hơn: khoảng 0,4 ms/ảnh so với 226–230 ms/ảnh của diffusion trên Tesla T4, chỉ tính thời gian enhancer. Pipeline đã chọn chưa đạt mục tiêu cả Accuracy, Macro-F1 và Weighted-F1 trên 90% ở mức vừa/nặng; chất lượng pixel cao hơn cũng không bảo đảm nhận diện tốt hơn.

Bảng đầy đủ nằm trong [CSV full-test](report/evidence/resnet18_fulltest_readme.csv); phân tích subset, ablation và giới hạn thí nghiệm nằm trong báo cáo.

## Mã nguồn và tài liệu

| Đường dẫn | Nội dung |
|---|---|
| [src/](src/) | Chuẩn bị dữ liệu, mô phỏng suy giảm, enhancer, training, metric và tổng hợp kết quả |
| [configs/resnet18.json](configs/resnet18.json) | Cấu hình thí nghiệm chính |
| [Notebook đã thực thi](notebooks/GTSRB_LLIE_Complete_Pipeline_Executed.ipynb) | Pipeline và kết quả subset/ablation có output |
| [Báo cáo PDF](report/bao_cao.pdf) | Nội dung và phân tích đầy đủ |
| [report/](report/README.md) | Báo cáo, slide PDF, số liệu và hướng dẫn biên dịch local |
| [Slide PDF](report/thuyet_trinh.pdf) | Tài liệu thuyết trình |

Cài các thư viện để làm việc với mã nguồn:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dataset, checkpoint, mã/weights bên ngoài và artifacts trong `data/`, `external/`, `outputs/` được lưu riêng, không có sẵn khi clone. `scripts/`, `checks/`, `PROJECT_GUIDE.md` và `DATASETS.md` chỉ giữ ở local. Notebook đã thực thi dùng để xem quy trình và kết quả; chạy lại toàn bộ cần bổ sung dữ liệu, weights và các công cụ local tương ứng. Khi chạy trên Kaggle, dùng bộ PyTorch/CUDA phù hợp môi trường thay vì cài đè bộ thư viện macOS trong lockfile.
