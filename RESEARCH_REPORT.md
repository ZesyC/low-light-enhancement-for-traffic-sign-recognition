# Tăng cường ảnh thiếu sáng cho phân loại biển báo giao thông

Báo cáo thực nghiệm GTSRB. Kết quả chính dùng classifier ResNet-18 train clean, ba seed và toàn bộ official test. Trọng tâm là so sánh xử lý ảnh với classifier cố định.

## 1. Bài toán và câu hỏi nghiên cứu

Đầu vào là crop biển báo RGB, đầu ra là một trong 43 lớp. Câu hỏi chính là ảnh tăng sáng có giúp classifier nhận đúng hơn so với cùng ảnh tối chưa xử lý hay không. Phạm vi này là phân loại crop, chưa bao gồm phát hiện vị trí biển báo trong toàn cảnh hoặc video.

RQ1: thiếu sáng và nhiễu làm giảm nhận diện bao nhiêu? RQ2: enhancer nào tốt trong cấu hình đã khảo sát? RQ3: PSNR/SSIM cao có đi cùng nhận diện tốt không? RQ4: tương quan thay đổi theo mức tối như thế nào? RQ5: pipeline chọn bằng validation có lợi ích và chi phí gì trên test?

## 2. Cơ sở xử lý ảnh và nghiên cứu liên quan

Gamma điều chỉnh giá trị pixel bằng hàm lũy thừa. HE cân bằng histogram kênh độ sáng Y trong YCrCb. CLAHE xử lý histogram theo ô cục bộ và giới hạn tương phản. Bản MSR trong dự án lấy trung bình chênh lệch log giữa ảnh và ảnh làm mờ ở nhiều sigma, sau đó chuẩn hóa theo percentile 1–99. Đây là cài đặt MSR đơn giản, không có bước color restoration của MSRCR.

Zero-DCE ước lượng đường cong tăng sáng bằng một mạng nhỏ, dùng các loss không cần ảnh tham chiếu khi huấn luyện. [Guo và cộng sự, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html).

Retinexformer sử dụng transformer dựa trên mô hình Retinex để phục hồi ảnh thiếu sáng. [Cai và cộng sự, ICCV 2023](https://github.com/caiyuanhao1998/Retinexformer). Wavelet-based Diffusion LLIE kết hợp diffusion có điều kiện trong miền wavelet và phục hồi thành phần tần số cao. [Jiang và cộng sự, ACM TOG 2023](https://arxiv.org/abs/2306.00306). Dự án dùng weights tác giả, không train lại các enhancer trên GTSRB.

## 3. Dữ liệu và giao thức

GTSRB gồm 39.209 ảnh final training và 12.630 ảnh official test. Dự án chia train/validation theo class và track: 31.350 train, 7.829 validation. Có 8 bản sao pixel chéo train–test trong một track lớp 14, nên cách ly cả track gồm 30 ảnh. [Thông tin benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html). Số mẫu và cách ly được kiểm kê từ manifest/audit của dự án.

Cắt ROI có padding cố định 5%, quy ước endpoint ghi trong manifest, resize RGB 64×64. Ảnh tối được tạo theo thứ tự gamma, Poisson, Gaussian rồi clip về [0,1].

| Mức | Gamma làm tối | Poisson peak | Gaussian sigma |
|---|---:|---:|---:|
| mild | 1.5 | 120 | 0.005 |
| medium | 2.2 | 60 | 0.01 |
| severe | 3.0 | 30 | 0.02 |

Train seed 11/22/33, noise seed 101/202/303. Chọn tham số và phương pháp bằng 1.290 ảnh validation trước test. Khóa checkpoint, config, sample ID, source hash và provenance. Mọi phương pháp dùng chung ID và ảnh tối cho từng điều kiện. Clean có ba lượt, không nhân bản theo noise seed.

Noise được trung bình trong từng training seed trước, rồi tính mean và sample SD qua ba training seed. Các tổ hợp training/noise không phải chín lần huấn luyện độc lập.

## 4. Cài đặt và kiểm chứng

ResNet-18 khởi tạo từ ImageNet, fine-tune toàn mạng với AdamW, lr 1e-4, weight decay 1e-4, batch 32, tối đa 20 epoch. Early stopping dùng patience 5, min_delta 0.001. Classifier resize toàn crop lên 224×224 rồi normalize ImageNet đúng một lần. Enhancer và metric ảnh vẫn ở 64×64.

Checkpoint clean tốt nhất ở epoch 12/10/3 cho seed 11/22/33. GPU là Tesla T4, torch 2.11.0+cu128 và torchvision 0.26.0+cu128 theo artifact môi trường. Load checkpoint offline không tải lại ImageNet weights.

Archive đã kiểm CRC và SHA256. Đã đối chiếu ID/nhãn và tính lại ba metric nhận diện từ 2.765.970 dự đoán. Tái tạo 43 crop validation từ raw cho pixel khớp chính xác với canonical PNG, và tái tạo cả ba mức suy giảm với float32 khớp chính xác. [Kiểm chứng test](outputs/resnet18_A/test_full/verification.json), [kiểm chứng raw replay](outputs/replay/verification.json).

## 5. Kết quả thực nghiệm

### 5.1. ResNet-18 clean, full official test

| Mức | Phương pháp | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---|---:|---:|---:|
| clean | clean | 99.15 ± 0.30 | 98.90 ± 0.40 | 99.13 ± 0.32 |
| medium | clahe | 80.91 ± 0.94 | 82.80 ± 1.46 | 83.13 ± 0.98 |
| medium | diffusion | 86.38 ± 2.62 | 83.87 ± 2.97 | 86.57 ± 2.46 |
| medium | gamma | 77.48 ± 0.92 | 79.77 ± 1.85 | 79.94 ± 1.04 |
| medium | he | 74.16 ± 4.57 | 77.27 ± 3.65 | 76.92 ± 5.54 |
| medium | identity | 75.40 ± 1.38 | 78.04 ± 1.79 | 77.56 ± 0.80 |
| medium | retinex | 53.07 ± 5.46 | 55.71 ± 5.31 | 59.06 ± 5.04 |
| medium | retinexformer | 78.68 ± 2.57 | 76.28 ± 2.31 | 79.79 ± 2.51 |
| medium | zero_dce | 69.53 ± 4.44 | 72.41 ± 3.12 | 73.79 ± 1.33 |
| mild | clahe | 95.51 ± 0.68 | 94.35 ± 1.12 | 95.72 ± 0.67 |
| mild | diffusion | 95.04 ± 1.33 | 93.45 ± 1.80 | 95.09 ± 1.28 |
| mild | gamma | 94.57 ± 0.95 | 93.49 ± 1.25 | 94.96 ± 0.82 |
| mild | he | 92.82 ± 2.89 | 92.31 ± 1.96 | 93.07 ± 2.76 |
| mild | identity | 93.06 ± 1.43 | 92.00 ± 1.71 | 93.43 ± 1.32 |
| mild | retinex | 81.41 ± 3.77 | 82.70 ± 3.12 | 83.94 ± 3.36 |
| mild | retinexformer | 86.52 ± 1.05 | 82.85 ± 1.80 | 86.50 ± 1.20 |
| mild | zero_dce | 90.47 ± 2.90 | 89.65 ± 2.37 | 91.74 ± 1.58 |
| severe | clahe | 59.44 ± 1.10 | 64.31 ± 3.47 | 64.50 ± 2.52 |
| severe | diffusion | 65.01 ± 3.97 | 64.88 ± 3.84 | 66.40 ± 4.29 |
| severe | gamma | 55.37 ± 2.59 | 60.48 ± 2.59 | 60.69 ± 1.42 |
| severe | he | 52.48 ± 5.28 | 56.98 ± 6.48 | 57.24 ± 8.03 |
| severe | identity | 56.12 ± 1.54 | 59.70 ± 1.64 | 60.18 ± 0.64 |
| severe | retinex | 31.27 ± 4.95 | 34.72 ± 5.71 | 37.41 ± 4.91 |
| severe | retinexformer | 55.60 ± 2.34 | 56.67 ± 1.15 | 59.70 ± 1.74 |
| severe | zero_dce | 46.44 ± 5.78 | 51.27 ± 4.60 | 52.85 ± 3.08 |

PSNR/SSIM và latency đầy đủ nằm trong [bảng tổng hợp](outputs/resnet18_A/test_full/aggregate.csv). PSNR của clean là vô hạn, SSIM là 1. Latency enhancer đo batch 1 với warm-up và đồng bộ GPU, gồm transfer/inference nhưng loại I/O và metric. Classifier đo riêng batch 32, khoảng 0,85–1,02 ms/ảnh cho ba seed. Không gọi latency enhancer là thời gian toàn pipeline.

![Macro-F1 theo mức](outputs/resnet18_A/test_full/figures/macro_f1_levels.png)

Validation chọn **diffusion**. Macro-F1 của pipeline này ở mild/medium/severe lần lượt 93.45%/83.87%/64.88%. Delta so identity lần lượt +1.45/+5.83/+5.18 điểm phần trăm. Clean và mild đạt cả ba metric >90%, medium/severe chưa đạt.

### 5.2. Ablation gamma và nhiễu

Classifier clean ResNet-18 cố định, tham số enhancer chọn trước bằng validation. 288 lượt trên cùng subset test 430 ảnh, ba seed CNN và ba noise seed. Thí nghiệm phụ này phân tích gamma và từng loại nhiễu, không dùng để đổi hyperparameter.

| Phương pháp | Gamma | Gamma + Gaussian | Gamma + Poisson | Gamma + cả hai |
|---|---:|---:|---:|---:|
| identity | 86.17 ± 0.98 | 83.46 ± 3.04 | 76.66 ± 2.70 | 75.35 ± 2.55 |
| gamma | 92.21 ± 0.83 | 87.35 ± 1.07 | 79.82 ± 2.40 | 76.81 ± 2.83 |
| he | 93.49 ± 1.53 | 85.07 ± 1.52 | 77.29 ± 5.86 | 75.25 ± 5.53 |
| clahe | 93.30 ± 1.01 | 88.20 ± 1.32 | 82.43 ± 1.06 | 80.14 ± 1.97 |
| retinex | 98.20 ± 0.61 | 73.33 ± 3.45 | 50.35 ± 4.57 | 50.97 ± 4.09 |
| zero_dce | 97.48 ± 1.36 | 87.13 ± 0.88 | 71.57 ± 5.03 | 69.30 ± 4.22 |
| retinexformer | 88.62 ± 0.53 | 82.80 ± 0.42 | 78.40 ± 3.02 | 73.63 ± 3.74 |
| diffusion | 95.04 ± 2.29 | 90.69 ± 0.92 | 85.38 ± 2.96 | 82.71 ± 3.08 |

Đơn vị Macro-F1: %. Noise được trung bình trong mỗi training seed trước, sau đó tính mean ± sample SD giữa ba seed.

Identity giảm từ 86.17% ở gamma-only xuống 75.35% khi thêm cả Poisson và Gaussian. Đây là ảnh hưởng trên mô phỏng, không suy ra camera đêm thực.

Cùng condition `medium` và seed được truyền vào hàm suy giảm. Khi tắt Poisson, chuỗi RNG trước bước Gaussian thay đổi, nên các biến thể không dùng cùng trường Gaussian pixel tuyệt đối.

Kết quả supplementary không thay thế bảng full-test 12.630 ảnh. Gamma-only không có nhiễu, ba noise seed tạo cùng ảnh và không được coi là ba phép đo độc lập.

## 6. Tương quan và phân tích lỗi

| Mức | Chất lượng ảnh | Pearson với Macro-F1 | Spearman với Macro-F1 | n |
|---|---|---:|---:|---:|
| mild | psnr_mean | 0.741 | 0.786 | 8 |
| mild | ssim_mean | 0.751 | 0.786 | 8 |
| medium | psnr_mean | 0.687 | 0.333 | 8 |
| medium | ssim_mean | 0.768 | 0.619 | 8 |
| severe | psnr_mean | 0.699 | 0.143 | 8 |
| severe | ssim_mean | 0.625 | 0.524 | 8 |

Mỗi phương pháp là một điểm, n=8, các điểm phụ thuộc vì dùng chung ảnh. Các tương quan là phân tích khám phá, không chứng minh quan hệ nhân quả. Ví dụ medium: Zero-DCE có PSNR 19,98 dB nhưng Macro-F1 72,41%, so diffusion 14,77 dB và 83,87%. Metric pixel cao không tự bảo đảm nhận diện tốt.

![SSIM và Macro-F1 mức vừa](outputs/resnet18_A/test_full/figures/scatter_medium_ssim_mean_macro_f1.png)

| Mức | Được giúp | Bị làm sai | Vẫn sai | Vẫn đúng |
|---|---:|---:|---:|---:|
| mild | 5090 | 2840 | 2801 | 102939 |
| medium | 15681 | 3201 | 12284 | 82504 |
| severe | 14589 | 4482 | 35289 | 59310 |

Số đếm trên chín tổ hợp training/noise mỗi mức, dùng để mô tả các lần dự đoán, không coi là số ảnh độc lập. Ví dụ chọn bằng sample ID cố định trong `example_ids.json`.

![Ví dụ giúp, làm hỏng, vẫn sai](outputs/resnet18_A/test_full/figures/failure_examples.png)

Năm lớp có F1 thấp nhất của pipeline đã chọn ở severe (mean qua ba noise và ba training seed):

| Class | Support test | F1 identity (%) | F1 diffusion (%) |
|---|---:|---:|---:|
| 30 | 150 | 24.30 | 25.46 |
| 42 | 90 | 12.78 | 33.71 |
| 6 | 150 | 32.85 | 37.56 |
| 24 | 90 | 26.81 | 42.53 |
| 10 | 660 | 30.65 | 45.58 |

Năm cặp nhầm lẫn thường gặp nhất của pipeline ở severe:

| Nhãn thật | Dự đoán sai | Số lần dự đoán |
|---:|---:|---:|
| 4 | 13 | 1293 |
| 10 | 13 | 1127 |
| 4 | 25 | 944 |
| 10 | 25 | 787 |
| 12 | 13 | 736 |

Support là số ảnh test của lớp, không nhân theo seed. Số nhầm lẫn cộng qua chín tổ hợp training/noise và không phải số ảnh độc lập.

Class 19 clean validation có F1=1, support 30 ở cả ba seed. Cần xem [per_class.csv](outputs/resnet18_A/test_full/per_class.csv) cho các mức tối. Paired bootstrap 1.000 lần trong `bootstrap.json` có điều kiện trên từng checkpoint. Official test thiếu track xác minh, nên bootstrap theo source_id chưa xử lý được mọi tương quan cùng biển.

## 7. Quy trình đề xuất, hạn chế và hướng phát triển

Trong giao thức A đã khảo sát, dùng tham số diffusion đã chọn bằng validation và classifier clean cố định. Diffusion có lợi ích nhận diện nhưng chậm khoảng 226–230 ms/ảnh; CLAHE khoảng 0,4 ms/ảnh và có Macro-F1 mild cao hơn trong quan sát test. Không đổi pipeline validation theo phương pháp thắng test.

Kết luận chỉ áp dụng cho GTSRB crop với thiếu sáng mô phỏng trên sRGB, chưa chứng minh camera đêm thực, mưa, lóa, nhòe chuyển động hoặc detection toàn cảnh. Thiếu sáng nặng có thể mất thông tin không phục hồi được. Các enhancer pretrained khác phân phối GTSRB. CNN cũ chỉ có bảng subset và thiếu artifact tương ứng trong workspace, nên không dùng bảng đó để kết luận paired cải thiện của ResNet full-test.

Hướng tiếp theo là phân tích tác động của xử lý ảnh lên nhiễu, màu sắc và chi tiết biển báo, sau đó kiểm tra dữ liệu đêm thực. Mục tiêu >90% ở mọi mức là mục tiêu cần đo, không phải kết quả đã đạt.

## Tài liệu và bàn giao

- [Hướng dẫn chạy](README.md), [kế hoạch](plan.md), [dữ liệu](DATASETS.md).
- [Slides HTML offline](outputs/resnet18_A/test_full/slides.html), [demo validation 43 lớp](outputs/research_evidence/demo/index.html).
- [GTSRB: Stallkamp và cộng sự, 2012](https://benchmark.ini.rub.de/gtsrb_news.html).
- [Zero-DCE: Guo và cộng sự, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html).
- [Retinexformer: Cai và cộng sự, ICCV 2023](https://github.com/caiyuanhao1998/Retinexformer).
- [Wavelet Diffusion: Jiang và cộng sự, ACM TOG 2023](https://arxiv.org/abs/2306.00306).
- [Multiscale Retinex: Jobson, Rahman và Woodell, 1997](https://ntrs.nasa.gov/api/citations/19990005051/downloads/19990005051.pdf).
- Config, environment, predictions, protocol và hashes nằm cùng thư mục kết quả. Source snapshot gốc A và receipt archive nằm trong `outputs/resnet18_A/`.
