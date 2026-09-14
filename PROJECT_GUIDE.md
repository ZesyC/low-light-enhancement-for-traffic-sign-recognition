# Định hướng dự án: Tăng cường ảnh thiếu sáng cho nhận diện biển báo giao thông

Ngày lập: 14/09/2026. Trạng thái: đã triển khai pipeline và smoke test; chưa có kết quả thực nghiệm cuối. Xem [README.md](README.md) để chạy.

Tài liệu này quy định mục tiêu và giao thức đánh giá xuyên suốt dự án. Xem [DATASETS.md](DATASETS.md) để chọn/tải dữ liệu và [plan.md](plan.md) để triển khai. Mọi thông số ghi “đề xuất” là quyết định thiết kế ban đầu, không phải kết quả hoặc cấu hình tối ưu đã được chứng minh.

## 1. Mục tiêu và phạm vi

Câu hỏi chính: tăng cường ảnh thiếu sáng có giúp một CNN phân loại biển báo chính xác hơn không, phương pháp nào phù hợp ở từng mức suy giảm, và PSNR/SSIM có phản ánh đúng lợi ích nhận diện không?

Đầu vào là ảnh đã chứa một biển báo được cắt vùng quan tâm (ROI); đầu ra là lớp biển báo. Đây là **phân loại ảnh**, không phải phát hiện biển báo trong toàn cảnh. Với GTSRB, dùng ROI nhãn thật theo quy tắc crop cố định; không huấn luyện YOLO và không tuyên bố kết quả áp dụng cho hệ thống phát hiện đầu-cuối.

Sản phẩm cần có:

1. Dữ liệu và split có thể tái tạo, sử dụng duy nhất GTSRB.
2. Bộ sinh suy giảm gamma + Gaussian + Poisson có ít nhất ba mức.
3. Bảy phương pháp: gamma correction, HE, CLAHE, Retinex cổ điển, Zero-DCE, Retinexformer, một mô hình diffusion cho tăng cường thiếu sáng.
4. Bảng PSNR, SSIM, Accuracy, Macro-F1 và chi phí suy luận trên cùng tập ảnh.
5. Phân tích tương quan, trường hợp thất bại và quy trình tiền xử lý được chọn bằng validation.

Không đặt trước yêu cầu phương pháp học sâu phải thắng hoặc tăng cường luôn có lợi. Kết quả “ảnh sáng hơn nhưng nhận diện kém hơn” vẫn trả lời đúng câu hỏi nghiên cứu.

## 2. Bộ dữ liệu duy nhất: GTSRB

Dùng **German Traffic Sign Recognition Benchmark (GTSRB)** đúng mô tả đề bài. Đây là bộ dữ liệu phân loại biển báo giao thông của Đức với 43 lớp. Nguồn: [trang benchmark](https://benchmark.ini.rub.de/) và [bài báo của nhóm GTSRB](https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf).

Giữ official test làm tập đánh giá cuối; tách validation từ official train theo track biển báo. CNN có số đầu ra cố định K=43. Tất cả phương pháp dùng cùng split, ảnh tham chiếu và hiện thực nhiễu.

Xem [DATASETS.md](DATASETS.md) để tải và kiểm kê. Ảnh gốc được dùng như ảnh tham chiếu, không mặc định là ảnh sạch nhiễu hay đủ sáng tuyệt đối. Dự án mô phỏng suy giảm từ ảnh gốc và đánh giá khả năng phục hồi thông tin phục vụ phân loại.

## 3. Câu hỏi và giả thuyết nghiên cứu

- RQ1: Accuracy/Macro-F1 giảm thế nào theo mức suy giảm?
- RQ2: Phương pháp nào phục hồi nhận diện tốt nhất với một bộ phân loại cố định?
- RQ3: Phương pháp nào cải thiện PSNR/SSIM nhưng làm mất màu sắc, chữ số hoặc đường viền quan trọng?
- RQ4: Quan hệ giữa chất lượng ảnh và nhận diện có còn tồn tại khi phân tích riêng theo mức suy giảm và bộ dữ liệu?
- RQ5: Phương án nào cân bằng nhận diện, tốc độ, bộ nhớ và tính đơn giản?

Giả thuyết để kiểm chứng: tăng độ sáng có thể đồng thời khuếch đại nhiễu; chất lượng ảnh tốt hơn không bảo đảm nhận diện tốt hơn; phương pháp tốt nhất có thể thay đổi theo mức suy giảm. Không trình bày các giả thuyết này như kết luận.

## 4. Dữ liệu, crop và chống rò rỉ

### 4.1. Thứ tự xử lý bắt buộc

Ảnh nguồn → kiểm kê/nhãn → gom nhóm cùng cảnh hoặc cùng biển → chia train/validation/test → crop → chuẩn hóa kích thước → sinh suy giảm → tăng cường → tính chất lượng và phân loại.

Chia split trước khi sinh phiên bản tối, nhiễu hoặc augmentation. Mọi crop/biến thể của cùng ảnh nguồn nằm cùng split. Các ảnh cùng track GTSRB phải ở cùng split; không chia ngẫu nhiên từng frame trong track.

GTSRB: giữ official test; tách khoảng 15–20% official train làm validation theo nhóm track và cân đối lớp trong khả năng cho phép. ID nhóm phải chứa cả class và track. Các ảnh cùng track có liên hệ gần; xem [mô tả GTSRB](https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf).

Không chia lại official test. Rà ảnh trùng bằng checksum và ảnh gần trùng; giữ tên file/track gốc khi tạo manifest. Nếu phát hiện vấn đề ở test, ghi nhận và báo rõ thay vì âm thầm loại mẫu.

### 4.2. Chính sách nhãn

Giữ đủ 43 lớp GTSRB với class ID 0–42. Lưu nhãn gốc và bảng tên lớp theo nguồn benchmark; không đổi mã hoặc gộp lớp.

Báo số lượng theo lớp và split. Khi tách validation, bảo đảm cả train và validation có đủ 43 lớp mà không làm vỡ track; điều chỉnh tỷ lệ nhóm nếu cần. Không loại lớp hiếm hoặc lớp mô hình dự đoán kém.

Crop đề xuất: box gốc cộng 5% mỗi phía, giới hạn trong ảnh; dùng cùng quy tắc mọi mẫu. Kiểm tra và xử lý ảnh lỗi/box không hợp lệ trước khi chạy; không âm thầm bỏ mẫu test. Khóa tỷ lệ padding trước khi test và giữ cả biển nhỏ để phản ánh độ khó của benchmark.

### 4.3. Manifest tối thiểu

Một dòng cho mỗi crop: `sample_id, dataset, source_path, source_id, group_id, bbox, original_label, class_id, split, width, height, source_sha256`.

Manifest biến thể bổ sung: `sample_id, condition, gamma_dark, poisson_peak, gaussian_sigma, noise_seed, path`. Lưu đường dẫn tương đối; không dùng tên file đơn lẻ làm ID toàn cục.

## 5. Mô phỏng thiếu sáng

### 5.1. Quy ước toán học

Đưa RGB về float32 trong [0,1], ký hiệu ảnh tham chiếu là x. Mô phỏng trên sRGB đã giải mã, không tuyên bố là mô hình vật lý RAW chính xác:

\[
z=x^{\gamma_d},\qquad p=\operatorname{Poisson}(Pz)/P,\qquad y=\operatorname{clip}(p+\epsilon,0,1),\quad \epsilon\sim\mathcal N(0,\sigma^2).
\]

Trong đó gamma_d > 1 làm tối; P > 0 là mức đếm hiệu dụng, P thấp làm nhiễu Poisson tương đối mạnh hơn; sigma là độ lệch chuẩn Gaussian trên thang [0,1]. Sinh Poisson theo tín hiệu, không cộng trực tiếp một biến Poisson không trừ trung bình vào ảnh. Khi muốn tắt Poisson, đặt p=z; khi tắt Gaussian, sigma=0.

| Mức đề xuất | gamma_d | P | sigma |
|---|---:|---:|---:|
| Nhẹ | 1.5 | 120 | 0.005 |
| Vừa | 2.2 | 60 | 0.010 |
| Nặng | 3.0 | 30 | 0.020 |

Đây là lưới khởi đầu để pilot trên train/validation, không phải chuẩn phổ quát. Kiểm tra ảnh, độ sáng trung bình và tỷ lệ pixel bị clip trước khi khóa. Do clipping và nhiễu, không bắt buộc mọi ảnh riêng lẻ đều có metric giảm đơn điệu.

### 5.2. Kích thước và tính tái lập

Giao thức chính đề xuất: resize crop RGB trực tiếp về 64×64 bằng cùng phép nội suy, lưu thành ảnh tham chiếu canonical; sinh suy giảm sau resize để nhiễu không bị resize làm mượt. Đây là đánh giá ở độ phân giải crop, không đại diện toàn bộ pipeline camera.

Nếu enhancer cần kích thước lớn hơn hoặc bội số cụ thể, ưu tiên pad rồi bỏ pad. Nếu buộc resize, ghi riêng phép resize; tính metric và đưa CNN về đúng 64×64. Mọi phương pháp phải nhận cùng y, không được tái sinh nhiễu khi đổi phương pháp.

Dùng ba noise seed đề xuất: 101, 202, 303. Seed từng ảnh được sinh ổn định từ seed gốc + sample_id + mức suy giảm; không dùng `hash()` mặc định Python. Lưu cấu hình hoặc chính ảnh đầu vào để tái lập. Tránh JPEG sau suy giảm; lưu float hoặc ảnh lossless và ghi rõ mức lượng tử hóa.

Ablation phụ ở mức vừa: gamma-only, gamma+Gaussian, gamma+Poisson, gamma+Gaussian+Poisson. Ba mức chính thay đổi đồng thời nhiều tham số nên không dùng riêng bảng chính để kết luận tác động độc lập của từng loại nhiễu.

## 6. Các phương pháp so sánh

| ID | Phương pháp | Cách triển khai trong dự án |
|---|---|---|
| clean | Ảnh tham chiếu | Baseline trên ảnh gốc, không gọi là trần tuyệt đối |
| identity | Ảnh tối không xử lý | Baseline bắt buộc cho mọi phép tính cải thiện |
| gamma | Gamma correction | x_hat = y^gamma_e, 0 < gamma_e < 1 |
| he | Histogram Equalization | Equalize kênh Y trong YCrCb, giữ Cr/Cb |
| clahe | CLAHE | Xử lý kênh Y, giữ màu; khóa clipLimit và tileGridSize |
| retinex | Multi-Scale Retinex (MSR) | Tự cài đặt biến thể xác định rõ bên dưới |
| zero_dce | Zero-DCE | Tích hợp mã/weights tác giả, inference cố định |
| retinexformer | Retinexformer | Checkpoint công khai phù hợp sRGB, inference cố định |
| diffusion | Wavelet-based Diffusion LLIE | Tích hợp mô hình của Jiang và cộng sự, ghi checkpoint và sampler |

HE/CLAHE dựa trên [tài liệu OpenCV](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html). Áp dụng riêng độ chói là lựa chọn của dự án để hạn chế lệch màu, không phải đảm bảo màu luôn được bảo toàn hoàn hảo.

MSR đề xuất trên từng kênh RGB: trung bình có trọng số bằng nhau của `log(x+eps) - log(G_sigma*x+eps)`, eps=1e-6, ba sigma khởi đầu [3, 10, 20] pixel ở ảnh 64×64. Đưa về [0,1] bằng percentile 1–99 chung trên toàn tensor RGB, clip; ảnh có khoảng percentile gần 0 thì trả đầu vào. Ghi tên đầy đủ “MSR + percentile rescaling”, không gọi là MSRCR vì chưa cài color restoration. Nền tảng: [bài báo MSR tại NASA](https://ntrs.nasa.gov/api/citations/19990005051/downloads/19990005051.pdf). Các sigma và rescaling trên là thiết kế thử nghiệm riêng.

Lưới validation nhỏ: gamma_e ∈ {0.4,0.6,0.8}; CLAHE clipLimit ∈ {1,2,4}, tileGridSize ∈ {(4,4),(8,8)}; MSR thử tối đa hai bộ sigma ([3,10,20] và [5,15,30]). Chọn **một cấu hình chung cho các mức** bằng Macro-F1 trung bình trên validation. Không dùng gamma tạo tối thật để tự động lấy nghịch đảo trong bảng chính; biến thể đó chỉ được báo riêng như oracle.

Nguồn mô hình học sâu:

- [Zero-DCE chính thức](https://github.com/Li-Chongyi/Zero-DCE): có snapshot và script inference, mã cho nghiên cứu/phi thương mại. Giữ bản Zero-DCE gốc thay vì âm thầm thay bằng Zero-DCE++.
- [Retinexformer chính thức](https://github.com/caiyuanhao1998/Retinexformer): có mã và weights. Tắt mọi điều chỉnh độ sáng sử dụng ground truth; README tác giả cũng chỉ rõ một số chế độ đánh giá có dùng mean ảnh tham chiếu.
- [Wavelet-based Diffusion chính thức](https://github.com/JianghaiSCU/Diffusion-Low-Light): có đường dẫn pretrained và `evaluate.py`; chọn làm đại diện diffusion cho LLIE. Việc link xuất hiện không có nghĩa đã tải/chạy thành công.

Ưu tiên dùng pretrained cho cả ba phương pháp, không train transformer/diffusion từ đầu. Khóa tên dataset pretraining, checkpoint SHA256, commit mã, range đầu vào, padding, số bước và seed suy luận. Nếu thiếu checkpoint hoặc không chạy được, ghi “chưa hoàn thành”; không thay bằng lọc ảnh rồi gọi là diffusion.

## 7. CNN và giao thức so sánh công bằng

### 7.1. Kiến trúc đề xuất

Một CNN nhỏ tự huấn luyện để tránh thêm ảnh hưởng pretraining:

`[B,3,64,64] → Conv3×3(32,pad=1)+ReLU+MaxPool2 → Conv3×3(64,pad=1)+ReLU+MaxPool2 → Conv3×3(128,pad=1)+ReLU → AdaptiveAvgPool(1) → Flatten → Linear(128,43)`.

Đầu ra logits có shape [B,43]; nhãn có shape [B] kiểu số nguyên. Cross-entropy nhận logits trực tiếp. Đề xuất Adam, learning rate 1e-3, batch size 64, tối đa 40 epoch, dừng sớm sau 7 epoch không cải thiện Macro-F1 validation. Chọn checkpoint tốt nhất trên validation. Các giá trị này cần pilot, không bảo đảm Accuracy cụ thể.

Nếu chuẩn hóa theo mean/std, chỉ ước lượng từ ảnh train gốc rồi dùng cố định mọi điều kiện. Không flip biển theo chiều ngang vì có thể đổi ý nghĩa. Baseline ban đầu không augmentation độ sáng để giữ cách diễn giải rõ.

### 7.2. Thí nghiệm chính A: khóa trọng số

Huấn luyện CNN bằng ảnh train gốc; chọn checkpoint bằng validation gốc, sau đó khóa CNN. Với từng seed huấn luyện, chính checkpoint đó nhận clean, dark và toàn bộ ảnh enhanced. CNN ở chế độ eval; enhancer không được nhìn nhãn test hoặc ảnh tham chiếu test.

Đề xuất ba training seed 11, 22, 33. Huấn luyện ba lần trên cùng split GTSRB, giữ nguyên kiến trúc và thủ tục, chỉ thay seed. Chọn tham số enhancer trên validation bằng trung bình qua seed và mức, sau khi đã chọn checkpoint CNN. Ghi rõ validation được dùng cho hai quyết định này.

### 7.3. Thí nghiệm phụ B: thích nghi dữ liệu

Chỉ thực hiện sau khi A hoàn tất: train cùng kiến trúc bằng ảnh tối hoặc ảnh được tăng cường, dùng cùng ngân sách/seed/split; test đúng phân phối tương ứng. Đây là đánh giá phối hợp huấn luyện–tiền xử lý; phải báo riêng vì trọng số đã khác. Không trộn các hàng A và B để kết luận tác động riêng của enhancer.

## 8. Chỉ số và cách tính

### 8.1. Chất lượng ảnh

So sánh y hoặc x_hat với đúng x cùng sample_id, cùng kích thước, cùng [0,1]. PSNR = 10 log10(1/MSE); MSE tính trên RGB. SSIM dùng `data_range=1.0, channel_axis=-1, win_size=7` và khóa các tham số còn lại trong cấu hình. Tính từng ảnh rồi trung bình, không suy PSNR trung bình từ MSE toàn bộ dataset. Tham khảo [API scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html).

Clean so chính nó có PSNR vô hạn và SSIM=1: báo như sanity check, loại khỏi tính tương quan. Không gán PSNR vô hạn thành một số hữu hạn tùy ý. Với ảnh đêm thực không có ảnh tham chiếu khớp, để PSNR/SSIM là N/A, chỉ đo nhận diện và phân tích trực quan.

### 8.2. Nhận diện

Báo Accuracy và Macro-F1 làm hai chỉ số chính; Weighted-F1, F1 từng lớp, confusion matrix là bổ sung. Macro-F1 lấy trung bình F1 các lớp để tránh lớp đông lấn át. Khóa danh sách 43 lớp GTSRB và dùng `zero_division=0`; báo support từng lớp. Xem [định nghĩa F1 của scikit-learn](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html).

Báo chênh lệch Accuracy/Macro-F1 bằng **điểm phần trăm** so với identity cùng mức/seed. Không so một phương pháp ở mức nhẹ với baseline mức nặng.

Lưu dự đoán từng ảnh và metric ảnh riêng để không phải chạy lại enhancer khi đổi cách tổng hợp. Thời gian enhancer đo riêng CNN, warm-up trước và đồng bộ GPU; ghi thiết bị, batch size, độ phân giải, precision, số bước diffusion; phân biệt thời gian tính và đọc/ghi file.

### 8.3. Tương quan và độ bất định

- Phân tích chính trên GTSRB theo từng mức suy giảm: mỗi phương pháp là một điểm (PSNR trung bình, Accuracy), tương tự SSIM và Macro-F1. Báo Pearson và Spearman, n điểm và biểu đồ. Tám phương pháp tính cả identity cho ít điểm, kết quả chỉ mang tính khám phá.
- Phân tích thay đổi: ΔPSNR/ΔSSIM so identity đối chiếu ΔAccuracy/ΔMacro-F1. Không coi các phương pháp dùng chung ảnh là mẫu độc lập.
- Biểu đồ gộp nhiều mức chỉ là bổ sung; xu hướng có thể do mức suy giảm chứ không do enhancer. Không suy tương quan thành quan hệ nhân quả.
- Ở cấp ảnh, có thể phân tích thay đổi xác suất lớp thật/đúng-sai; không gán “Accuracy từng ảnh” như một độ chính xác liên tục.
- Báo mean ± std qua training seed và noise seed với cách gộp rõ: tổng hợp noise seed trong mỗi training seed trước, rồi báo biến thiên training seed; thêm bảng độ nhạy noise seed. Các tổ hợp này không phải 9 thí nghiệm độc lập hoàn toàn.
- Đề xuất paired bootstrap 1.000 lần cho chênh lệch so identity, lấy lại mẫu theo group_id; khi không có nhóm test thì theo source_id và ghi giới hạn. Giữ các phiên bản của cùng ảnh cùng mẫu bootstrap. CI này có điều kiện trên checkpoint, không thay thế biến thiên huấn luyện.

## 9. Ma trận thí nghiệm và ngân sách

Trên GTSRB, một training seed, một noise seed: 1 clean + 3 mức × (identity + 7 enhancer) = 25 lượt đánh giá điều kiện. Ba training seed × ba noise seed: clean chỉ cần chạy một lần/seed, tức 3 + 3×3×3×8 = 219 lượt đánh giá CNN trên tập test GTSRB. Tăng cường được cache và dùng lại giữa các CNN; PSNR/SSIM không phụ thuộc training seed.

Tổng cộng 3 lần huấn luyện CNN chính và 219 lượt điều kiện đánh giá CNN trên GTSRB. Một “lượt điều kiện” gồm nhiều ảnh, không phải một lần forward. Ngân sách thực tế phải đo sau pilot; không thể suy thời gian từ số lượt đơn thuần.

Diffusion pilot sớm trên một batch để đo thời gian/VRAM. Nếu không đủ tài nguyên chạy toàn test, định trước một subset theo lớp và nhóm bằng seed, chạy **tất cả phương pháp trên cùng subset** để tạo bảng so sánh đầy đủ; thêm bảng toàn test cho phương pháp chạy được. Ghi độ phủ và số mẫu từng lớp; không đặt kết quả khác số mẫu cạnh nhau như tương đương.

Giảm training/noise seed xuống một chỉ cho pilot; kết quả một seed phải công khai giới hạn. Không tự bỏ phương pháp bắt buộc khỏi sản phẩm cuối.

## 10. Chọn quy trình tiền xử lý

Quy tắc đề xuất: chọn phương pháp/cấu hình có Macro-F1 validation trung bình cao nhất trên ba mức; báo cả kết quả từng mức. Khi chênh lệch dưới 0.5 điểm phần trăm, ưu tiên thời gian suy luận thấp hơn. Mốc 0.5 là quy tắc vận hành đặt trước, không phải kiểm định tương đương thống kê.

Luôn cho phép identity được chọn. Khóa quyết định trước khi mở bảng test. Chọn một quy trình chung cho GTSRB dựa trên trung bình ngang trọng số giữa ba mức suy giảm.

Không xây bộ tự nhận biết mức tối ở giai đoạn chính. Một chính sách chọn phương pháp theo mức suy giảm biết trước chỉ là oracle; hệ thống thực tế cần bộ ước lượng từ đầu vào và thí nghiệm riêng.

Kết luận chỉ áp dụng cho loại biển, split, mô phỏng, độ phân giải và CNN đã kiểm chứng. Chưa đủ để khẳng định hoạt động tốt trên camera ngoài đường ban đêm, chói đèn, mưa, motion blur hoặc đối tượng chưa có trong tập lớp.

## 11. Tổ chức triển khai dự kiến

Mã đã được triển khai trong `src/`, CLI tại `scripts/run.py`, kiểm tra tại `checks/`. Cấu trúc định hướng ban đầu:

```text
PROJECT_GUIDE.md
DATASETS.md
plan.md
configs/                  # Split, degradation, enhancer, CNN, evaluation
src/data.py               # Manifest, split, crop, loader
src/degradation.py        # Gamma và hai loại nhiễu
src/enhancement.py        # Phương pháp cổ điển và adapter
src/model.py              # Một CNN dùng chung kiến trúc
scripts/                  # Prepare, train, enhance, evaluate, report
checks/                   # Các kiểm tra nhỏ có thể chạy độc lập
notebooks/                # EDA và xem kết quả; logic chính nằm ở src
data/                     # Raw, manifests, processed, cache; không commit ảnh
outputs/                  # Checkpoint, predictions, metrics, figures, logs
```

Tạo thư mục khi bắt đầu phần việc tương ứng; chưa cần scaffold toàn bộ. Dùng Python/NumPy/OpenCV cho xử lý ảnh, PyTorch cho CNN và mô hình học sâu, scikit-image/scikit-learn cho metric, matplotlib cho hình. Ghim phiên bản sau smoke test; mô hình bên ngoài có thể cần môi trường riêng và trao đổi ảnh lossless. Không nâng/hạ toàn bộ môi trường chỉ để chạy một repo cũ.

Bảng tổng hợp tối thiểu: `dataset, split, protocol, subset_id, condition, method, config_id, train_seed, noise_seed, n_images, accuracy, macro_f1, psnr_mean, ssim_mean, latency_ms`.

Log thí nghiệm kèm ngày chạy, config, commit nếu có Git, environment, checkpoint hash, split hash. Đánh dấu rõ planned/running/completed/failed; không điền số dự kiến vào bảng kết quả thật.

## 12. Tiêu chí hoàn thành và báo cáo

Dự án đạt yêu cầu khi có dữ liệu và thí nghiệm GTSRB đúng yêu cầu gốc, bảy phương pháp thực chạy, baseline clean/dark, ba mức suy giảm, metric ảnh và nhận diện trên tập chung, phân tích tương quan có giới hạn, quy trình được chọn trên validation và kết quả tái tạo được.

Bố cục báo cáo: (1) bài toán/câu hỏi; (2) cơ sở xử lý ảnh và nghiên cứu liên quan; (3) dữ liệu và giao thức; (4) cài đặt; (5) kết quả; (6) tương quan và lỗi; (7) quy trình đề xuất, hạn chế, hướng phát triển.

Hình bắt buộc: ví dụ ba mức tối; cùng một crop qua mọi phương pháp; phân bố lớp; đường Accuracy/F1 theo mức; scatter PSNR/SSIM với nhận diện; confusion matrix và nhóm trường hợp nhận diện bị cải thiện/làm hỏng. Chọn ví dụ theo quy tắc và cố định sample_id, không chỉ chọn ảnh đẹp.

Mọi thay đổi sau pilot cần ghi ngày, lý do, phần giao thức bị ảnh hưởng. Nếu đã dùng test để chọn tham số, phải công khai và tổ chức tập kiểm tra mới độc lập trước khi tuyên bố kết quả đánh giá cuối.
