# Kế hoạch thực hiện bài tập lớn

Ngày lập: 14/09/2026. Khung đề xuất: 8 tuần; điều chỉnh theo hạn nộp và tài nguyên thực tế. Pipeline đã được cài đặt và kiểm tra kỹ thuật; các checkbox dưới đây vẫn là tiêu chí nghiệm thu nghiên cứu, không tự động hoàn thành chỉ vì đã có code. Xem README.md.

Đọc [PROJECT_GUIDE.md](PROJECT_GUIDE.md) trước để nắm giao thức, [DATASETS.md](DATASETS.md) để tải và kiểm kê GTSRB. Dự án chỉ dùng GTSRB đúng mô tả đề bài, giữ đủ 43 lớp và official test.

## 1. Kết quả phải bàn giao

- [ ] Dữ liệu GTSRB có nguồn, phiên bản, nhãn, điều kiện sử dụng và manifest.
- [ ] Train/validation/test tách theo nhóm, không rò rỉ ảnh gốc/biến thể.
- [ ] Suy giảm gamma + Gaussian + Poisson ở ba mức đã khóa.
- [ ] CNN chung kiến trúc, checkpoint và log huấn luyện.
- [ ] Clean, dark và đủ bảy phương pháp tăng cường thực chạy.
- [ ] Bảng PSNR, SSIM, Accuracy, Macro-F1 trên cùng tập mẫu.
- [ ] Phân tích tương quan, chi phí, bất định và ví dụ thất bại.
- [ ] Quy trình tiền xử lý chọn trên validation, kiểm chứng trên test.
- [ ] Báo cáo, slide và hướng dẫn chạy lại.

## 2. Trình tự và điểm kiểm soát

`Dữ liệu hợp lệ → split khóa → suy giảm kiểm chứng → pilot CNN/enhancer → cấu hình khóa → validation chọn pipeline → test cuối → phân tích → báo cáo`.

Thử khả năng chạy pretrained và diffusion ngay tuần 1–2, không đợi đến cuối mới kiểm tra checkpoint/GPU. Không chạy ma trận test lớn trước khi các sanity check đạt.

| Tuần | Công việc trọng tâm | Sản phẩm kiểm tra được |
|---|---|---|
| 1 | Xác minh nguồn, tải/kiểm kê dữ liệu, smoke test mô hình ngoài | Báo cáo audit, dữ liệu thử, log checkpoint và phần cứng |
| 2 | Split, crop, manifest, mô phỏng suy giảm | Manifest khóa, checks, gallery ba mức |
| 3 | CNN baseline, gamma/HE/CLAHE/MSR | Checkpoint pilot, metric validation cổ điển |
| 4 | Zero-DCE, Retinexformer, diffusion | Ba adapter chạy được, ảnh đầu ra và thời gian pilot |
| 5 | Chọn tham số validation, ước lượng ngân sách, khóa giao thức | Config cuối, checkpoint CNN, test/subset manifest cố định |
| 6 | Chạy test có kiểm soát, lưu cache và dự đoán | Bảng đầy đủ, log lỗi và độ phủ |
| 7 | Tương quan, bootstrap, ablation và phân tích lỗi | Biểu đồ, bảng chênh lệch, pipeline đề xuất |
| 8 | Viết báo cáo, tái chạy mẫu, chuẩn bị demo | Báo cáo/slide, hướng dẫn, gói kết quả tái lập |

GTSRB được chuẩn bị từ tuần 1; mọi phương pháp dùng cùng manifest và split đã khóa.

## 3. Giai đoạn 1 — Xác minh dữ liệu và khả năng chạy

### Việc cần làm

- [ ] Đọc nguồn GTSRB trong DATASETS.md; ghi tên archive, ngày tải, điều kiện sử dụng và checksum.
- [ ] Tải GTSRB từ nguồn được xác minh, gồm nhãn official test; không dùng tập test thiếu nhãn để giả lập điểm số.
- [ ] Đọc cấu trúc annotation thực tế, thống kê nhãn và số ảnh/box theo lớp.
- [ ] Giữ tên file và track gốc; lập source_id và group_id theo class + track.
- [ ] Kiểm tra ít nhất 50 ảnh và ROI phủ đủ 43 lớp, có cả biển nhỏ và ảnh ánh sáng kém.
- [ ] Xác nhận đủ 43 lớp, class ID 0–42 và mapping nhãn official test.
- [ ] Kiểm tra trùng lặp, lỗi giải mã, box ngoài biên, thiếu nhãn.
- [ ] Ghi phần cứng, RAM/VRAM, dung lượng đĩa và thời gian GPU có thể dùng.
- [ ] Tải thử checkpoint từ ba repo chính thức và chạy một ảnh thử trong môi trường riêng khi cần.

### Nghiệm thu

Có báo cáo audit chứa số mẫu thật và ảnh có vẽ box. Mỗi crop truy lại được ảnh nguồn và nhãn. Chưa khôi phục được track của official train thì chưa chia validation ngẫu nhiên từng ảnh. Chưa rõ quyền sử dụng thì ghi trạng thái và hoàn tất xác minh nguồn trước khi công bố bản sao dữ liệu.

Checkpoint xuất hiện trong README chưa được tính là “chạy được”: phải có đầu ra thực, không NaN và đúng số kênh/kích thước sau adapter. Nếu thiếu checkpoint, ghi lỗi sớm và tìm bản tác giả khác có xuất xứ rõ; không đổi danh tính phương pháp âm thầm.

## 4. Giai đoạn 2 — Split và chuẩn hóa

### Việc cần làm

- [ ] Tạo manifest theo schema trong PROJECT_GUIDE.md.
- [ ] GTSRB: giữ official test, tạo validation theo class + track từ official train.
- [ ] Bảo đảm train và validation đủ 43 lớp; điều chỉnh tỷ lệ nhóm nếu cần, không loại lớp hiếm.
- [ ] Kiểm tra group_id không giao nhau giữa các split; kiểm tra ảnh trùng nội dung chéo split.
- [ ] Tạo crop với padding cố định, resize 64×64 và lưu canonical reference.
- [ ] Lưu manifest và checksum, khóa split cho toàn bộ phương pháp.

### Kiểm tra runnable cần để lại

Một file kiểm tra dữ liệu nhỏ, không cần framework: assert đường dẫn tồn tại, ID duy nhất, nhãn thuộc mapping, box hợp lệ, split không giao group/source, crop RGB đúng shape và hữu hạn. Kiểm tra này không thay thế rà nhãn bằng mắt.

### Nghiệm thu

Bảng phân bố lớp/split, gallery crop và kiểm tra dữ liệu đều đạt. Mọi ảnh tối/tăng cường sau này phải kế thừa sample_id và split từ manifest này.

## 5. Giai đoạn 3 — Sinh suy giảm

### Việc cần làm

- [ ] Cài công thức gamma → Poisson → Gaussian → clip đúng hướng gamma.
- [ ] Tạo ba cấu hình pilot trong PROJECT_GUIDE.md và ba seed nhiễu.
- [ ] Tạo gallery cố định sample_id từ train/validation.
- [ ] Đo độ sáng trung bình, PSNR/SSIM với ảnh tham chiếu và tỷ lệ clipping.
- [ ] Điều chỉnh cấu hình nếu ảnh nặng gần như mất toàn bộ thông tin trên phần lớn tập; ghi lý do trước khi khóa.
- [ ] Cache hoặc lưu seed ổn định để mọi enhancer nhận đúng cùng ảnh tối.

### Kiểm tra runnable cần để lại

- Tắt cả suy giảm/nhiễu trả đúng đầu vào trong sai số số học.
- Pixel 0.5 qua gamma=2 thành 0.25 trước nhiễu; gamma enhancement=0.5 làm sáng pixel đó.
- Cùng seed và ID cho cùng đầu ra; seed khác cho nhiễu khác trên mẫu không suy biến.
- Không thay shape, không NaN, output trong [0,1].
- Trên mảng thử đủ lớn, Poisson không clipping có trung bình gần tín hiệu và phương sai gần z/P theo dung sai thống kê hợp lý.

### Nghiệm thu

Có cặp tham chiếu–tối truy vết được, biểu đồ mức suy giảm và cấu hình cuối. Không lựa chọn tham số bằng kết quả test.

## 6. Giai đoạn 4 — CNN baseline và phương pháp cổ điển

### Việc cần làm

- [ ] Cài CNN ba lớp conv như tài liệu; đầu ra cố định 43 lớp GTSRB.
- [ ] Kiểm tra forward shape [B,43], loss hữu hạn và optimizer cập nhật được trọng số.
- [ ] Thử học trên một tập train nhỏ để phát hiện lỗi pipeline; đây là kiểm tra kỹ thuật, không phải kết quả tổng quát hóa.
- [ ] Train pilot trên ảnh gốc, lưu loss, Accuracy và Macro-F1 train/validation.
- [ ] Xem confusion matrix, kiểm tra mapping nếu mô hình gần đoán ngẫu nhiên.
- [ ] Cài gamma, HE, CLAHE và MSR; không thêm denoiser ngoài yêu cầu vào một phương pháp mà không đặt tên riêng.
- [ ] Kiểm tra ảnh đen, ảnh hằng, ảnh kích thước nhỏ và crop nhiều màu.
- [ ] Chạy validation: clean, identity và bốn enhancer cổ điển.

### Nghiệm thu

Các phương pháp chạy chung một hợp đồng RGB [0,1] → RGB [0,1]. Không chỉnh metric hoặc lọc mẫu để nâng điểm baseline. Không áp chỉ tiêu Accuracy cứng trước khi hiểu mức khó và nhãn của dữ liệu.

## 7. Giai đoạn 5 — Ba phương pháp học sâu

### Việc cần làm

- [ ] Tích hợp Zero-DCE từ nguồn tác giả, ghi checkpoint và điều kiện sử dụng.
- [ ] Tích hợp Retinexformer với đúng config/checkpoint đi cùng nhau.
- [ ] Tích hợp Wavelet-based Diffusion LLIE, ghi sampler, số bước và seed.
- [ ] Kiểm tra RGB/BGR, dtype, range, pad/unpad, resize và thứ tự file.
- [ ] Tắt chế độ dùng mean/độ sáng của ground truth trong inference/đánh giá của repo ngoài.
- [ ] Chạy 32–100 crop validation; đo thời gian warm-up, thời gian/ảnh và VRAM thực.
- [ ] Kiểm tra chữ số/ký hiệu nhỏ sau tăng cường, không chỉ độ sáng tổng thể.
- [ ] Lưu output dùng chung cho mọi CNN seed; không cần chạy enhancer lại vì CNN khác seed.

### Nghiệm thu

Đủ ba phương pháp có kết quả thật trên cùng pilot, mapping input/output chính xác, không truy cập ảnh tham chiếu trong bước tăng cường. Ghi mọi adapter thay đổi từ repo gốc. Với diffusion, khóa seed suy luận chính; thử seed thứ hai trên validation để đánh giá độ nhạy nếu có ngẫu nhiên.

## 8. Giai đoạn 6 — Khóa thí nghiệm trước test

### Việc cần làm

- [ ] Chọn tham số enhancer từ lưới nhỏ bằng Macro-F1 validation trung bình theo mức.
- [ ] Train ba CNN seed trên GTSRB và lưu checkpoint tốt nhất theo validation gốc.
- [ ] Khóa split, cấu hình suy giảm, mô hình, normalization và tham số metric.
- [ ] Chọn pipeline dự kiến theo quy tắc validation; lưu quyết định trước khi chạy test.
- [ ] Ước lượng ngân sách từ pilot: thời gian/ảnh × số ảnh × số điều kiện; tính cả cache và ghi đĩa.
- [ ] Nếu diffusion quá nặng: tạo subset phân tầng theo lớp/nhóm bằng seed, trước khi xem điểm test; chạy mọi phương pháp trên subset này.
- [ ] Chuẩn bị file liệt kê từng run dự kiến và trạng thái, tránh bỏ sót điều kiện.

### Nghiệm thu

Một snapshot cấu hình duy nhất mô tả được ma trận test. Có đủ clean + identity + bảy enhancer. Mọi giới hạn về subset, seed hoặc lớp phải ghi trước và hiển thị trong bảng cuối.

## 9. Giai đoạn 7 — Đánh giá cuối

### Việc cần làm

- [ ] Chạy đủ phương pháp trên GTSRB, ba mức và các seed đã khóa.
- [ ] Tính PSNR/SSIM từng ảnh so đúng canonical reference.
- [ ] Tính Accuracy/Macro-F1 và lưu class prediction, xác suất lớp thật, support từng lớp.
- [ ] Báo clean baseline riêng; không nhân bản clean theo noise seed.
- [ ] Đo thời gian enhancer riêng CNN, có thiết bị/batch size/precision.
- [ ] Kiểm tra mỗi bảng so sánh có cùng tập sample_id, không chỉ cùng số lượng.
- [ ] Với run lỗi, sửa lỗi kỹ thuật và chạy lại cùng cấu hình; nếu thay phương pháp/tham số, ghi phiên bản và mức độ ảnh hưởng tới tính độc lập của test.

### Kiểm tra runnable cần để lại

Dùng một mảng nhãn nhỏ tính tay được Accuracy/F1; identity image có SSIM=1 và PSNR vô hạn; so ảnh sai ID phải bị phát hiện. Bảng tổng hợp phải có đủ condition/method/sample_id theo manifest, không âm thầm bỏ ảnh inference lỗi.

### Nghiệm thu

Bảng kết quả không có số giả hoặc ô trống không giải thích. `N/A` chỉ cho metric không có ý nghĩa, ví dụ PSNR trên ảnh đêm thực không paired; “failed/not run” dùng cho phần chưa chạy. Hai trạng thái này không được lẫn nhau.

## 10. Giai đoạn 8 — Phân tích và lựa chọn quy trình

### Việc cần làm

- [ ] Vẽ đường Accuracy/Macro-F1 theo mức suy giảm và phương pháp.
- [ ] Vẽ scatter PSNR/SSIM–Accuracy/Macro-F1 trên GTSRB theo từng mức.
- [ ] Tính Pearson/Spearman, ghi số điểm và giới hạn phụ thuộc giữa các điểm.
- [ ] Tính delta so identity cùng điều kiện; phân tích độ nhạy seed.
- [ ] Paired bootstrap theo nhóm cho delta của pipeline đã chọn với identity.
- [ ] Chạy ablation bốn tổ hợp suy giảm ở mức vừa nếu ngân sách cho phép; báo riêng bảng chính.
- [ ] Phân tích F1 lớp hiếm và các nhầm lẫn giữa biển giống nhau.
- [ ] Chọn ví dụ “sáng hơn nhưng sai”, “tăng cường giúp đúng”, “không phục hồi được”; lưu ID.
- [ ] Đối chiếu pipeline đã chọn từ validation với kết quả test, kể cả khi không thắng.
- [ ] Viết giới hạn giữa thiếu sáng mô phỏng và ảnh đêm thực tế.

### Nghiệm thu

Trả lời đủ RQ1–RQ5 bằng số liệu và hình. Không tuyên bố “tối ưu tuyệt đối”; chỉ kết luận tốt nhất trong các cấu hình, bộ dữ liệu và ngân sách đã khảo sát.

## 11. Giai đoạn 9 — Báo cáo và tái lập

- [ ] Viết theo bố cục ở PROJECT_GUIDE.md, trích nguồn dataset và từng mô hình.
- [ ] Bảng chính chỉ chứa phương pháp chạy trên cùng tập ảnh; bảng toàn test/subset tách rõ.
- [ ] Chú thích đơn vị, số mẫu, seed, metric và chiều tốt/xấu trên mọi bảng/hình.
- [ ] Lưu configs, manifests, dependency versions, checkpoint hashes, metrics, predictions và figures.
- [ ] Viết README với lệnh thực đã chạy; không trình bày lệnh dự kiến như đã tồn tại.
- [ ] Tái chạy từ raw trên một subset nhỏ và đối chiếu output/metric theo dung sai đã ghi.
- [ ] Chuẩn bị demo: ảnh nguồn → thiếu sáng → các enhancer → dự đoán cùng CNN.
- [ ] Soát mô tả đề bài: GTSRB, ba mức, bảy phương pháp, PSNR/SSIM, Accuracy/F1, tương quan, quy trình.

## 12. Quản lý rủi ro cụ thể

| Rủi ro | Dấu hiệu cần kiểm tra | Hành động |
|---|---|---|
| Ảnh cùng track GTSRB lọt vào train và validation | Tên file cùng track, ảnh gần giống | Chia nhóm theo class + track trước khi sinh biến thể |
| Nhãn test hoặc mapping sai | Thiếu CSV, class ID lệch | Đối chiếu annotation gốc và giữ class ID 0–42 |
| Lớp ít mẫu hoặc nhãn đọc sai | Support thấp, crop không khớp tên | Audit loader, chia validation theo track có đủ lớp; không lọc theo điểm test |
| Không tải/chạy được pretrained | Link lỗi, config không khớp, dependency cũ | Smoke test sớm, môi trường riêng, ghi nguồn checkpoint thay thế |
| Diffusion quá chậm | Pilot vượt ngân sách | Subset cố định cho tất cả phương pháp; không bỏ diffusion |
| PSNR cao nhưng CNN kém | Delta chất lượng dương, delta F1 âm | Phân tích chi tiết hình/màu, giữ nguyên kết quả |
| Chỉ một seed vì thiếu tài nguyên | Không đủ budget đã dự kiến | Ghi giới hạn, ưu tiên đủ phương pháp/mức/tập chung trước mở rộng |
| Không đủ thời gian chạy ma trận | Ma trận chưa chạy đủ | Giảm ablation/thí nghiệm B trước; giữ đủ phương pháp trên tập mẫu chung |

## 13. Nhật ký quyết định

Cập nhật khi có thay đổi đáng kể; chưa dùng bảng này để điền kết quả chưa đo.

| Ngày | Quyết định | Cơ sở | Tác động |
|---|---|---|---|
| 14/09/2026 | Chỉ sử dụng GTSRB | Phạm vi cập nhật theo yêu cầu người thực hiện và mô tả đề bài | 43 lớp, 3 lần huấn luyện CNN chính, 219 lượt điều kiện đánh giá |
| 14/09/2026 | Ưu tiên pretrained cho enhancer học sâu | Phạm vi bài tập lớn, cần tập trung so sánh xử lý ảnh | Không train diffusion/transformer từ đầu |
| 14/09/2026 | Thí nghiệm A khóa CNN là chính | Cần đo tác động riêng của tăng cường | Train thích nghi là phần phụ |
| Chưa chốt | Archive GTSRB và split validation theo track | Chờ tải/kiểm kê | Giữ official test, khóa manifest trước khi sinh suy giảm |
| Chưa chốt | Phần cứng, deadline, ngân sách | Chờ thông tin thực tế và pilot | Điều chỉnh lịch/subset minh bạch |

| 14/09/2026 | Sửa nguồn train sang GTSRB_Final_Training_Images.zip | Metadata ERDA và README archive xác nhận Training_fixed là bản cuộc thi cũ | Giữ official final train; pilot cũ lưu riêng, không trộn checkpoint/split |
| 14/09/2026 | Cách ly train track lớp 14/00023 | 8 ảnh trong track trùng pixel official test | Giữ test nguyên trạng; 30 ảnh train có split quarantine, không dùng để học/chọn checkpoint |
