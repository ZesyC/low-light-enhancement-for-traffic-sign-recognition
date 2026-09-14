# Dữ liệu dự án: GTSRB

Ngày cập nhật: 14/09/2026. Phạm vi: chỉ sử dụng **German Traffic Sign Recognition Benchmark (GTSRB)** đúng mô tả bài tập lớn. Đã có mã chuẩn bị và kiểm kê dữ liệu; xem README.md và `data/manifests/audit.json` cho trạng thái chạy thực tế.

Xem [PROJECT_GUIDE.md](PROJECT_GUIDE.md) để nắm giao thức thí nghiệm và [plan.md](plan.md) để triển khai.

## 1. Vì sao dùng GTSRB?

GTSRB là bộ dữ liệu phân loại biển báo giao thông của Đức, gồm 43 lớp và các ảnh biển báo trong điều kiện chụp đa dạng. Đây là dữ liệu phù hợp để khảo sát ảnh hưởng của tăng cường thiếu sáng tới CNN phân loại. Tham khảo [trang benchmark](https://benchmark.ini.rub.de/) và [bài báo của nhóm xây dựng GTSRB](https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf).

Dự án giữ đủ 43 lớp với class ID 0–42. Ảnh gốc đóng vai trò tham chiếu; từ mỗi ảnh sinh các phiên bản thiếu sáng bằng gamma kết hợp Gaussian và Poisson. Không mặc định ảnh gốc hoàn toàn sạch nhiễu hoặc có ánh sáng lý tưởng.

Phạm vi là phân loại một biển báo đã biết vùng ảnh, không phát hiện biển trong ảnh toàn cảnh. Không dùng GTSDB thay GTSRB vì đó là bài toán khác.

## 2. Nguồn tải

Nguồn gốc để trích dẫn: [trang dữ liệu GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html). Trang này trả lỗi 502 trong lần kiểm tra hiện tại.

Nguồn hiện dùng là [archive do nhóm tác giả GTSRB công bố trên ERDA](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). Metadata nguồn phân biệt **official final train** và bản train cuộc thi cũ. Đã tải và kiểm tra archive khi triển khai ngày 14/09/2026.

| Thành phần | Archive |
|---|---|
| Official final train | [GTSRB_Final_Training_Images.zip](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip) |
| Official test | [GTSRB_Final_Test_Images.zip](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip) |
| Nhãn test | [GTSRB_Final_Test_GT.zip](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip) |

**Sửa sau kiểm kê:** `GTSRB-Training_fixed.zip` mà Torchvision dùng chỉ có 26.640 ảnh train cuộc thi cũ. Pipeline dùng `GTSRB_Final_Training_Images.zip` để giữ official final train. Pilot ban đầu trên bản cũ được lưu riêng trong các thư mục có hậu tố `training_fixed`, không trộn vào kết quả final.

Ghi rõ tên archive đã dùng; không trộn các bản đóng gói rồi giả định cấu trúc giống nhau. Kiểm tra điều kiện sử dụng và yêu cầu trích dẫn từ nguồn dữ liệu; không suy giấy phép ảnh từ giấy phép thư viện tải ảnh.

### Các bước tiếp nhận

1. Tải đủ ba thành phần, lưu archive gốc tại `data/raw/gtsrb/`.
2. Giải nén vào thư mục riêng, giữ nguyên tên file, thư mục lớp và annotation.
3. Lưu URL, ngày tải, tên archive và checksum vào `data/manifests/gtsrb_sources.json`.
4. Trên macOS, dùng `shasum -a 256 /duong/dan/toi/archive.zip` để tính SHA-256; thay đường dẫn bằng file thật.
5. Mở annotation và ảnh mẫu trước khi viết loader; kiểm kê số ảnh, số lớp, số track và nhãn test thực tế.

Các thư mục này được tạo bởi `python -m scripts.run download` và `prepare`; dữ liệu không đưa vào Git.

## 3. Kiểm kê ảnh và nhãn

- Xác nhận đủ 43 lớp, class ID 0–42; lưu bảng tên lớp từ nguồn benchmark.
- Ghép nhãn test bằng tên file trong annotation, không suy nhãn từ thứ tự thư mục hoặc thứ tự đọc file.
- Kiểm tra lỗi giải mã, ảnh trùng, nhãn thiếu và ROI ngoài biên.
- Ghi phân bố số ảnh từng lớp, kích thước ảnh, độ sáng và số nhóm track trong official train.
- Kiểm tra trực quan ít nhất 50 ảnh phủ đủ lớp, gồm biển nhỏ và ảnh ánh sáng kém.
- Giữ toàn bộ official test. Nếu có lỗi, kiểm tra lại archive/loader và ghi nhận, không âm thầm loại ảnh khó.

## 4. Chia train, validation và test

**Giữ official test nguyên trạng; tạo validation từ official train theo track.** Các ảnh chụp liên tiếp trong cùng track có thể rất giống nhau, nên chia ngẫu nhiên từng ảnh dễ làm validation quá lạc quan. Cơ sở về track: [bài báo GTSRB](https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf).

Quy tắc dự án:

1. Với official train, dùng cặp `(class_id, track_id)` làm `group_id`, xác định track từ tên file/metadata gốc đã kiểm tra.
2. Dành khoảng 15–20% official train làm validation theo nhóm; bảo đảm train và validation đều có đủ 43 lớp.
3. Không ép đúng tỷ lệ bằng cách chia một track sang hai split. Khóa seed chia nhóm và lưu manifest.
4. Không suy track của official test từ tiền tố tên file nếu bản phân phối không cung cấp thông tin đó. Khi thiếu track test, dùng source_id cho truy vết và bootstrap, ghi rõ giới hạn.
5. Sinh ảnh tối, nhiễu và augmentation sau khi chia tập. Mọi biến thể của một ảnh phải kế thừa split của ảnh gốc.
6. Validation dùng chọn checkpoint CNN và tham số tăng cường; test chỉ dùng đánh giá cuối theo giao thức đã khóa.

## 5. ROI và ảnh tham chiếu

Dùng ROI nhãn thật để tạo crop theo PROJECT_GUIDE.md: cộng 5% mỗi phía và giới hạn trong ảnh. Xác minh quy ước tọa độ/biên của annotation bằng cách vẽ box trước khi crop; không mặc định tọa độ pixel là chuẩn hóa hoặc ngược lại.

Resize crop về RGB 64×64 bằng cùng phép nội suy, lưu ảnh tham chiếu canonical; sinh thiếu sáng sau resize. Giữ cả biển nhỏ, không bỏ ảnh chỉ vì khó nhận diện.

Loader mặc định của Torchvision trả ảnh RGB và nhãn, không tự crop theo ROI trong annotation. Nếu dùng loader đó, vẫn cần bước đọc ROI riêng để khớp giao thức crop của dự án. Xem [mã nguồn loader](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/gtsrb.html).

Không tăng cường hoặc denoise ảnh tham chiếu rồi gọi đó là ảnh gốc. Mọi phương pháp nhận cùng một phiên bản tối và được so với đúng ảnh tham chiếu tương ứng.

## 6. Manifest cần lưu

Tạo `data/manifests/gtsrb_samples.csv` với các trường cốt lõi thống nhất cùng PROJECT_GUIDE.md:

| Trường | Nội dung |
|---|---|
| `sample_id` | ID duy nhất của mẫu |
| `dataset` | Giá trị cố định `gtsrb` |
| `source_path`, `source_id` | Đường dẫn và ID ảnh nguồn |
| `group_id` | Class + track ở train/validation; nhóm được xác minh hoặc source_id ở test |
| `bbox` | ROI dùng crop, ghi rõ quy ước tọa độ và padding |
| `original_label`, `class_id` | Nhãn gốc và class ID giữ nguyên 0–42 |
| `split` | train, validation hoặc test |
| `width`, `height` | Kích thước ảnh nguồn |
| `source_sha256` | Checksum ảnh nguồn |
| `crop_path` | Đường dẫn ảnh tham chiếu 64×64 |

Manifest biến thể bổ sung `sample_id, condition, gamma_dark, poisson_peak, gaussian_sigma, noise_seed, path`. Cấu hình crop/resize lưu cùng phiên bản manifest; hash manifest được ghi vào log thí nghiệm.

Các trường trên do dự án quy định, không phải khẳng định tất cả có sẵn trong archive.

## 7. PSNR/SSIM và giới hạn của dữ liệu

Nhánh đánh giá chính có cặp tham chiếu nhân tạo: ảnh gốc x → ảnh suy giảm y → ảnh tăng cường x_hat. Tính PSNR/SSIM của y và x_hat so với chính x; tính Accuracy/Macro-F1 bằng nhãn GTSRB trên cùng CNN đã khóa trọng số.

Kết quả phản ánh thiếu sáng mô phỏng trên GTSRB. Không suy thành bằng chứng về mọi điều kiện camera ban đêm, chói đèn, mưa hoặc motion blur. Dự án không yêu cầu thu thêm ảnh đêm thực; nếu mở rộng sau này mà thiếu ảnh tham chiếu đồng đăng ký thì không tính PSNR/SSIM cho các ảnh đó.

## 8. Điều kiện dữ liệu sẵn sàng

- [ ] Đủ archive ảnh train, ảnh test và nhãn test, có URL và checksum.
- [ ] Mapping 43 lớp hợp lệ và bảng kiểm kê số ảnh thực tế.
- [ ] Train/validation chia theo track, official test giữ nguyên.
- [ ] ROI/crop kiểm tra trực quan; không thiếu ảnh hoặc nhãn không giải thích.
- [ ] Manifest khóa và kiểm tra không giao nhóm train/validation.
- [ ] Ảnh tham chiếu và biến thể truy vết được theo sample_id.

Chỉ đánh dấu hoàn thành sau khi kiểm tra dữ liệu thực, không dựa vào việc đã có đường dẫn tải.

## Phát hiện khi triển khai trên final archive

Có 8 ảnh train lớp 14, track 00023 trùng pixel với official test. Toàn bộ 30 ảnh trong track này được giữ trong manifest với `split=quarantine`, không dùng train/validation; official test không đổi. Xem `data/manifests/quarantine.json` và `duplicate_audit.json`. Đây là quyết định chống rò rỉ dữ liệu trước huấn luyện/test, không dựa trên điểm mô hình.
