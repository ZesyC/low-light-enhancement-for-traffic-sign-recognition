# Paper baseline cho nghiên cứu tăng cường thiếu sáng và phân loại biển báo

Ngày đối chiếu nguồn và chốt lựa chọn baseline: 26/09/2026. Người thực hiện đã thống nhất chọn Zero-DCE làm baseline học sâu chính. Đây là quyết định thiết kế thí nghiệm, chưa phải kết quả tái lập; lượt nghiên cứu này chỉ rà tài liệu và nguồn công khai, chưa tải/chạy checkpoint hay thay đổi giao thức.

## Lựa chọn baseline đã chốt

Chọn **Zero-DCE (CVPR 2020)** làm baseline phương pháp học sâu để chạy lại trong giao thức GTSRB hiện tại. Dùng **Kagde và cộng sự (CVIP 2023, xuất bản 2024)** làm nghiên cứu liên quan gần chủ đề và dữ liệu nhất trong các ứng viên đã kiểm tra. Bổ sung **Tomas và cộng sự (IJACSA 2026)** vì bài này đánh giá đúng tác vụ phân loại biển đã định vị và dùng Zero-DCE.

Lý do chọn Zero-DCE: có code và checkpoint chính thức, đã được tích hợp trong repo, và có thể đánh giá trên cùng GTSRB với cùng CNN cố định. Lựa chọn này dựa trên khả năng tái lập và mức phù hợp với dự án; hiệu quả nhận diện cần được đo bằng thí nghiệm.

| Thành phần | Vai trò trong bài viết |
|---|---|
| Clean | Mốc nhận diện trên ảnh gốc |
| Identity | Mốc ảnh tối không xử lý để đo lợi ích của tăng cường |
| Zero-DCE | Baseline học sâu chính, chạy lại theo giao thức GTSRB của dự án |
| Retinexformer và diffusion | Các phương pháp đối chiếu bổ sung; vẫn chạy đủ bảy enhancer đã cam kết |
| Kagde — modified MIRNet + YOLOv4 | Related Work; khác mô hình và metric nên chưa so trực tiếp với CNN hiện tại |

Khi viết paper, ghi: **“Zero-DCE được đánh giá lại theo giao thức GTSRB của chúng tôi.”** Mọi tuyên bố tốt hơn baseline phải dựa trên kết quả chạy chung tập ảnh, mức suy giảm, seed và checkpoint CNN; lựa chọn này không tự động khóa snapshot thí nghiệm.

Không có ứng viên nào ở đây cung cấp một con số Accuracy/Macro-F1 có thể chép vào bảng của dự án để tuyên bố thắng: khác mô hình, tập đánh giá hoặc quy trình tạo thiếu sáng. So sánh định lượng phải là kết quả chúng ta chạy lại cùng giao thức.

## Ba ứng viên đã đối chiếu

### 1. Zero-DCE: baseline thực nghiệm nên chạy

Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, Runmin Cong. **Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement**, CVPR 2020, tr. 1780–1789. [Paper chính thức](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf), [mã tác giả](https://github.com/Li-Chongyi/Zero-DCE).

Repo tác giả có PyTorch, script inference và pretrained `Epoch99.pth`; giấy phép dành cho nghiên cứu/phi thương mại. Đây là phương pháp tăng cường ảnh chung, không phải paper phân loại GTSRB. Ưu điểm thực tế là có thể giữ CNN hiện tại và thay tiền xử lý bằng checkpoint tác giả. Chưa xác minh checkpoint chạy thành công trong lượt nghiên cứu này. [Nguồn code và checkpoint](https://github.com/Li-Chongyi/Zero-DCE).

**Cách ghi hàng trong paper:** “Zero-DCE (Guo et al., 2020), pretrained, evaluated under our GTSRB protocol”. Không gọi đó là tái lập toàn bộ bài Zero-DCE. Trong dự án, phương pháp này đã nằm trong kế hoạch bảy enhancer của `PROJECT_GUIDE.md`.

### 2. Paper gần chủ đề/GTSRB: làm mốc Related Work

Manas Kagde, Priyanka Choudhary, Rishi Joshi, Somnath Dey. **Automatic Signboard Recognition in Low Quality Night Images**. Hội nghị CVIP 2023; bản Springer xuất bản 03/07/2024, CCIS 2010, tr. 478–490. DOI: [10.1007/978-3-031-58174-8_40](https://doi.org/10.1007/978-3-031-58174-8_40). [Bản tác giả miễn phí](https://arxiv.org/abs/2308.08941).

Bài dùng modified MIRNet → YOLOv4, GTSRB 43 lớp và GTSDB. Trên 1.600 ảnh chất lượng thấp được chọn thủ công, mAP@0.5 tăng 84,78 → 90,18, tức **5,40 điểm phần trăm**. Toàn GTSRB tăng 96,65 → 96,75. Modified MIRNet học từ LOL; sửa spatial attention bằng median pooling. [Toàn văn tác giả, mục 3.2 và 4](https://arxiv.org/html/2308.08941v1).

**Giới hạn đối chiếu:** dự án đo phân loại crop bằng CNN cố định; bài dùng YOLO và mAP, chọn subset thủ công. Vì vậy không so trực tiếp mAP với Accuracy. Chưa tìm thấy code/checkpoint chính thức cho phiên bản modified MIRNet qua các nguồn bài và tìm kiếm tên bài/tác giả trong lượt này; điều đó không chứng minh code không tồn tại. MIRNet nguyên bản không tương đương modified MIRNet của Kagde.

### 3. Paper đúng tác vụ phân loại nhưng khác dữ liệu

John Paul Q. Tomas, Carlo Miguel P. Legaspi, Karl Anthony S. Dalangin, Gabriel Paul Q. Lim. **Traffic Sign Classification Under Varying Lighting Conditions in the Philippines Using Transfer Learning with ResNet50 and Zero-DCE**. IJACSA 17(3), 2026. DOI: [10.14569/IJACSA.2026.0170327](https://doi.org/10.14569/IJACSA.2026.0170327).

Bài phân loại ảnh biển đã định vị: khoảng 5.000 ảnh địa phương, 7 lớp, split 70/10/20; GTSRB dùng trong pretraining. ResNet50 nhiều giai đoạn đạt 96,43% và phiên bản Zero-DCE đạt 98,21% trên dữ liệu của họ. [Trang nhà xuất bản](https://thesai.org/Publications/ViewPaper?Code=IJACSA&Issue=3&SerialNo=27&Volume=17). PDF ghi dùng code Zero-DCE chính thức và ảnh 32×32. [PDF, mục phương pháp](https://thesai.org/Downloads/Volume17No3/Paper_27-Traffic_Sign_Classification_Under_Varying_Lighting_Conditions.pdf).

**Giới hạn:** 98,21% không phải kết quả GTSRB 43 lớp. Chưa tìm được repo tác giả chứa đầy đủ split, dữ liệu địa phương và checkpoint toàn pipeline. Dùng làm Related Work về tương tác giữa tăng cường và học chuyển giao; không coi là đối thủ đã tái lập trong bảng của dự án.

## Kế hoạch tiếp nối: so sánh baseline và viết paper

**Trạng thái: đã lập kế hoạch, chưa bắt đầu.** Thực hiện sau khi hoàn thành [kế hoạch dự án hiện tại](plan.md). Phần cải thiện CNN trong [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) phải có quyết định chốt cấu hình và kết quả tổng hợp; các phương án tùy chọn chưa làm cần ghi rõ lý do bỏ qua. Không bắt buộc thử mọi kiến trúc hoặc hyperparameter để mở giai đoạn này.

Kế hoạch hiện tại đã bao gồm chạy bảy enhancer và báo cáo bài tập lớn. Giai đoạn tiếp nối tận dụng checkpoint, dự đoán và báo cáo đó để tạo bảng đối chiếu Zero-DCE và bản thảo paper; chỉ bổ sung phần còn thiếu hoặc không hợp lệ.

### Bước 1 — Nghiệm thu đầu vào từ kế hoạch hiện tại

- [ ] Đối chiếu đủ checkpoint, learning curves và log của ba seed CNN 11/22/33; ghi kiến trúc, augmentation, epoch được chọn và SHA256 từng checkpoint.
- [ ] Xác nhận manifest, split theo track, crop 64×64, cấu hình suy giảm và subset test nếu có đã được khóa; mọi phương pháp dùng cùng sample ID và seed nhiễu.
- [ ] Có kết quả clean, identity và đủ bảy enhancer cho ma trận đã khóa. Giao thức mặc định đủ ba seed huấn luyện và ba seed nhiễu có 219 lượt điều kiện CNN.
- [ ] Xác nhận lựa chọn pipeline từ validation được lưu trước khi xem test; báo cáo và dự đoán từng ảnh truy vết được về snapshot thí nghiệm.

**Đầu ra:** danh sách artifact được sử dụng và thiếu sót còn lại. Nếu chưa đủ đầu vào, hoàn thành phần tương ứng của `plan.md` trước khi sang bước 2; không đánh dấu run thiếu là hoàn tất.

### Bước 2 — Xác minh baseline Zero-DCE

- [ ] Đối chiếu adapter với code tác giả: Zero-DCE nguyên bản, commit, pretrained `Epoch99.pth`, SHA256, RGB [0,1], output cuối và resize/pad.
- [ ] Xác nhận enhancer chỉ nhận ảnh tối; không dùng nhãn hoặc ảnh tham chiếu test để điều chỉnh đầu ra. Với mỗi seed CNN, giữ cùng checkpoint cho mọi enhancer.
- [ ] Tận dụng kết quả Zero-DCE hợp lệ đã có; chỉ chạy bổ sung khi thiếu hoặc có lỗi kỹ thuật, theo giao thức đã lưu. Ghi thay đổi nếu phải sửa code sau khi đã xem test.

**Đầu ra:** hồ sơ tái lập baseline và xác nhận bảng so sánh dùng chung giao thức. Kagde và Tomas tiếp tục là Related Work; tái lập modified MIRNet + YOLOv4 sẽ cần một kế hoạch riêng.

### Bước 3 — Lập bảng đối chiếu và phân tích

- [ ] Lập bảng clean riêng; với từng mức tối, báo identity, Zero-DCE và các enhancer còn lại bằng Accuracy, Macro-F1, PSNR, SSIM và latency. Ghi số mẫu, thiết bị, độ phân giải và phạm vi full test/subset.
- [ ] Báo chênh lệch Accuracy/Macro-F1 theo điểm phần trăm so với identity và so với Zero-DCE; trung bình các noise seed trong từng CNN seed trước, rồi tính mean ± SD giữa ba CNN seed.
- [ ] Tính paired bootstrap CI 95% cho chênh lệch Macro-F1 của pipeline đã chọn trên validation so với Zero-DCE, tái lấy mẫu theo nhóm/source và ghép đúng sample ID. Nếu pipeline được chọn chính là Zero-DCE, báo điều đó và tập trung so với identity.
- [ ] Giữ pipeline đã chọn từ validation. Phép so sánh mới chỉ được đặt ra sau khi xem test phải ghi là phân tích bổ sung sau khi xem dữ liệu; không dùng nó để chọn lại pipeline hay gọi test đã xem là đánh giá độc lập mới.
- [ ] Phân tích từng mức tối, lớp yếu, tương quan PSNR/SSIM với nhận diện, chi phí và ví dụ giúp/hại dự đoán. Giữ cả kết quả không vượt baseline.

**Đầu ra:** bảng CSV và hình cho paper, kèm kết luận có thể truy về dữ liệu. Ưu tiên dùng dự đoán đã lưu; bổ sung phép tổng hợp cần thiết sau khi kiểm tra khả năng của `src/metrics.py` và `src/reporting.py`.

### Bước 4 — Chốt đóng góp và viết bản thảo

- [ ] Cập nhật tài liệu liên quan tại thời điểm viết; đối chiếu Zero-DCE, Kagde và Tomas để xác định phần đóng góp được bằng chứng hỗ trợ.
- [ ] Chọn nơi gửi và template theo phạm vi nghiên cứu thực tế; thống nhất ngôn ngữ, giới hạn trang và định dạng trích dẫn trước khi dàn bản thảo.
- [ ] Viết Introduction → Related Work → Experimental Protocol → Results → Discussion/Limitations → Conclusion; hoàn thiện Abstract sau khi kết quả đã chốt.
- [ ] Mô tả rõ baseline là “Zero-DCE được đánh giá lại theo giao thức GTSRB của chúng tôi”; phân biệt kết quả tự chạy với số liệu được trích từ paper khác.
- [ ] Giới hạn kết luận theo GTSRB thiếu sáng mô phỏng, CNN, pretrained và ngân sách đã dùng. Mọi đề xuất phương pháp mới hoặc đánh giá đêm thực cần được lập thành nghiên cứu bổ sung với giao thức riêng.

**Đầu ra:** bản thảo hoàn chỉnh, tài liệu tham khảo và bộ bảng/hình đồng nhất với kết quả thực nghiệm.

### Bước 5 — Rà soát và chuẩn bị bàn giao

- [ ] Đối chiếu mọi con số trong Abstract, bảng và kết luận với CSV; kiểm tra tên metric, chiều tốt/xấu, đơn vị, số mẫu và cách tính CI.
- [ ] Kiểm tra một lượt tái chạy đại diện và tái tạo bảng từ dự đoán đã lưu; ghi môi trường, commit, hash, lệnh thực chạy và vị trí artifact.
- [ ] Rà trích dẫn, giới hạn so sánh, chất lượng hình và định dạng theo template đã chọn.
- [ ] Bàn giao bản thảo nguồn, PDF, BibTeX, bảng/hình và hướng dẫn tái lập để người thực hiện duyệt trước khi gửi bài.

**Tiêu chí hoàn tất:** bản thảo có số liệu truy vết được, so sánh Zero-DCE trên cùng giao thức, mô tả rõ đóng góp và giới hạn, cùng gói tái lập đủ dùng. Lịch thực hiện được ước lượng sau bước 1 dựa trên phần còn thiếu; không chạy lại toàn bộ thí nghiệm chỉ để hoàn thành checklist này.

## Cách định vị bài viết

Với phạm vi hiện tại, cách diễn đạt có căn cứ là **nghiên cứu thực nghiệm có kiểm soát về ảnh hưởng của tăng cường thiếu sáng đến phân loại biển báo**. Chỉ bổ sung/chạy nhiều enhancer chưa đủ để gọi là phương pháp mới. Cần kết quả thực cho thấy điều gì: phương pháp nào giúp/hại từng mức, PSNR/SSIM có liên quan đến nhận diện không, biến thiên qua seed và chi phí suy luận.

Các khác biệt về split theo track, dữ liệu suy giảm và phân tích bất định là đặc điểm giao thức dự kiến, chưa chứng minh tính mới so với toàn bộ literature hoặc bảo đảm được nhận đăng. Để có tuyên bố rộng hơn camera ban đêm thực tế, cần dữ liệu đêm thực và giao thức bổ sung; GTSRB suy giảm mô phỏng chỉ hỗ trợ kết luận trong phạm vi mô phỏng.

## BibTeX tối thiểu

```bibtex
@inproceedings{guo2020zerodce,
  title={Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
  author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1780--1789},
  year={2020}
}

@inproceedings{kagde2024signboard,
  title={Automatic Signboard Recognition in Low Quality Night Images},
  author={Kagde, Manas and Choudhary, Priyanka and Joshi, Rishi and Dey, Somnath},
  booktitle={Computer Vision and Image Processing: CVIP 2023},
  series={Communications in Computer and Information Science},
  volume={2010},
  pages={478--490},
  year={2024},
  publisher={Springer},
  doi={10.1007/978-3-031-58174-8_40}
}

@article{tomas2026traffic,
  title={Traffic Sign Classification Under Varying Lighting Conditions in the Philippines Using Transfer Learning with ResNet50 and Zero-DCE},
  author={Tomas, John Paul Q. and Legaspi, Carlo Miguel P. and Dalangin, Karl Anthony S. and Lim, Gabriel Paul Q.},
  journal={International Journal of Advanced Computer Science and Applications},
  volume={17},
  number={3},
  year={2026},
  doi={10.14569/IJACSA.2026.0170327}
}
```
