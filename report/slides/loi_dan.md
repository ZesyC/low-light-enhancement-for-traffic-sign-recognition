# Lời dẫn thuyết trình

24 slide, khoảng 18–20 phút. Có thể rút ngắn phần bảng full-test nếu thời gian hạn chế.

## Slide 1: Giới thiệu đề tài

Thời lượng khoảng 30 giây. Giới thiệu tên đề tài và thành viên. Nhấn mạnh đối tượng khảo sát là các phép xử lý ảnh, còn ResNet-18 đóng vai trò bộ nhận dạng cố định để kiểm chứng. Logo và thông tin nhóm lấy từ paper mẫu theo yêu cầu người dùng.

## Slide 2: Mục tiêu và phạm vi

Khoảng 45 giây. Ảnh tăng sáng có thể dễ nhìn hơn nhưng cũng có thể khuếch đại nhiễu hoặc làm sai nét số. Đề bài cần đánh giá cả chất lượng ảnh và tác vụ nhận dạng. Dùng crop đã có ROI, không tuyên bố hệ thống phát hiện biển báo toàn cảnh. Nguồn: đề bài người dùng, README.md và báo cáo mục 1.

## Slide 3: GTSRB và chuẩn hóa đầu vào

Khoảng 45 giây. Giải thích track là chuỗi ảnh liên tiếp của cùng một biển. Chia ở cấp track tránh các ảnh gần giống nhau nằm ở cả train và validation. 30 ảnh cách ly không thuộc tập train cuối. Phần lớn phân tích chi tiết dùng 430 ảnh cân bằng lớp, full-test sẽ được đối chiếu riêng. Nguồn: notebooks/GTSRB_LLIE_Complete_Pipeline_Executed.ipynb, cell 5; configs/resnet18.json; src/data.py.

## Slide 4: Mô phỏng thiếu sáng và nhiễu

Khoảng 60 giây. Đọc thứ tự: Gamma rồi Poisson, Gaussian, cuối cùng clip. Chỉ vào histogram dịch về phía tối và vùng chữ số bị nhiễu. Giải thích p càng nhỏ thì nhiễu Poisson càng mạnh tương đối. Đây là mô phỏng trên RGB, không phải mô hình vật lý đầy đủ của cảm biến. Nguồn: src/degradation.py, configs/resnet18.json; hình lấy nguyên từ notebook cell 7.

## Slide 5: Bảy phương pháp tăng cường ảnh

Khoảng 45 giây. Phân biệt bốn phương pháp truyền thống và ba phương pháp học sâu. Tổng cộng tám nhánh đánh giá khi tính Identity. Không coi Clean là một thuật toán tăng cường. Nguồn: src/enhancement.py. Zero-DCE: https://arxiv.org/abs/2001.06826; Retinexformer: https://arxiv.org/abs/2303.06705; diffusion: https://arxiv.org/abs/2306.00306.

## Slide 6: Gamma và lựa chọn mức tăng sáng

Khoảng 50 giây. Chỉ vào ba ảnh trên để so mức nâng sáng. Ở vùng tối đạo hàm alpha nhân Y mũ alpha trừ 1 có thể lớn, vì thế nhiễu được khuếch đại. Hàng dưới dùng để minh họa CLAHE; giữ lưới 8x8 nên đây không phải cấu hình được chọn cuối. Nguồn: notebook cell 9 và cell 13; src/enhancement.py.

## Slide 7: HE: ánh xạ từ phân phối tích lũy

Khoảng 50 giây. Giải thích CDF là phần pixel có mức sáng không vượt quá k. Mức sáng mới phụ thuộc vị trí tích lũy trong histogram. HE xử lý cả vùng biển và nền bằng một ánh xạ nên không thích ứng với chiếu sáng từng vùng. Chỉ xử lý kênh Y giúp hạn chế biến đổi màu, nhưng lượng tử hóa và clipping vẫn có thể làm thay đổi màu.

## Slide 8: CLAHE: giới hạn tương phản theo từng vùng

Khoảng 55 giây. Giải thích nội suy để tránh các khối vuông có ranh giới rõ. Clip limit hạn chế khuếch đại tương phản nhưng không thay thế bộ khử nhiễu. tileGridSize là số ô theo hai chiều, không phải kích thước pixel. Cấu hình được lựa chọn theo nhận dạng trên validation. Hình ở slide tiếp theo dùng cấu hình mặc định để minh họa cơ chế.

## Slide 9: HE và CLAHE trên kênh độ sáng

Khoảng 60 giây. Histogram rộng hơn chưa chứng minh ảnh tốt hơn. HE có thể kéo giãn vùng sáng mạnh. CLAHE thích ứng theo vùng nhưng ô quá nhỏ trên ảnh 64x64 làm thống kê nhạy hơn với nhiễu. tileGridSize là số ô, không phải kích thước pixel của ô. Cấu hình chọn 4x4 tương ứng ô khoảng 16x16 pixel. Nguồn: src/enhancement.py; https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html.

## Slide 10: Retinex đa tỉ lệ và tính nhạy nhiễu

Khoảng 50 giây. Giải thích ảnh làm mờ là ước lượng chiếu sáng chậm biến thiên. Trừ trong miền log giúp làm rõ phản xạ nhưng không phân biệt biên thật với nhiễu. Tránh kết luận MSR luôn kém; thí nghiệm chỉ Gamma cho kết quả rất tốt và sẽ được trình bày ở phần ablation. Nguồn: src/enhancement.py và notebook cell 13.

## Slide 11: MSR: từ phản xạ ước lượng đến ảnh đầu ra

Khoảng 50 giây. Đây là MSR giản lược, không có phục hồi màu MSRCR. Nhiều tỉ lệ làm mờ không đồng nghĩa với có khử nhiễu riêng. Cần phân biệt khả năng xử lý ảnh chỉ tối với ảnh tối kèm nhiễu; thí nghiệm ablation phía sau cho thấy khác biệt rất lớn.

## Slide 12: Zero-DCE: học đường cong tăng sáng

Khoảng 50 giây. Công thức là quy ước dấu trong mô tả phương pháp gốc; tham số có thể làm tăng hoặc giảm cường độ. Mạng dự đoán cách điều chỉnh ảnh chứ không dùng một gamma cố định. Không cần cặp ảnh khi huấn luyện gốc không có nghĩa dự án đã huấn luyện mạng. Trong thí nghiệm chỉ nạp checkpoint.

## Slide 13: Retinexformer: phục hồi có hướng dẫn chiếu sáng

Khoảng 50 giây. Sơ đồ diễn tả ý tưởng, không phải toàn bộ các lớp mạng. Retinexformer không phải phép MSR rồi thêm Transformer; nó học cách dùng thông tin chiếu sáng để phục hồi. Trọng số học trên miền khác GTSRB nên cần đo thực nghiệm, không mặc định tốt hơn CLAHE.

## Slide 14: Diffusion trong miền wavelet

Khoảng 55 giây. Phân biệt diffusion phục hồi ảnh có điều kiện với sinh ảnh tự do. Wavelet giúp biểu diễn riêng phần biến thiên chậm và chi tiết; không có nghĩa tự động giữ đúng mọi nét số. EMA là trọng số trung bình trượt của mô hình. Sau phần minh họa sẽ so chất lượng ảnh, F1 và thời gian xử lý để đánh giá đánh đổi.

## Slide 15: Tăng cường học sâu trên cùng một ảnh

Khoảng 50 giây. Hình này dùng tham số đã chọn, khác hình tham số mặc định ở phần HE và CLAHE. Nhìn vào nền trắng, viền đỏ và chữ số. Ảnh diffusion không sáng nhất nhưng giữ kết quả nhận dạng tốt trên tập khảo sát. Một mẫu chỉ minh họa cơ chế, không đại diện xếp hạng toàn tập. Nguồn và bài báo: notebook cell 14; https://arxiv.org/abs/2001.06826; https://arxiv.org/abs/2303.06705; https://arxiv.org/abs/2306.00306.

## Slide 16: Giao thức đánh giá với ResNet-18 cố định

Khoảng 60 giây. 219 lượt là ma trận đánh giá, không phải 219 lần huấn luyện. Enhancer và metric ảnh làm việc ở 64x64. ResNet được huấn luyện ba lần với augmentation nhẹ trên ảnh gốc, sau đó khóa. Macro-F1 tính trọng số bằng nhau cho 43 lớp. Chín tổ hợp seed không phải chín lần train độc lập. Nguồn: configs/resnet18.json, src/model.py và notebook cells 11, 16, 17.

## Slide 17: Macro-F1 giảm khi thiếu sáng nặng hơn

Khoảng 60 giây. Không đọc mọi ô. Chỉ ba dòng Identity, CLAHE, diffusion để mô tả xu hướng. Diffusion được chọn từ validation, không chọn sau khi xem bảng này. Mức nhẹ CLAHE và diffusion chỉ cách 0,03 điểm phần trăm, chưa phải bằng chứng ưu thế chắc chắn. Ở mức nặng ngay cả diffusion còn xa clean. Nguồn: data/resnet18_notebook_aggregate.csv, tính lại từ 219 bản ghi notebook.

## Slide 18: PSNR cao hơn chưa bảo đảm nhận dạng tốt hơn

Khoảng 60 giây. So Zero-DCE với diffusion: PSNR 20,08 lớn hơn 14,90 nhưng F1 69,30 thấp hơn 82,71. CLAHE có SSIM cao hơn diffusion nhưng F1 thấp hơn. Cơ chế giữ hoặc làm mất chi tiết là cách giải thích hợp lý, cần xem từng ảnh để xác nhận nguyên nhân. Nguồn: notebook và bảng CSV kèm theo.

## Slide 19: Tương quan giữa chất lượng ảnh và Macro-F1

Khoảng 55 giây. Spearman so thứ hạng giữa phương pháp theo chất lượng và theo F1. Pearson đo liên hệ tuyến tính, với PSNR/F1 là 0,715; 0,660; 0,686 và SSIM/F1 là 0,741; 0,755; 0,691. Không nói tương quan là quan hệ nhân quả. Tám phương pháp cùng dữ liệu nên không độc lập; MSR thấp ở cả hai trục có thể ảnh hưởng Pearson. Nguồn: data/resnet18_correlations.csv.

## Slide 20: Nhiễu làm thay đổi thứ hạng phương pháp

Khoảng 60 giây. MSR không luôn kém: tốt nhất khi chỉ có Gamma trong bảng đầy đủ. Thêm nhiễu làm MSR suy giảm mạnh, phù hợp cơ chế log nhạy ở vùng tối. Trong cấu hình này Poisson ảnh hưởng mạnh hơn Gaussian, nhưng hai mức nhiễu không cân bằng năng lượng nên không suy rộng cho mọi trường hợp. Ba đường là các phương pháp đại diện, không phải toàn bộ tám phương pháp.

## Slide 21: Đối chiếu trên 12.630 ảnh test

Khoảng 45 giây. Nói rõ đây là số liệu dự án công bố trong README, không phải lấy chất lượng ảnh subset ghép với F1 full-test. Chỉ trình bày ba phương án để đối chiếu xu hướng. Bảng đủ tám phương pháp có trong báo cáo. Chưa kiểm tra lại dự đoán từng ảnh full-test ở máy này. Nguồn: data/resnet18_fulltest_readme.csv, README.md.

## Slide 22: Hiệu năng nhận dạng và chi phí xử lý

Khoảng 45 giây. Độ trễ chỉ bao gồm khối tăng cường theo cách đo dự án, không bao gồm detector, I/O và toàn bộ classifier. Không chuyển ms/ảnh thành khẳng định FPS thực tế của hệ thống. Nếu mục tiêu ràng buộc thời gian thay đổi, cần lựa chọn lại trên validation theo mục tiêu đó.

## Slide 23: Quy trình tiền xử lý đề xuất

Khoảng 50 giây. Trong thí nghiệm tạo ảnh tối trước enhancer; trong triển khai ảnh đã tối thì không làm tối thêm. Diffusion là lựa chọn đã xác định bằng validation. Không tự chuyển giữa CLAHE và diffusion theo nhãn mild/medium/severe vì nhãn này không biết sẵn ở triển khai. Một bộ chọn thích nghi cần được nghiên cứu riêng. Nguồn: src/experiment.py, cấu hình và báo cáo mục 6.

## Slide 24: Kết luận và hướng phát triển

Khoảng 40 giây. Kết lại theo đúng câu hỏi mở đầu. Không tuyên bố tăng sáng giải quyết hoàn toàn điều kiện tối nặng. Khi hỏi về mở rộng, đề xuất kiểm tra dữ liệu đêm thực, giữ rõ nhãn và cách đánh giá, thêm khử nhiễu nhưng phải kiểm tra mất nét số. Các kết quả chi tiết, tài liệu thuật toán và giới hạn có trong báo cáo.
