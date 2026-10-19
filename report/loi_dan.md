# Lời dẫn thuyết trình của Nhóm 2

Bản lời dẫn bám theo 24 slide, dự kiến khoảng 20–23 phút tùy tốc độ nói và thời gian chỉ hình. Các đoạn dưới đây có thể dùng để tập nói trực tiếp. Khi trình bày bảng số liệu, chỉ nhấn vào những giá trị được nêu trong lời dẫn, không cần đọc toàn bộ bảng.

## Slide 1: Giới thiệu đề tài

Em xin kính chào thầy và các bạn. Chúng em là Nhóm 2, gồm Kiều Thị Thu Trang, Trần Minh Sang và Ngô Quang Thiện. Hôm nay, nhóm xin trình bày đề tài “Khảo sát ảnh hưởng của xử lý ảnh đến hiệu năng nhận dạng biển báo giao thông trong điều kiện thiếu sáng”, dưới sự hướng dẫn của TS. Nguyễn Ngọc Giang.

Trong đề tài này, nhóm tập trung vào các phương pháp tăng cường ảnh. ResNet-18 được sử dụng làm bộ phân loại để kiểm tra xem thông tin cần cho việc nhận dạng có được giữ lại sau xử lý hay không. Bài trình bày sẽ đi từ dữ liệu và cách tạo ảnh thiếu sáng, đến nguyên lý các phương pháp, kết quả so sánh và quy trình tiền xử lý được lựa chọn.

## Slide 2: Mục tiêu và phạm vi

Câu hỏi chính của nhóm là: khi một bức ảnh trở nên sáng hơn, liệu mô hình có nhận dạng biển báo chính xác hơn hay không? Hai điều này có thể liên quan nhưng không phải lúc nào cũng đi cùng nhau. Tăng sáng có thể giúp nhìn rõ biển báo, nhưng cũng có thể làm nhiễu nổi lên hoặc làm mất những nét nhỏ của chữ số.

Vì vậy, nhóm đánh giá đồng thời chất lượng ảnh bằng PSNR, SSIM và hiệu năng nhận dạng bằng Accuracy, Macro-F1. Nhóm mô phỏng ba mức thiếu sáng và so sánh bảy phương pháp tăng cường trên cùng điều kiện.

Phạm vi đầu vào là ảnh đã cắt vùng biển báo, với 43 lớp GTSRB. Nhóm chưa giải quyết việc tìm vị trí biển báo trong toàn cảnh. Trước hết, em xin giới thiệu cách chuẩn bị dữ liệu để phép so sánh được nhất quán.

## Slide 3: GTSRB và chuẩn hóa đầu vào

GTSRB gồm 43 lớp biển báo giao thông. Trong thí nghiệm, nhóm sử dụng 31.350 ảnh huấn luyện, 7.829 ảnh validation và giữ nguyên 12.630 ảnh kiểm thử chính thức. Ngoài ra, có 30 ảnh được cách ly vì thuộc một track chứa ảnh trùng với tập test.

Track có thể hiểu là chuỗi ảnh liên tiếp của cùng một biển báo. Nhóm chia train và validation theo track để hạn chế việc những ảnh rất giống nhau xuất hiện ở cả hai tập, khiến kết quả đánh giá quá lạc quan.

Mỗi ảnh được cắt theo vùng biển báo, thêm lề 5% và đưa về RGB 64 × 64. Nhóm dùng 1.290 ảnh validation để chọn tham số tăng cường. Phần phân tích chi tiết dùng 430 ảnh test, tương ứng 10 ảnh mỗi lớp; kết quả trên toàn bộ tập test sẽ được trình bày riêng ở cuối phần thực nghiệm.

## Slide 4: Mô phỏng thiếu sáng và nhiễu

Từ ảnh gốc, nhóm tạo ảnh suy giảm theo thứ tự: Gamma làm tối, nhiễu Poisson, nhiễu Gaussian và cuối cùng giới hạn giá trị về khoảng từ 0 đến 1.

Gamma lớn hơn 1 làm ảnh tối đi. Nhiễu Poisson phụ thuộc vào tín hiệu, và trong cách mô phỏng này, tham số p càng nhỏ thì nhiễu tương đối càng mạnh. Nhiễu Gaussian được cộng thêm để tạo dao động cường độ. Ba mức nhẹ, vừa và nặng thay đổi đồng thời độ tối và mức nhiễu.

Nhìn vào các ảnh và histogram, có thể thấy phân bố cường độ dịch về phía tối, trong khi nét số dần khó quan sát hơn. Mọi phương pháp sẽ nhận đúng cùng một phiên bản ảnh suy giảm để so sánh công bằng. Đây là mô phỏng trên ảnh RGB, nên kết quả chưa đại diện đầy đủ cho mọi tình huống camera ban đêm.

## Slide 5: Bảy phương pháp tăng cường ảnh

Nhóm khảo sát bảy phương pháp, chia thành hai nhóm. Nhóm truyền thống gồm Gamma, HE, CLAHE và Multi-Scale Retinex, gọi tắt là MSR. Các phương pháp này chủ yếu điều chỉnh cường độ, histogram hoặc ước lượng thành phần chiếu sáng của ảnh.

Nhóm học sâu gồm Zero-DCE, Retinexformer và diffusion trong miền wavelet. Ba phương pháp này sử dụng trọng số đã được huấn luyện trước; nhóm không huấn luyện lại các bộ tăng cường trên GTSRB.

Ngoài bảy phương pháp trên, Identity là ảnh tối được giữ nguyên để làm đối chứng. Clean là ảnh gốc tham chiếu, giúp đánh giá mức suy giảm so với điều kiện đầu vào ban đầu. Như vậy, với mỗi mức thiếu sáng sẽ có tám nhánh đánh giá nếu tính cả Identity. Em sẽ trình bày nguyên lý các phương pháp, bắt đầu từ Gamma.

## Slide 6: Gamma và lựa chọn mức tăng sáng

Gamma tăng sáng bằng cách lấy cường độ ảnh đầu vào lũy thừa với hệ số alpha nằm giữa 0 và 1. Alpha càng nhỏ thì vùng tối được nâng sáng càng mạnh. Tuy nhiên, phép biến đổi tác động lên cả tín hiệu và nhiễu, nên tăng sáng mạnh chưa chắc giữ được nét biển báo tốt hơn.

Ở hàng ảnh phía trên, chúng ta so sánh ba giá trị alpha là 0,4; 0,6 và 0,8. Nhóm chọn alpha bằng kết quả nhận dạng trên validation, thay vì chỉ chọn ảnh trông sáng nhất. Cấu hình được chọn là 0,8.

Hàng ảnh phía dưới minh họa ảnh hưởng của tham số CLAHE và sẽ được giải thích ở các slide sau. Điểm cần ghi nhớ ở đây là mức tăng sáng phải phục vụ việc nhận dạng, không chỉ phục vụ cảm nhận bằng mắt.

## Slide 7: HE và ánh xạ từ phân phối tích lũy

HE là cân bằng histogram toàn cục. Đầu tiên, phương pháp thống kê số pixel ở từng mức sáng để tạo histogram. Sau đó, từ histogram tính phân phối tích lũy, hay CDF. Có thể hiểu CDF tại mức k là tỷ lệ pixel có cường độ không vượt quá mức k.

HE dùng thông tin tích lũy đó để ánh xạ mức sáng cũ sang mức sáng mới, nhằm phân bố lại các giá trị cường độ. Trong thí nghiệm, nhóm chuyển RGB sang YCrCb và chỉ xử lý kênh độ sáng Y, sau đó ghép lại với hai kênh màu.

Ưu điểm của HE là nhanh và ít tham số. Giới hạn là toàn ảnh dùng chung một ánh xạ, dù vùng biển báo và vùng nền có thể có điều kiện sáng khác nhau. Vì thế, histogram trải rộng hơn chưa đủ để kết luận các chi tiết cần nhận dạng đã rõ hơn.

## Slide 8: CLAHE và giới hạn tương phản theo vùng

CLAHE xử lý theo từng vùng nhỏ thay vì dùng một histogram chung cho cả ảnh. Quy trình gồm chia ảnh thành các ô, tính histogram từng ô, giới hạn những bin vượt ngưỡng, rồi xây dựng ánh xạ cục bộ. Kết quả giữa các ô được nội suy để tránh xuất hiện ranh giới khối rõ rệt.

Hai tham số quan trọng là clip limit và kích thước lưới. Clip limit giới hạn mức tập trung của histogram, giúp kiểm soát khuếch đại tương phản; nó không phải ngưỡng độ sáng của từng pixel.

Với ảnh 64 × 64, lưới 4 × 4 tạo các ô khoảng 16 × 16 pixel, còn lưới 8 × 8 tạo các ô nhỏ hơn. Ô quá nhỏ có thể khiến thống kê nhạy với nhiễu. Validation chọn clip limit bằng 2 và lưới 4 × 4. Tiếp theo là hình minh họa HE và CLAHE trên cùng một ảnh.

## Slide 9: HE và CLAHE trên kênh độ sáng

Ở slide này, chúng ta quan sát đồng thời ảnh đầu ra và histogram. Khi so sánh, em xin tập trung vào ba vùng: nền trắng bên trong biển báo, viền đỏ và nét của chữ số. Đây là những chi tiết có ích cho việc phân lớp.

HE điều chỉnh độ sáng theo thống kê toàn ảnh, nên có thể kéo giãn cả vùng sáng và vùng nhiễu. CLAHE thích ứng theo vùng và có giới hạn tương phản, nhưng hiệu quả vẫn phụ thuộc vào cách chia ô và mức nhiễu đầu vào.

Hình trên dùng tham số mặc định để minh họa cơ chế; riêng CLAHE đang dùng lưới 8 × 8. Cấu hình được chọn cho đánh giá là clip limit 2, lưới 4 × 4. Vì vậy, chúng ta cần phân biệt hình minh họa với cấu hình cuối cùng, đồng thời kiểm chứng bằng kết quả nhận dạng thay vì chỉ nhìn histogram.

## Slide 10: Retinex đa tỉ lệ và tính nhạy với nhiễu

Retinex xuất phát từ cách mô tả ảnh gồm thành phần phản xạ và thành phần chiếu sáng. Trong đó, chiếu sáng thường biến thiên tương đối chậm, còn phản xạ chứa những cấu trúc như biên, hình dạng và nét chữ.

MSR dùng các ảnh làm mờ Gaussian ở nhiều tỉ lệ để ước lượng thành phần chiếu sáng. Sau đó, phương pháp lấy chênh lệch trong miền log và tổng hợp kết quả giữa các tỉ lệ. Trong cấu hình được chọn, ba độ lệch chuẩn Gaussian là 5, 15 và 30.

Cách xử lý này có thể làm rõ chi tiết bị che bởi độ tối. Tuy nhiên, dao động nhiễu ở vùng tối cũng có thể được nhấn mạnh, vì phương pháp không có bước khử nhiễu riêng. Do đó, nhóm sẽ kiểm tra riêng ảnh chỉ tối và ảnh tối kèm nhiễu ở phần ablation, thay vì kết luận ngay rằng MSR luôn tốt hoặc luôn kém.

## Slide 11: Từ phản xạ ước lượng đến ảnh MSR đầu ra

Sau khi tính chênh lệch log ở ba tỉ lệ và lấy trung bình, kết quả chưa nằm sẵn trong miền cường độ ảnh mong muốn. Nhóm tiếp tục chuẩn hóa bằng hai mốc percentile 1% và 99%, rồi giới hạn đầu ra về khoảng từ 0 đến 1.

Hai mốc percentile giúp giảm ảnh hưởng của một số giá trị cực trị khi kéo giãn kết quả. Tuy nhiên, phép chuẩn hóa và clipping vẫn có thể làm thay đổi màu hoặc mất thông tin ở hai đầu miền giá trị. Nếu khoảng giữa hai mốc quá nhỏ, phương pháp trả lại ảnh đầu vào để tránh phép chia không ổn định.

Biến thể nhóm sử dụng là MSR giản lược, không có bước phục hồi màu riêng như MSRCR. Vì vậy, khi xem kết quả cần quan sát cả độ sáng, màu sắc và nhiễu. Tiếp theo, em chuyển sang các phương pháp học sâu, bắt đầu với Zero-DCE.

## Slide 12: Zero-DCE và đường cong tăng sáng

Khác với Gamma dùng một hệ số cố định, Zero-DCE sử dụng mạng để ước lượng các tham số điều chỉnh theo ảnh. Những tham số này tạo ra các đường cong thay đổi cường độ ở từng vị trí, và phép điều chỉnh được áp dụng lặp để tăng khả năng biểu diễn.

Trong công thức trên slide, I là ảnh ở mỗi bước, còn A là tham số do mạng dự đoán. Tùy tham số, đường cong có thể tăng hoặc giảm cường độ tại một vùng. Nhờ đó, cách điều chỉnh có thể thích ứng với nội dung ảnh.

Nghiên cứu gốc huấn luyện bằng các ràng buộc không cần ảnh tham chiếu ghép cặp. Trong đề tài này, nhóm chỉ sử dụng checkpoint có sẵn để suy luận. Dù tăng sáng thích ứng, đầu ra vẫn cần kiểm tra xem nhiễu có bị khuếch đại và nét số có được giữ lại hay không.

## Slide 13: Retinexformer và thông tin chiếu sáng

Retinexformer cũng khai thác ý tưởng về chiếu sáng và phản xạ, nhưng sử dụng mô hình học sâu để phục hồi ảnh. Có thể theo dõi sơ đồ theo hai phần: trước hết ước lượng thông tin chiếu sáng, sau đó dùng thông tin đó để hỗ trợ khối Transformer phục hồi ảnh.

Transformer cho phép mô hình học quan hệ giữa các vùng ảnh. Vì vậy, Retinexformer không đơn thuần là thực hiện MSR rồi thêm một bước xử lý phía sau. Sơ đồ trên chỉ thể hiện ý tưởng chính, không mô tả toàn bộ kiến trúc mạng.

Nhóm sử dụng trọng số được huấn luyện trên LOL-v1 và không huấn luyện lại trên GTSRB. Ảnh đầu vào 64 × 64 đáp ứng yêu cầu kích thước chia hết cho 4 của cấu hình sử dụng. Vì dữ liệu huấn luyện trước khác với ảnh biển báo, hiệu quả nhận dạng vẫn cần được đo thực nghiệm.

## Slide 14: Diffusion trong miền wavelet

Phương pháp diffusion được khảo sát là một mô hình phục hồi ảnh có điều kiện, trong đó ảnh thiếu sáng cung cấp thông tin để hướng dẫn quá trình phục hồi. Biểu diễn wavelet tách thông tin tần số thấp và các thành phần chi tiết tần số cao, giúp mô hình xử lý các thành phần này theo thiết kế của nó.

Trong cấu hình sử dụng, nhóm thực hiện 10 bước lấy mẫu và dùng trọng số EMA, tức trọng số trung bình trượt của mô hình. Đầu ra được lấy từ kết quả dự đoán ảnh, cắt về kích thước cần thiết và giới hạn trong miền từ 0 đến 1.

Diffusion có nhiều bước xử lý hơn các phép biến đổi truyền thống, nên cần đánh giá cả chất lượng nhận dạng lẫn thời gian chạy. Trước khi xem bảng số liệu, chúng ta sẽ quan sát các phương pháp học sâu trên cùng một ảnh biển báo.

## Slide 15: Các phương pháp trên cùng một ảnh

Các ảnh trên slide được tạo từ cùng một crop validation ở mức tối vừa, sử dụng tham số đã được chọn. Chúng ta nên so sánh nét chữ số, độ rõ của viền đỏ và sự thay đổi màu nền, thay vì chỉ nhìn xem ảnh nào sáng nhất.

Có thể thấy mỗi phương pháp tạo ra cách phân bố độ sáng và màu sắc khác nhau. Một phương pháp làm ảnh sáng rõ hơn về tổng thể vẫn có thể làm nét số nhòe hoặc khiến nhiễu nổi bật. Ngược lại, ảnh không sáng nhất vẫn có thể giữ những đặc trưng phù hợp cho bộ phân loại.

Tuy nhiên, một ảnh minh họa chưa đủ để xếp hạng toàn bộ phương pháp. Muốn kết luận, nhóm cần đánh giá trên cùng tập ảnh và cùng mô hình nhận dạng. Vì vậy, phần tiếp theo trình bày cách nhóm kiểm soát thí nghiệm và tổng hợp các chỉ số.

## Slide 16: Giao thức đánh giá với ResNet-18 cố định

Để tách ảnh hưởng của tăng cường ảnh khỏi ảnh hưởng của huấn luyện mô hình, nhóm huấn luyện ResNet-18 trên ảnh gốc với ba seed là 11, 22 và 33. Sau khi chọn checkpoint bằng validation, trọng số được giữ cố định khi nhận ảnh gốc, ảnh tối và ảnh tăng cường.

Ba noise seed là 101, 202 và 303 tạo các hiện thực nhiễu để kiểm tra độ ổn định. Mỗi phương pháp nhận cùng ảnh tối tương ứng. Tổng cộng có 219 lượt đánh giá điều kiện, chứ không phải 219 lần huấn luyện.

PSNR và SSIM được đo ở 64 × 64. Khi phân loại, ảnh mới được resize lên 224 × 224 và chuẩn hóa ImageNet. Kết quả được lấy trung bình qua noise seed trong mỗi training seed, rồi tổng hợp mean và độ lệch chuẩn qua ba training seed. Macro-F1 giúp các lớp có vai trò ngang nhau khi đánh giá.

## Slide 17: Macro-F1 khi mức thiếu sáng tăng

Bảng này trình bày Macro-F1 trên tập con 430 ảnh. Em xin tập trung vào ba dòng Identity, CLAHE và diffusion để thấy xu hướng chính.

Với Identity, Macro-F1 giảm từ 90,38% ở mức nhẹ xuống 75,35% ở mức vừa và 54,81% ở mức nặng. Như vậy, độ tối kết hợp nhiễu làm việc phân loại khó lên rõ rệt. Diffusion đạt lần lượt 92,01%; 82,71% và 63,72%, cải thiện so với Identity khoảng 1,63; 7,36 và 8,91 điểm phần trăm.

Ở mức nhẹ, diffusion và CLAHE chỉ chênh 0,03 điểm phần trăm, nên không nên xem đây là ưu thế chắc chắn. Khi tối nặng, ngay cả diffusion vẫn còn cách xa kết quả trên ảnh gốc. Diffusion đã được chọn bằng validation trước khi xem kết quả test này. Tiếp theo, nhóm xem liệu các chỉ số chất lượng ảnh có phản ánh đúng lợi ích nhận dạng hay không.

## Slide 18: Chất lượng ảnh và nhận dạng không luôn đồng nhất

Ở đây, các phương pháp được so sánh trên cùng 430 ảnh ở mức tối vừa. Zero-DCE có PSNR cao nhất trong các dòng được trình bày, đạt 20,08 dB, nhưng Macro-F1 chỉ đạt 69,30%. Trong khi đó, diffusion có PSNR thấp hơn, là 14,90 dB, nhưng Macro-F1 đạt 82,71%.

CLAHE cũng có SSIM cao hơn diffusion, nhưng Macro-F1 thấp hơn. Điều này cho thấy thứ hạng theo PSNR hoặc SSIM có thể khác thứ hạng theo hiệu năng phân loại.

PSNR đo sai số cường độ, SSIM đo mức tương đồng cấu trúc, còn ResNet-18 cần những đặc trưng phục vụ nhận dạng biển báo. Vì vậy, nhóm sử dụng các chỉ số chất lượng để bổ trợ phân tích, nhưng chọn pipeline theo mục tiêu nhận dạng. Bảng số liệu cho thấy sự khác biệt; muốn giải thích chính xác lỗi của từng ảnh thì vẫn cần quan sát chi tiết hơn.

## Slide 19: Tương quan giữa chất lượng ảnh và Macro-F1

Để xem mối liên hệ rộng hơn giữa hai nhóm chỉ số, nhóm tính tương quan trên tám phương pháp ở từng mức suy giảm, bao gồm Identity và loại ảnh Clean. Biểu đồ thể hiện Spearman, tức mức độ phù hợp giữa thứ hạng theo chất lượng ảnh và thứ hạng theo Macro-F1.

Với PSNR, hệ số giảm từ khoảng 0,667 ở mức nhẹ xuống 0,333 và 0,286 ở hai mức sau. Với SSIM, hệ số là khoảng 0,667 ở mức nhẹ và 0,619 ở mức vừa, nặng. Trong khảo sát này, thứ hạng theo SSIM phù hợp với F1 hơn thứ hạng theo PSNR ở hai mức suy giảm mạnh hơn.

Đây vẫn là phân tích khám phá vì mỗi mức chỉ có tám điểm và các phương pháp dùng chung tập ảnh. Tương quan dương không có nghĩa tăng một chỉ số chất lượng sẽ trực tiếp làm tăng hiệu năng nhận dạng.

## Slide 20: Vai trò của nhiễu qua thí nghiệm ablation

Ba mức suy giảm chính thay đổi đồng thời nhiều tham số. Để hiểu rõ hơn vai trò của từng loại nhiễu, nhóm giữ Gamma bằng 2,2 và so sánh bốn điều kiện: chỉ Gamma, thêm Gaussian, thêm Poisson và thêm cả hai loại nhiễu. Checkpoint và tham số tăng cường được giữ nguyên.

Kết quả nổi bật là MSR đạt Macro-F1 98,20% khi chỉ làm tối bằng Gamma, nhưng giảm còn 50,97% khi thêm cả hai nhiễu, tức giảm 47,23 điểm phần trăm. Như vậy, không thể kết luận MSR luôn kém; hiệu quả của nó phụ thuộc mạnh vào điều kiện đầu vào.

Trong cấu hình này, việc thêm Poisson làm MSR giảm mạnh hơn thêm Gaussian. Tuy nhiên, hai mức nhiễu không được cân bằng về năng lượng, nên kết quả đó không có nghĩa Poisson luôn gây hại hơn trong mọi tình huống. Biểu đồ chỉ chọn ba phương pháp đại diện để dễ theo dõi.

## Slide 21: Đối chiếu trên toàn bộ tập test

Sau phần phân tích trên 430 ảnh, nhóm đối chiếu với thí nghiệm riêng trên toàn bộ 12.630 ảnh test. Hai phần được trình bày tách biệt, không dùng chất lượng ảnh của tập con để ghép với Macro-F1 của toàn bộ tập test.

So với Identity, diffusion tăng Macro-F1 lần lượt 1,45; 5,83 và 5,18 điểm phần trăm ở mức nhẹ, vừa và nặng. CLAHE đạt kết quả cao hơn diffusion ở mức nhẹ, còn diffusion có giá trị trung bình cao hơn trong hai mức còn lại.

Các kết quả này cho thấy xu hướng tăng cường ảnh có ích trong một số điều kiện, nhưng không có một phương pháp vượt trội rõ ở mọi mức. Nhóm vẫn giữ pipeline đã chọn từ validation, thay vì đổi lựa chọn sau khi nhìn bảng test. Phần full-test là đối chiếu kết quả, chưa được kiểm tra lại dự đoán từng ảnh trong quá trình soạn báo cáo.

## Slide 22: Hiệu năng và chi phí xử lý

Ngoài hiệu năng nhận dạng, nhóm xem xét thời gian của riêng khối tăng cường. Ở mức tối vừa trên tập con, Gamma mất khoảng 0,219 mili giây mỗi ảnh, CLAHE khoảng 0,400 mili giây, còn diffusion khoảng 222,658 mili giây.

Diffusion đạt Macro-F1 82,71%, cao hơn CLAHE 2,57 điểm phần trăm, nhưng phải đánh đổi bằng chi phí xử lý lớn hơn. Nếu ưu tiên Macro-F1 theo giao thức hiện tại, diffusion là phương án đã được chọn. Nếu cần khảo sát một giải pháp xử lý nhẹ, CLAHE là ứng viên đáng xem xét.

Cần lưu ý mô hình sâu chạy trên Tesla T4, còn các phép truyền thống chạy CPU. Các con số chỉ đo thời gian enhancer, chưa gồm đọc ghi ảnh, mô hình phân loại hoặc phát hiện biển báo. Vì vậy, chúng chưa phải độ trễ của một hệ thống nhận dạng hoàn chỉnh.

## Slide 23: Quy trình tiền xử lý được đề xuất

Từ kết quả lựa chọn trên validation, nhóm đề xuất quy trình gồm bốn bước. Đầu tiên, cắt vùng biển báo, thêm lề 5% và chuẩn hóa về RGB 64 × 64. Tiếp theo, tăng cường bằng diffusion với cấu hình đã khóa. Sau đó, kiểm tra đầu ra, resize lên 224 × 224 và chuẩn hóa ImageNet đúng một lần. Cuối cùng, ảnh được đưa qua checkpoint ResNet-18 cố định để phân loại.

Với ảnh thực đã bị thiếu sáng, không thực hiện thêm bước mô phỏng làm tối. Bước đó chỉ phục vụ thiết kế thí nghiệm.

CLAHE với clip limit 2 và lưới 4 × 4 là phương án để tiếp tục khảo sát khi có yêu cầu xử lý nhẹ. Nhóm chưa xây dựng bộ tự động chuyển giữa CLAHE và diffusion. Do đó, quy trình được gọi là lựa chọn tốt nhất trong phạm vi các phương án và giao thức đã khảo sát.

## Slide 24: Kết luận và hướng phát triển

Qua khảo sát, nhóm rút ra ba ý chính. Thứ nhất, tăng cường ảnh có thể cải thiện khả năng nhận dạng trên cùng một bộ phân loại cố định. Thứ hai, PSNR hoặc SSIM cao hơn chưa bảo đảm Accuracy và Macro-F1 cao hơn. Thứ ba, nhiễu ảnh hưởng mạnh đến hiệu quả của các phương pháp tăng sáng, thể hiện rõ qua thí nghiệm MSR.

Trong giao thức hiện tại, diffusion được lựa chọn theo Macro-F1 validation, còn CLAHE là ứng viên cho hướng xử lý nhẹ. Tuy nhiên, thí nghiệm mới sử dụng ảnh crop và suy giảm tổng hợp. Hướng tiếp theo là thử ảnh đêm thực, đồng thời khảo sát khử nhiễu kết hợp tăng sáng và kiểm tra xem các nét biển báo có bị mất đi hay không.

Phần trình bày của Nhóm 2 xin kết thúc tại đây. Chúng em cảm ơn thầy và các bạn đã lắng nghe, và xin nhận các câu hỏi, góp ý.
