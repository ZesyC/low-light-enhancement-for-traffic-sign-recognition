# Báo cáo bài tập lớn môn Xử lý ảnh

Đề tài: **Khảo sát ảnh hưởng của xử lý ảnh đến hiệu năng nhận dạng biển báo giao thông trong điều kiện thiếu sáng.**

- [Báo cáo PDF](bao_cao.pdf): bản báo cáo hoàn chỉnh.
- [Slide PDF](thuyet_trinh.pdf): bản thuyết trình.
- `latex/main.tex`: tài liệu chính, biên dịch bằng XeLaTeX.
- `latex/sections/`: nội dung tiếng Việt chia theo phần.
- `latex/images/`: hình thực nghiệm trích từ notebook ResNet-18 và biểu đồ giải tích.
- `latex/tables/`: bảng LaTeX sinh từ số liệu nguồn.
- `evidence/`: CSV nguồn và `provenance.json` ghi SHA-256, vị trí cell, giới hạn kiểm chứng.
- `build_assets.py`: trích xuất và tổng hợp lại kết quả ResNet-18, không đọc kết quả classifier cũ.
- `report_mau.pdf`: mẫu gốc được giữ nguyên.

`latex/` và `slides/` được giữ riêng ở local và đã loại khỏi Git. Clone repo chỉ có các bản PDF cùng tài liệu/số liệu được theo dõi; hướng dẫn biên dịch dưới đây cần bộ nguồn local.

## Biên dịch trên Windows PowerShell

Chạy từ thư mục gốc dự án, dùng MiKTeX/XeLaTeX và font Times New Roman đã có trên máy:

```powershell
Push-Location report/latex
xelatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
xelatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
Copy-Item -LiteralPath report/latex/main.pdf -Destination report/bao_cao.pdf
```

Hai lượt biên dịch cập nhật mục lục, số hình/bảng và tham chiếu. `--disable-installer` ngăn MiKTeX tự cài thêm gói. Không cần BibTeX hoặc kết nối Internet khi biên dịch. Khi tải lên Overleaf, tải toàn bộ thư mục `latex/`, chọn compiler XeLaTeX; nếu môi trường không có Times New Roman, cần cung cấp font hợp lệ hoặc dùng TeX Gyre Termes và chấp nhận khác biệt kiểu chữ.

Để tạo lại bảng/hình từ nguồn đang có, chạy `python report/build_assets.py` từ gốc dự án. Script dùng Python stdlib, NumPy và Matplotlib đã có trong requirements của dự án; không cài thêm dependency. Logo là ảnh nhúng trích từ trang bìa mẫu; script không tạo lại logo.

## Định dạng và nguồn

Mẫu có 31 trang, A4, một cột, Times New Roman; phần thân chủ yếu 12,5 pt PDF, khoảng cách dòng khoảng 14,4 pt. Báo cáo dùng 12,5 bp/14,4 bp trong XeLaTeX để tương đương đơn vị PDF; lề trái 30 mm, phải 20 mm, trên 32 mm, dưới 30 mm; căn đều, không thụt đầu dòng, tiêu đề đậm đánh số, số trang giữa chân trang. Bìa có khung viền đen, giữ trường, Nhóm 2 và thành viên của mẫu; môn đổi thành Xử lý ảnh và giảng viên hướng dẫn là TS. Nguyễn Ngọc Giang.

Toàn bộ số liệu nhận dạng dùng **ResNet-18**. Bộ 219 lượt × 430 ảnh và 288 lượt ablation lấy từ notebook đã thực thi; script tính lại tổng hợp 219 bản ghi và tương quan. Kiểm chứng mức dự đoán từng ảnh là kết quả được ghi trong notebook, không phải chạy lại tại máy này. Full-test 12.630 ảnh lấy từ README và được tách rõ, không ghép với PSNR/SSIM của subset. Không sử dụng kết quả hoặc hình từ thí nghiệm classifier cũ.

Các phương pháp: Gamma, HE, CLAHE, MSR giản lược, Zero-DCE, Retinexformer, Wavelet-based Diffusion; Identity là đối chứng. PSNR/SSIM đo tại 64×64; ResNet-18 nhận ảnh 224×224 sau resize và normalize. Mục tiêu lựa chọn là Macro-F1 validation; không lấy kết quả test để chọn lại pipeline.
