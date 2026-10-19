# Slide thuyết trình LaTeX Beamer

24 slide tiếng Việt, tỉ lệ 16:9, phù hợp khoảng 18–20 phút. Thiết kế theo phong cách bài giảng: nền trắng, dải tiêu đề xám nhạt, tiêu đề và header bảng màu xanh dương, footer mỏng gồm bốn ô: Nhóm 3, Xử lý ảnh, trích dẫn paper (nếu có) và số trang. Nội dung tập trung xử lý ảnh, toàn bộ kết quả nhận dạng dùng ResNet-18.

- `main.tex`: mã nguồn LaTeX, có ghi chú `\note{...}` cho từng slide.
- `thuyet_trinh.pdf`: bản trình chiếu cuối.
- `loi_dan.md`: lời dẫn và gợi ý thời lượng để tập thuyết trình.
- `images/`: hình thực nghiệm lấy từ notebook đã chạy, logo từ paper mẫu.
- `data/`: số liệu nguồn dùng đối chiếu các bảng và biểu đồ.
- `build.ps1`: biên dịch hai lần bằng XeLaTeX và xuất PDF.

## Biên dịch

Từ gốc dự án, chạy PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File report/slides/build.ps1
```

Hoặc chạy thủ công tại thư mục `report/slides`:

```powershell
xelatex --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
xelatex --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=_build main.tex
Copy-Item -LiteralPath _build/main.pdf -Destination thuyet_trinh.pdf
```

Máy hiện có MiKTeX, Beamer, PGFPlots và Arial. Không cài thêm dependency. Chép toàn bộ thư mục này để giữ hình và dữ liệu. Với Overleaf chọn XeLaTeX; nếu không có Arial, đổi `Arial` thành `TeX Gyre Heros` trong hai lệnh đặt font.

## Nội dung và nguồn

Các bảng chính, ablation và tương quan thuộc notebook ResNet-18 trên 430 ảnh. Slide 21 là full-test 12.630 ảnh từ README của dự án, được ghi rõ là lần chạy riêng. Không dùng chất lượng ảnh subset ghép với F1 full-test. Không dùng số liệu của classifier cũ.

Thanh footer chỉ hiển thị paper liên quan theo tác giả, hội nghị/tạp chí và năm; có thể bấm để mở bài báo. Slide không có paper để trống ô giữa. Thông tin truy vết notebook/code được giữ trong `\note`, không xuất hiện trong PDF trình chiếu. Báo cáo đầy đủ và dẫn nguồn học thuật nằm ở `../bao_cao.pdf`.

