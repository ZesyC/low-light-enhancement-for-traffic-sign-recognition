# Kế hoạch cải thiện mô hình CNN phân loại biển báo giao thông

## Kết quả hiện tại (Baseline)

Mô hình: CNN 3 tầng Conv2d (32 -> 64 -> 128), không Dropout, không BatchNorm, không Data Augmentation.

| Chỉ số | Seed 11 | Seed 22 | Seed 33 | Trung bình |
|---|---|---|---|---|
| Accuracy | 76.22% | 74.61% | 75.32% | **75.38%** |
| Macro-F1 | 64.71% | 61.71% | 64.82% | **63.75%** |
| Weighted-F1 | 75.63% | 73.78% | 74.70% | **74.70%** |
| Epoch tốt nhất | 40 | 36 | 38 | - |
| Train Macro-F1 tại epoch tốt nhất | 97.7% | 96.9% | 97.4% | 97.3% |
| Gap (Train - Val) Macro-F1 | 33.0pp | 35.2pp | 32.6pp | **33.6pp** |

### Các lớp yếu nhất (F1 trung bình 3 seed)

| Class ID | F1 trung bình | Support | Vấn đề chính |
|---|---|---|---|
| 0 | 0.00 | 30 | Toàn bộ bị nhầm thành class 1 |
| 19 | 0.00 | 30 | Toàn bộ bị nhầm thành class 30 |
| 20 | 0.04 | 60 | Gần như toàn bộ bị nhầm |
| 21 | 0.05 | 60 | Gần như toàn bộ bị nhầm |
| 37 | 0.12 | 30 | Bị nhầm rất nhiều |
| 27 | 0.39 | 60 | Dưới một nửa đúng |
| 29 | 0.51 | 60 | Khoảng một nửa đúng |

### Vấn đề chính cần giải quyết

1. **Overfitting nặng**: Gap train-val Macro-F1 trung bình 33.6 điểm phần trăm.
2. **Các lớp thiểu số bị bỏ qua hoàn toàn**: Class 0, 19, 20, 21 có F1 = 0 hoặc gần 0.
3. **Mô hình quá đơn giản**: 3 tầng Conv2d không có cơ chế chính quy hóa (regularization).

---

## Cải thiện 1: Data Augmentation

### Mục tiêu

Giảm overfitting bằng cách tăng tính đa dạng của dữ liệu huấn luyện mà không cần thu thập thêm ảnh mới.

### Kế hoạch thực hiện

**File cần sửa**: `src/training.py` (class `CropDataset`)

**Bước 1**: Thêm thư viện `torchvision.transforms` vào phần import của `src/training.py`.

**Bước 2**: Tạo một đối tượng transform dùng cho tập train:
```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomRotation(degrees=10),            # Xoay ngẫu nhiên +-10 độ
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Dịch chuyển nhẹ 5%
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Thay đổi màu sắc nhẹ
    T.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # Xóa ngẫu nhiên vùng nhỏ
])
```

**Bước 3**: Sửa class `CropDataset` để nhận thêm tham số `transform`:
```python
class CropDataset(Dataset):
    def __init__(self, root, rows, transform=None):
        self.root, self.rows = Path(root), rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tensor = image_tensor(read_image(self.root / row['crop_path']))
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, int(row['class_id'])
```

**Bước 4**: Truyền `transform=train_transform` cho CropDataset của tập train, giữ `transform=None` cho tập validation.

**Lưu ý quan trọng**:
- Chỉ áp dụng augmentation cho tập **train**, KHÔNG áp dụng cho tập validation/test.
- Không dùng RandomHorizontalFlip vì nhiều biển báo giao thông có ý nghĩa khác nhau khi lật ngang (ví dụ biển rẽ trái vs rẽ phải).
- Giữ mức độ augmentation nhẹ (rotation +-10 độ, translate 5%) để không làm biến dạng hình dáng biển báo.

### Bảng đánh giá kết quả

| Chỉ số | Baseline | Sau Augmentation | Thay đổi |
|---|---|---|---|
| Accuracy (trung bình 3 seed) | 75.38% | **79.93%** | **+4.55pp** |
| Macro-F1 (trung bình 3 seed) | 63.75% | **70.00%** | **+6.25pp** |
| Weighted-F1 (trung bình 3 seed) | 74.70% | **79.41%** | **+4.71pp** |
| Gap Train-Val Macro-F1 | 33.6pp | **25.3pp** | **-8.3pp** |
| F1 Class 0 | 0.00 | 0.07 | +0.07 |
| F1 Class 19 | 0.00 | 0.00 | 0.00 |
| F1 Class 20 | 0.04 | **0.40** | **+0.36** |
| F1 Class 21 | 0.05 | 0.06 | +0.01 |
| F1 Class 37 | 0.12 | 0.19 | +0.07 |
| Epoch tốt nhất (trung bình) | 38 | 39.3 | +1.3 |
| Ghi chú | - | Cả 3 seed đạt Macro-F1 69.35%-70.82%; class 19 vẫn có F1 = 0. | Cải thiện tổng quát hóa, chưa xử lý xong mất cân bằng lớp. |

---

## Cải thiện 2: Batch Normalization + Dropout

### Mục tiêu

Tăng khả năng tổng quát hóa của mô hình bằng 2 kỹ thuật chính quy hóa phổ biến nhất trong deep learning.

### Kế hoạch thực hiện

**File cần sửa**: `src/model.py` (hàm `make_cnn`)

**Bước 1**: Thêm BatchNorm2d sau mỗi tầng Conv2d (trước hàm kích hoạt ReLU) để ổn định phân phối giá trị giữa các tầng.

**Bước 2**: Thêm Dropout (tỷ lệ 0.3 - 0.5) trước tầng Linear cuối cùng để tắt ngẫu nhiên một phần nơ-ron, ép mô hình học đặc trưng tổng quát hơn.

**Bước 3**: Kiến trúc mới sẽ như sau:
```python
def make_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),          # Mới: chuẩn hóa phân phối
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),          # Mới: chuẩn hóa phân phối
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),         # Mới: chuẩn hóa phân phối
        nn.ReLU(),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.4),            # Mới: tắt ngẫu nhiên 40% nơ-ron
        nn.Linear(128, 43),
    )
```

**Giải thích**:
- **BatchNorm2d**: Chuẩn hóa giá trị đầu ra của mỗi tầng Conv2d về trung bình 0, phương sai 1 trước khi truyền vào ReLU. Giúp gradient ổn định hơn, model hội tụ nhanh hơn, đồng thời có tác dụng regularization nhẹ.
- **Dropout(0.4)**: Trong mỗi bước lan truyền tới, ngẫu nhiên tắt 40% giá trị của vector 128 chiều trước khi vào tầng phân loại. Ép mô hình không phụ thuộc vào bất kỳ đặc trưng đơn lẻ nào.

**Lưu ý**:
- BatchNorm có hành vi khác nhau giữa `model.train()` và `model.eval()`. Code hiện tại đã gọi đúng `model.train(split == 'train')` nên không cần sửa thêm.
- Dropout chỉ kích hoạt khi `model.train(True)`, tự động tắt khi `model.eval()`, nên cũng không ảnh hưởng đến inference.
- Nên thử nhiều giá trị dropout: 0.3, 0.4, 0.5 để xem giá trị nào tốt nhất.

### Bảng đánh giá kết quả

| Chỉ số | Baseline | Sau BN + Dropout | Thay đổi |
|---|---|---|---|
| Accuracy (trung bình 3 seed) | 75.38% | **92.45%** | **+17.07pp** |
| Macro-F1 (trung bình 3 seed) | 63.75% | **87.57%** | **+23.82pp** |
| Weighted-F1 (trung bình 3 seed) | 74.70% | **92.17%** | **+17.47pp** |
| Gap Train-Val Macro-F1 | 33.6pp | **4.27pp** | **-29.33pp** |
| F1 Class 0 | 0.00 | **0.83** | **+0.83** |
| F1 Class 19 | 0.00 | 0.00 | 0.00 |
| F1 Class 20 | 0.04 | **0.77** | **+0.73** |
| F1 Class 21 | 0.05 | **0.84** | **+0.79** |
| Epoch tốt nhất (trung bình) | 38 | 37.7 | -0.3 |
| Ghi chú | - | Cả 3 seed hoàn thành 40 epoch; Macro-F1 87.01%–88.57%; class 19 vẫn F1 = 0. | Regularization hiệu quả, overfitting giảm mạnh, class 19 chưa xử lý xong. |

### Đánh giá kết quả

Cả 3 seed đều `completed`, chạy hết 40/40 epoch, và checkpoint khớp đúng epoch có validation Macro-F1 tốt nhất.

| Seed | Best epoch | Validation Accuracy | Validation Macro-F1 | Train Macro-F1 tại best epoch | Gap Train–Val F1 |
|---|---|---|---|---|---|
| 11 | 37 | 92.11% | 87.01% | 91.71% | 4.70pp |
| 22 | 39 | 92.81% | 88.57% | 92.39% | 3.82pp |
| 33 | 37 | 92.43% | 87.14% | 91.45% | 4.30pp |
| Trung bình ± SD | 37.7 | **92.45% ± 0.35pp** | **87.57% ± 0.86pp** | 91.85% | **4.27pp** |

BatchNorm và Dropout giảm overfitting rất mạnh: gap train–val Macro-F1 từ 33.6pp xuống còn 3.82–4.70pp. Accuracy và Macro-F1 đều vượt mục tiêu (90% và 85%), đồng thời ổn định giữa các seed.

Class 0, 20 và 21 đã phục hồi rõ. Class 19 vẫn F1 = 0 ở cả 3 seed, chủ yếu bị nhầm thành class 23 hoặc 31. Mục tiêu “không còn lớp nào có F1 = 0” chưa đạt, nhưng CNN hiện tại đủ tốt để khóa làm bộ phân loại cố định cho thí nghiệm chính A. Xử lý mất cân bằng lớp để lại cho bước 3 nếu cần.

---

## Cải thiện 3: Xử lý mất cân bằng lớp (Class Imbalance)

### Mục tiêu

Buộc mô hình phải học đều tất cả 43 lớp, đặc biệt các lớp chỉ có 30-60 ảnh (class 0, 19, 20, 21, 37).

### Kế hoạch thực hiện

Có 2 phương án, nên thử cả 2 và so sánh:

#### Phương án A: Weighted Cross-Entropy Loss

**File cần sửa**: `src/training.py` (hàm `train`)

**Bước 1**: Tính trọng số nghịch đảo với số lượng mẫu của mỗi lớp:
```python
from collections import Counter

counts = Counter(int(r['class_id']) for r in train_rows)
total = sum(counts.values())
weights = torch.tensor([total / (43 * counts[c]) for c in range(43)],
                       dtype=torch.float32).to(device)
```

**Bước 2**: Truyền `weight` vào hàm loss:
```python
loss = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
```

**Giải thích**: Lớp chỉ có 30 ảnh sẽ có trọng số lớn hơn lớp có 450 ảnh, nên khi mô hình đoán sai lớp nhỏ, giá trị loss bị phạt nặng hơn gấp 15 lần, buộc optimizer phải chú ý đến lớp đó.

#### Phương án B: WeightedRandomSampler (Oversampling)

**File cần sửa**: `src/training.py` (hàm `train`)

**Bước 1**: Tính trọng số cho từng mẫu và tạo sampler:
```python
from torch.utils.data import WeightedRandomSampler

counts = Counter(int(r['class_id']) for r in train_rows)
sample_weights = [1.0 / counts[int(r['class_id'])] for r in train_rows]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows),
                                replacement=True,
                                generator=torch.Generator().manual_seed(seed))
```

**Bước 2**: Thay `shuffle=True` bằng `sampler=sampler` trong DataLoader của tập train:
```python
DataLoader(CropDataset(root, train_rows), batch_size=cfg['batch_size'],
           sampler=sampler, num_workers=0)
```

**Giải thích**: Thay vì lấy tuần tự, DataLoader sẽ lấy lại (resample) các ảnh của lớp nhỏ nhiều lần hơn trong mỗi epoch, đảm bảo mỗi batch có phân phối lớp tương đối đều.

**Lưu ý**:
- Phương án A đơn giản hơn, chỉ sửa 2 dòng code.
- Phương án B cần bỏ `shuffle=True` vì `sampler` và `shuffle` không dùng đồng thời được.
- Có thể kết hợp cả A và B nhưng thường chỉ cần 1 trong 2.

### Bảng đánh giá kết quả

| Chỉ số | Baseline | Weighted Loss (A) | Oversampling (B) | Thay đổi tốt nhất |
|---|---|---|---|---|
| Accuracy (trung bình 3 seed) | 75.38% | | | |
| Macro-F1 (trung bình 3 seed) | 63.75% | | | |
| Weighted-F1 (trung bình 3 seed) | 74.70% | | | |
| Gap Train-Val Macro-F1 | 33.6pp | | | |
| F1 Class 0 | 0.00 | | | |
| F1 Class 19 | 0.00 | | | |
| F1 Class 20 | 0.04 | | | |
| F1 Class 21 | 0.05 | | | |
| Epoch tốt nhất (trung bình) | 38 | | | |
| Ghi chú | | | | |

---

## Cải thiện 4: Tăng độ phức tạp mô hình (Model Capacity)

### Mục tiêu

Tăng khả năng biểu diễn của mô hình để phân biệt được các lớp có hình dạng tương tự nhau (ví dụ: class 0 vs class 1 chỉ khác chữ số "20" vs "30" trên nền tròn đỏ).

### Kế hoạch thực hiện

Có 2 phương án:

#### Phương án A: Tăng số tầng CNN hiện tại (Deeper CNN)

**File cần sửa**: `src/model.py`

```python
def make_cnn():
    return nn.Sequential(
        # Block 1: 64x64 -> 32x32
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),    # Mới: thêm 1 tầng
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Block 2: 32x32 -> 16x16
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),    # Mới: thêm 1 tầng
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Block 3: 16x16 -> 8x8
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),  # Mới: thêm 1 tầng
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(128, 43),
    )
```

**Giải thích**: Mỗi block tăng thêm 1 tầng Conv2d, tăng tổng số tầng từ 3 lên 6. Giúp mô hình học được các đặc trưng tinh tế hơn (chi tiết chữ số, đường viền) mà vẫn giữ nguyên kích thước đầu ra.

#### Phương án B: Dùng ResNet-18 pre-trained (Transfer Learning)

**File cần sửa**: `src/model.py`

```python
import torchvision.models as models

def make_cnn():
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 43),
    )
    return backbone
```

**Giải thích**: ResNet-18 đã được huấn luyện trên ImageNet (1.2 triệu ảnh), đã học được các đặc trưng căn bản như cạnh, góc, màu sắc. Chỉ cần thay tầng phân loại cuối cùng từ 1000 lớp (ImageNet) thành 43 lớp (GTSRB) và fine-tune.

**Lưu ý quan trọng**:
- Phương án B sẽ làm tăng độ phức tạp tính toán đáng kể (từ ~100K tham số lên ~11M tham số).
- Phương án B có thể cần giảm learning rate xuống 0.0001 để fine-tune tốt hơn.
- Nên ưu tiên thử Phương án A trước vì nó đơn giản hơn và giữ nguyên phong cách kiến trúc của dự án.
- Phương án B có thể cần sửa thêm `src/training.py` để giảm learning rate cho backbone và giữ learning rate cao cho tầng fc mới.
- Nếu dùng Phương án B, ảnh đầu vào 64x64 vẫn chạy được vì ResNet-18 có AdaptiveAvgPool2d.

### Bảng đánh giá kết quả

| Chỉ số | Baseline | Deeper CNN (A) | ResNet-18 (B) | Thay đổi tốt nhất |
|---|---|---|---|---|
| Accuracy (trung bình 3 seed) | 75.38% | | | |
| Macro-F1 (trung bình 3 seed) | 63.75% | | | |
| Weighted-F1 (trung bình 3 seed) | 74.70% | | | |
| Gap Train-Val Macro-F1 | 33.6pp | | | |
| F1 Class 0 | 0.00 | | | |
| F1 Class 19 | 0.00 | | | |
| F1 Class 20 | 0.04 | | | |
| F1 Class 21 | 0.05 | | | |
| Epoch tốt nhất (trung bình) | 38 | | | |
| Số tham số mô hình | ~100K | ~200K | ~11M | |
| Thời gian train / epoch | | | | |
| Ghi chú | | | | |

---

## Cải thiện 5: Điều chỉnh Hyperparameter

### Mục tiêu

Tìm bộ siêu tham số phù hợp hơn để mô hình hội tụ tốt và tránh overfitting.

### Kế hoạch thực hiện

**File cần sửa**: `configs/default.json` (phần `cnn`)

**Các tham số cần thử nghiệm**:

| Tham số | Giá trị hiện tại | Giá trị thử nghiệm | Lý do |
|---|---|---|---|
| learning_rate | 0.001 | 0.0005, 0.0003 | LR hiện tại có thể quá cao, gây dao động val_loss |
| batch_size | 64 | 32, 128 | Batch nhỏ hơn tăng nhiễu khi cập nhật, có thể tổng quát tốt hơn |
| epochs | 40 | 60, 80 | Mô hình có thể chưa hội tụ hết (seed 11 chạy hết 40 epoch) |
| patience | 7 | 10, 15 | Tăng patience để cho mô hình có hội vượt qua các "bình nguyên" |

**Thử nghiệm gợi ý (theo thứ tự ưu tiên)**:

1. Giảm learning_rate xuống 0.0005, giữ nguyên các tham số khác.
2. Giảm learning_rate xuống 0.0003, tăng epochs lên 60.
3. Giảm batch_size xuống 32, giữ learning_rate = 0.001.
4. Tăng patience lên 10 hoặc 15.

**Lưu ý**:
- Mỗi lần chỉ thay đổi 1 tham số để biết tham số nào thực sự ảnh hưởng.
- Lưu kết quả vào thư mục output riêng, ví dụ `outputs/train_lr0005/`, `outputs/train_bs32/`.
- Giảm learning_rate thường là bước hiệu quả nhất khi val_loss dao động mạnh.

### Bảng đánh giá kết quả

| Cấu hình | LR | BS | Epochs | Patience | Accuracy | Macro-F1 | Gap | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.001 | 64 | 40 | 7 | 75.38% | 63.75% | 33.6pp | |
| Thử nghiệm 1 | 0.0005 | 64 | 40 | 7 | | | | |
| Thử nghiệm 2 | 0.0003 | 64 | 60 | 7 | | | | |
| Thử nghiệm 3 | 0.001 | 32 | 40 | 7 | | | | |
| Thử nghiệm 4 | 0.001 | 64 | 60 | 15 | | | | |
| Cấu hình tốt nhất | | | | | | | | |

---

## Thứ tự thực hiện khuyến nghị

Nên thực hiện theo thứ tự từ trên xuống dưới. Mỗi bước xây dựng trên kết quả của bước trước:

```
Bước 1: Data Augmentation (Cải thiện 1)
   |
   v
Bước 2: BatchNorm + Dropout (Cải thiện 2)
   |
   v
Bước 3: Weighted Loss hoặc Oversampling (Cải thiện 3)
   |
   v
Bước 4: Đánh giá kết quả tổng hợp sau 3 bước trên
   |
   v  (Nếu Macro-F1 vẫn dưới 85%)
Bước 5: Tăng độ phức tạp mô hình (Cải thiện 4)
   |
   v
Bước 6: Fine-tune hyperparameter (Cải thiện 5)
```

**Lý do thứ tự này**:
- Bước 1-2-3 là các thay đổi nhẹ, an toàn, không ảnh hưởng đến cấu trúc dự án.
- Bước 4 (mô hình lớn hơn) chỉ cần khi 3 bước trước vẫn chưa đủ.
- Bước 5 (hyperparameter) nên làm cuối cùng vì giá trị tốt nhất phụ thuộc vào kiến trúc và dữ liệu đã được cải thiện ở các bước trước.

---

## Bảng tổng hợp cuối cùng

Sau khi hoàn thành tất cả các cải thiện, điền vào bảng dưới để so sánh tổng thể:

| Chỉ số | Baseline | +Augment | +BN/Drop | +ClassBalance | +Model | +Hyper | Kết quả cuối |
|---|---|---|---|---|---|---|---|
| Accuracy | 75.38% | **79.93%** | **92.45%** | | | | |
| Macro-F1 | 63.75% | **70.00%** | **87.57%** | | | | |
| Weighted-F1 | 74.70% | **79.41%** | **92.17%** | | | | |
| Gap Train-Val | 33.6pp | **25.3pp** | **4.27pp** | | | | |
| F1 Class 0 | 0.00 | 0.07 | **0.83** | | | | |
| F1 Class 19 | 0.00 | 0.00 | 0.00 | | | | |
| F1 Class 20 | 0.04 | **0.40** | **0.77** | | | | |
| F1 Class 21 | 0.05 | 0.06 | **0.84** | | | | |
| F1 Class 37 | 0.12 | 0.19 | **0.67** | | | | |
| Số tham số | ~100K | ~100K | ~100K | | | | |
| Thời gian train | | Chưa đo | Chưa đo | | | | |

### Mục tiêu hướng tới

- Accuracy >= 90%
- Macro-F1 >= 85%
- Gap Train-Val Macro-F1 <= 10pp
- Không còn lớp nào có F1 = 0
