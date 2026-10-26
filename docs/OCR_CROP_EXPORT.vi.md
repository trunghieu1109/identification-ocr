# Xuất ảnh crop và chữ OCR mặt trước

Script `scripts/export-front-ocr.py` nhận **một ảnh mặt trước**, dùng pipeline
namlhn GPU (Paddle detector + bộ đọc DocU VietOCR) để lưu từng ảnh crop và text
tương ứng. Các ảnh được giữ ngay ở lượt nhận dạng, gồm phần đệm và căn chỉnh
phối cảnh thực tế, trước bước resize/chuẩn hóa tensor của bộ nhận dạng.

Chạy từ thư mục repository, dùng image Docker và các model đã cài:

```bash
python3 scripts/export-front-ocr.py /duong/dan/front.jpg \
  --docker --output exports/ocr-front/lan-1
```

`--docker` chạy một container tạm từ service `namlhn-gpu`, dùng cấu hình và
cache model của Compose. Cần NVIDIA GPU và image đã được build theo README.
Ảnh đầu vào và mã nguồn được mount chỉ đọc; kết quả được ghi vào thư mục xuất.
Container tạm khởi tạo riêng model, nên cần đủ VRAM bên cạnh các worker đang chạy.

Nếu chạy trực tiếp trong môi trường đã có đủ thư viện và model GPU, bỏ
`--docker`. Mặc định dùng VietOCR Seq2Seq; `--model vgg_transformer` chọn
biến thể Transformer. Tham số `--recognizer vietocr` vẫn được hỗ trợ.
Tăng cường ảnh theo `NAMLHN_GPU_TEXT_ENHANCEMENT` (`adaptive`). Muốn giữ riêng
kết quả nhận dạng lượt đầu:

```bash
python3 scripts/export-front-ocr.py front.png --docker --enhancement off
```

Không truyền `--output` thì script tạo thư mục có thời gian trong
`exports/ocr-front/`. Thư mục xuất phải mới hoặc đang trống.

Luồng GPU giữ toàn ảnh và hiệu chỉnh sáng/tương phản/nhiễu khi cần.
Báo cáo thêm `card.png`, `prepared_card.png` để xem toàn ảnh trước/sau.
`--card-enhancement off` tắt hiệu chỉnh này trong một lần export.
Cách tìm/cắt từng dòng chữ vẫn giữ nguyên như namlhn trước đây.
Xem [hành vi ID và tiền xử lý](CARD_PREPROCESSING.vi.md).

Kết quả gồm:

- `index.html`: mở bằng trình duyệt để xem ảnh crop cạnh text; có các lượt đọc
  ban đầu và tăng cường nếu vùng đó được đọc lại.
- `source.png`: ảnh đầu vào đã giải mã theo cùng cách với worker.
- `boxes.png`: ảnh với các vùng chữ đánh số để đối chiếu vị trí.
- `crops/0001.png`, `crops/0001.txt`, …: đúng ảnh crop của lượt được chọn và
  text tương ứng, UTF-8.
- `attempts/`: ảnh crop ban đầu và sau tăng cường của những vùng được đọc lại.
- `text.txt`: toàn bộ dòng có chữ theo thứ tự hiển thị.
- `result.json`: text cuối, text model trả về từng lượt, điểm tin cậy, tọa độ
  bốn góc, lượt được chọn và kết quả pipeline.
- `export.zip`: gói toàn bộ các file trên để tải về.

`rows[].readings.primary.text` trong JSON giữ chuỗi model trả về lượt đầu.
`rows[].text` và các file `.txt` dùng text được chọn, chuẩn hóa Unicode NFC và
bỏ khoảng trắng hai đầu giống pipeline. Vùng nhận dạng ra chuỗi trống vẫn
được xuất; `line_number: null` cho biết vùng này được bỏ qua trong danh sách
chữ trên UI. `number` là số vùng trên `boxes.png`, `line_number` là số dòng
trong danh sách UI.

Chỉ khi yêu cầu xuất thì pipeline mới sao chép crop trong bộ nhớ; API thông
thường giữ hành vi hiện tại. Script tách trường từ OCR, không chạy QR.
