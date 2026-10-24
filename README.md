# identification-ocr — OCR và QR căn cước

Giao diện tiếng Việt để tải ảnh **mặt trước hoặc mặt sau CCCD**, xem trước ảnh
và trích xuất thông tin. Mặt trước: số CCCD, họ tên, ngày sinh, giới tính, quốc
tịch, quê quán, nơi thường trú. Mặt sau: **ngày cấp, cơ quan cấp**, đặc điểm nhân
dạng. Có sao chép từng trường, xem văn bản OCR, xem JSON và tải kết quả JSON.

Mặc định tự nhận diện mặt thẻ từ chữ đọc được. Có thể chọn **Mặt sau — ngày cấp,
cơ quan cấp** trước khi trích xuất. Cơ quan cấp được ghép từ các dòng chữ in trên
thẻ; không điền một cơ quan mặc định khi ảnh không đọc được.

OCR dùng trực tiếp `IDCardExtractor` của
[namlhn/vietnam-cccd-ocr](https://github.com/namlhn/vietnam-cccd-ocr): PaddleOCR
phát hiện chữ, VietOCR nhận dạng tiếng Việt, parser của upstream gán các trường.
Ứng dụng không tạo kết quả giả khi OCR chưa cài đặt hoặc gặp lỗi.

Có thêm chế độ **Đọc mã QR**: tải ảnh mặt thẻ có QR hoặc ảnh cắt riêng vùng QR,
lấy thông tin trực tiếp từ mã, bao gồm ngày cấp, sao chép và tải JSON.
Đọc QR chạy bằng [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp/tree/v2.3.0/wrappers/python),
không cần cài upstream OCR hoặc tải model và không gửi ảnh sang dịch vụ bên ngoài.

## Chạy bằng Docker (CPU)

```bash
docker compose up -d --build
```

Mở **http://HOST_IP:8888** từ máy khác, hoặc **http://localhost:8888** trên máy chủ.
Swagger: http://HOST_IP:8888/docs. Docker bind cổng `8888` trên `0.0.0.0`.

Đổi cổng bằng `OCR_PORT=8889 docker compose up -d --build`. Lần build đầu cần
mạng để tải mã nguồn và thư viện; yêu cầu OCR đầu tiên cần mạng và thêm thời gian
để tải model. Model tải về được cache trong các Docker volume. Model dùng CPU,
không cần NVIDIA GPU. Cần RAM phù hợp với PaddleOCR và PyTorch; nên bắt đầu với
ít nhất 4 GB theo khuyến nghị upstream.

Mặc định dùng upstream commit `2d951ac603a43677572ff39ed6a21bdebe622263`.
Để đổi phiên bản upstream, đặt `OCR_REF` thành commit SHA trước khi build.
Commit đã checkout được ghi tại `/opt/ocr-upstream-revision` trong image.

```bash
docker compose logs -f identification-ocr
docker compose down
```

## Chạy trực tiếp

Cần Git, Python **3.10–3.12**, venv và các thư viện hệ thống OpenCV/Paddle.
Trên Ubuntu/Debian thường cần `libgomp1`, `libglib2.0-0`, `libgl1`.

```bash
bash scripts/setup.sh
.venv/bin/python -m uvicorn identification_demo.web:app --host 0.0.0.0 --port 8888
```

`setup.sh` clone upstream vào `.vendor/vietnam-cccd-ocr`, cài PyTorch CPU và
requirements của upstream cùng dịch vụ. Không commit thư mục `.vendor`, model
hoặc ảnh CCCD vào Git.

Nếu đã có upstream và môi trường Python phù hợp:

```bash
python -m pip install -r requirements.txt
OCR_SOURCE_DIR=/absolute/path/to/vietnam-cccd-ocr \
  python -m uvicorn identification_demo.web:app --host 0.0.0.0 --port 8888
```

`OCR_DEVICE` mặc định `cpu`, `OCR_MODEL` mặc định `vgg_transformer`.
Có thể chọn `vgg_seq2seq` và đặt `OCR_WEIGHTS=/absolute/path/seq2seqocr.pth`
nếu đã có weight tương ứng. Chạy một worker để tránh tải nhiều bản model.

## Đọc mã QR

Trên giao diện, chọn **Đọc mã QR**, tải ảnh JPG/PNG rồi nhấn **Đọc mã QR**.
Giới hạn 10 MiB và 20 megapixel, tự xoay ảnh theo EXIF. Nên chụp rõ toàn bộ mã
và viền xung quanh; có thể cắt riêng vùng QR khi ảnh toàn thẻ khó đọc.

Hỗ trợ tách cấu trúc văn bản 7 thành phần, phân cách bằng `|`:

```text
Số căn cước|Số CMND cũ|Họ tên|Ngày sinh|Giới tính|Nơi cư trú|Ngày cấp
```

Số CMND cũ có thể trống. Ngày dạng `DDMMYYYY` hoặc `DD/MM/YYYY` được kiểm tra
hợp lệ rồi hiển thị `DD/MM/YYYY`. Số giấy tờ được giữ dạng chuỗi để bảo toàn
số 0 đầu. Quốc tịch, quê quán và cơ quan cấp không được suy ra từ QR.
Với mã có cấu trúc khác, ứng dụng trả trạng thái `unsupported_qr`, để các trường
trống. Khi có nhiều mã khác nhau trong ảnh,
ứng dụng yêu cầu cắt riêng một mã và không tự chọn thông tin của một người.

Chỉ dùng QR thì không cần chạy `setup.sh`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn identification_demo.web:app --host 0.0.0.0 --port 8888
```

Nếu dịch vụ đã cài trước đó, cài lại `requirements.txt` và khởi động lại;
với Docker chạy `docker compose up -d --build` để có thư viện đọc QR mới.

## API

```bash
curl --fail-with-body http://localhost:8888/api/qr/cccd \
  -F 'file=@/path/to/cccd-with-qr.jpg'
```

`POST /api/qr/cccd` nhận multipart field `file`. Trả `source: "qr"`,
`fields` gồm `id_number`, `old_id_number`, `name`, `date_of_birth`, `gender`,
`place_of_residence`, `issue_date`; kèm `message`, `recognized_fields`,
`total_fields`, `processing_time_ms`, `request_id`. Ngày cấp luôn có key
`fields.issue_date`, được định dạng `DD/MM/YYYY`; nếu thiếu hoặc không hợp lệ
thì trả chuỗi trống. API, giao diện và JSON tải về không chứa nội dung QR gốc
(`raw_qr`, `qr_codes`, `raw_text`).

- HTTP 200: đã xử lý ảnh, `status` là `complete`, `partial`, `no_qr`,
  `unsupported_qr` hoặc `multiple_qr`. `success: true` nghĩa là yêu cầu xử lý
  thành công; cần kiểm tra `status` để biết có đọc được thông tin hay không.
  `complete` cho phép thiếu số CMND cũ, nên `recognized_fields` có thể là 6/7.
- HTTP 413/422: cùng giới hạn và kiểm tra ảnh như OCR.
- HTTP 429: bộ đọc QR đang bận; có `Retry-After: 5`.
- HTTP 503: chưa cài bộ đọc QR hoặc gặp lỗi xử lý.

QR được đọc trong bộ nhớ, không ghi ảnh đã giải mã ra file. Không lưu ảnh hay
kết quả vào database. API trả `Cache-Control: no-store`.

API OCR:

```bash
curl --fail-with-body http://localhost:8888/api/ocr/cccd \
  -F 'file=@/path/to/cccd-front.jpg'

curl --fail-with-body http://localhost:8888/api/ocr/cccd \
  -F 'file=@/path/to/cccd-back.jpg' -F 'side=back'
```

`POST /api/ocr/cccd` nhận multipart `file` và tùy chọn `side=auto|front|back`
(mặc định `auto`). Mặt trước trả `fields` với các key
`id_number`, `name`, `date_of_birth`, `gender`, `nationality`, `place_of_origin`,
`place_of_residence`. Mặt sau trả `issue_date` (DD/MM/YYYY), `issuing_authority`,
`identifying_features`. Kết quả có `side`, `detected_side`, `raw_text`,
`recognized_fields`, `total_fields`, `processing_time_ms` và `request_id`.

- HTTP 200: OCR chạy xong. `status` là `complete`, `partial`, `unstructured`
  (có văn bản nhưng chưa tách được trường) hoặc `no_text`.
- HTTP 413: ảnh vượt 10 MiB.
- HTTP 422: thiếu ảnh, ảnh trống, hỏng, định dạng không hỗ trợ, quá 20 MP,
  hoặc mặt thẻ đã chọn trái với các nhãn OCR nhận diện được.
- HTTP 429: một ảnh khác đang được xử lý; thử lại sau.
- HTTP 503: chưa khởi tạo được OCR hoặc xử lý gặp lỗi.

`GET /api/health` kiểm tra HTTP service và trả `engine.source_available`,
`engine.model_loaded` và `qr.available`. Trạng thái QR độc lập với OCR.
HTTP service hoạt động không có nghĩa model đã tải xong;
model chỉ khởi tạo khi yêu cầu OCR đầu tiên đến.

Ảnh JPG/PNG được kiểm tra nội dung, xoay đúng EXIF và chuyển RGB trước khi đọc.
File xử lý nằm trong thư mục tạm riêng và được xóa khi yêu cầu kết thúc, kể cả
khi OCR lỗi. Không lưu ảnh hoặc kết quả vào database. Trường không đọc được để
chuỗi trống; kết quả luôn cần đối chiếu với thẻ gốc.

## Kiểm tra

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

Test dùng engine thay thế để kiểm tra hợp đồng HTTP, ảnh lỗi, kết quả thiếu
trường, hướng ảnh EXIF và dọn file tạm; không cần tải model. Để kiểm chứng OCR
thực tế cần chạy dịch vụ đã cài upstream và upload ảnh CCCD thật có quyền sử dụng.

Test QR kiểm tra Unicode tiếng Việt, số 0 đầu, ngày không hợp lệ, CMND cũ trống,
ảnh lỗi, EXIF, không có QR, nhiều QR và cấu trúc chưa hỗ trợ. Test tích hợp dùng
`qrcode` tạo ảnh QR với dữ liệu giả lập rồi đọc bằng bộ giải mã thật, gồm ảnh
PNG, JPEG và ảnh xoay; không cần model OCR hoặc ảnh giấy tờ cá nhân.

Chưa tự phát hiện/crop toàn bộ thẻ; parser dựa vào nhãn chữ và bố cục CCCD.
Dùng ảnh chụp thẳng, rõ chữ, thẻ chiếm phần lớn khung hình. Parser mặt sau ưu
tiên ngày cấp và cơ quan cấp trên CCCD thông thường. Trường chưa đọc được để
trống; văn bản OCR giúp kiểm tra khi ảnh có bố cục hoặc nhãn khác.

Service áp dụng compatibility patch trong `patches/` để xử lý đúng nhãn quốc tịch
song ngữ `Quốc tịch / Nationality` và trả văn bản nhận dạng cho parser mặt sau.
Patch được áp dụng trong Docker và `setup.sh`;
test regression chạy khi môi trường đã cài upstream. Weight VietOCR nằm trong
volume cache; file ảnh upload vẫn dùng thư mục `/tmp` riêng và được dọn sau xử lý.
