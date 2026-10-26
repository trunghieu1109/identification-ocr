# identification-ocr — OCR và QR căn cước

Giao diện tiếng Việt nhận **ảnh mặt trước và mặt sau của cùng một CCCD** trong
một lần gửi. Dịch vụ đọc QR trên cả hai ảnh trước, ưu tiên các giá trị hợp lệ từ
QR, sau đó chạy OCR từng mặt để bổ sung các trường còn thiếu. QR lỗi hoặc không
có cấu trúc hỗ trợ thì chuyển sang OCR. Kết quả có nguồn của từng trường,
sao chép từng giá trị, xem JSON và tải JSON. API mặc định không chứa raw text
hay QR gốc; giao diện có thêm chữ OCR mặt sau khi yêu cầu chế độ đối chiếu.

Thông tin gồm số CCCD, CMND cũ (nếu có), họ tên, ngày sinh, giới tính, quốc tịch,
quê quán, nơi thường trú, **ngày cấp, cơ quan cấp, ngày hết hạn**. Key đặc điểm nhân dạng vẫn
được giữ trong API chung nhưng để trống và không phát sinh OCR bổ sung. Cơ quan
cấp được ghép từ chữ thực tế trên thẻ; không điền một cơ quan mặc định.

Có hai mức OCR, chọn trên giao diện hoặc gửi multipart `quality`:

Từ `0.6.5`, sau khi trích xuất, mục **Chữ OCR mặt sau** trên giao diện hiển
thị từng dòng trước/sau đọc bổ sung, tô các dòng được chọn để ghép cơ quan
cấp và hiển thị chữ ghép cùng giá trị trường cuối cùng. Có nút sao chép chữ
OCR để đối chiếu. Đây là dữ liệu của lượt đọc được chọn trả về, không chạy
thêm OCR. Giao diện gửi `include_back_ocr=true` trong request hiện tại;
API thông thường không gửi cờ này vẫn không trả chữ OCR. Khi bật cờ, JSON
cũng có `ocr.back.debug`.

Từ `0.7.0`, luồng GPU chính dùng **namlhn GPU**, cùng worker với tab thử nghiệm.
Cả CPU và GPU trả **12 trường**, bổ sung `fields.expiry_date` dạng `DD/MM/YYYY`.
Ngày hết hạn phải gắn với nhãn trên mặt trước hoặc mặt sau; không lấy ngày sinh,
ngày cấp hoặc ngày đứng riêng. Không đọc được hay có kết quả mâu thuẫn thì trả
`""`. Thẻ ghi không thời hạn trả `expiry_date: ""`, `expiry.expiry_type: "unlimited"`
và không tính trường này là thiếu. `field_sources.expiry_date` ghi mặt OCR đọc được.

| Mức | Giá trị API | Bộ xử lý OCR |
| --- | --- | --- |
| Trung bình | `medium` | `namlhn/vietnam-cccd-ocr`, CPU |
| Cao (mặc định trong Compose) | `high` | namlhn GPU: Paddle detector + bộ đọc DocU Seq2Seq, CUDA |

Compose đặt `OCR_DEFAULT_QUALITY=high`: request không gửi `quality` và giao
diện mặc định dùng GPU cho cả hai mặt, gồm cơ quan cấp. Không warm model OCR
CPU khi dùng mặc định này; CPU chỉ khởi tạo nếu chủ động chọn `medium` hoặc
gọi API OCR CPU riêng. Chạy ngoài Compose không đặt biến này giữ mặc định
`medium` để tương thích các client cũ.

**Cả hai mức đều đọc QR trước và chỉ dùng OCR để bổ sung trường trống.** Mức
cao dùng detector namlhn và bộ đọc DocU Seq2Seq; cần đánh giá trên ảnh thực tế để so sánh
độ chính xác. Nếu GPU không hoạt động, không tự đổi sang OCR CPU: giữ phần QR
đã đọc và báo lỗi OCR trong kết quả; khi hoàn toàn chưa có dữ liệu thì trả 503.

Mức trung bình dùng trực tiếp `IDCardExtractor` của
[namlhn/vietnam-cccd-ocr](https://github.com/namlhn/vietnam-cccd-ocr): PaddleOCR
phát hiện chữ, VietOCR nhận dạng tiếng Việt, parser của upstream gán các trường.
Ứng dụng không tạo kết quả giả khi OCR chưa cài đặt hoặc gặp lỗi.

Bộ đọc QR mặc định của API chính và API QR riêng dùng **WeChatQRCode**
(OpenCV), cùng model và cách xử lý với phương án WeChatQRCode trong tab thử
nghiệm. ZXing và QRDet + ZXing tiếp tục có sẵn trong tab/API thử nghiệm để
dùng khi cần. Chỉ thay bộ đọc bên trong: endpoint, multipart field, schema
response, mã lỗi và nguồn `qr_front` / `qr_back` giữ nguyên.
Các API QR và OCR riêng vẫn có sẵn để kiểm tra từng bộ xử lý. Đọc QR riêng
không cần model OCR. Toàn bộ xử lý chạy cục bộ trên máy chủ.

## Thử nghiệm QR

Tab **Thử nghiệm QR** (hoặc mở `/#qr`) nhận một ảnh hoặc **Hai mặt** JPG/PNG,
tối đa 10 MB mỗi ảnh. Chọn **Hai mặt** để tải mặt trước và mặt sau, chọn
**Song song** (mặc định) hoặc **Tuần tự**. Mỗi mặt dùng cùng phương án tìm vùng,
xử lý và giải mã; không bỏ qua mặt sau khi mặt trước đọc được QR.
Chạy từng phương án hoặc **Chạy cả 6 phương án** lần lượt trên cùng bộ ảnh:

| Phương án | Xử lý |
| --- | --- |
| ZXing | ZXing-cpp phát hiện và giải mã trực tiếp trên ảnh gốc. |
| WeChatQRCode | Bộ đọc WeChat hiện tại; tìm vùng, tăng độ phân giải khi cần và giải mã. |
| QRDet + ZXing | QRDet tìm vùng, nắn phối cảnh, thêm viền trắng; ZXing giải mã. |
| WeChat + Restormer | QRDet tìm vùng từ ảnh gốc; Restormer khử mất nét, WeChat giải mã ảnh phục hồi. |
| WeChat + SwinIR | QRDet tìm vùng từ ảnh gốc; SwinIR-S tăng độ phân giải x4, WeChat giải mã ảnh phục hồi. |
| WeChat + Restormer + SwinIR | Cùng vùng QR; Restormer trước, SwinIR-S x4 sau, cuối cùng WeChat giải mã. |

**Ba phương án phục hồi chỉ dùng trong tab/API thử nghiệm.** API chính và
API QR riêng tiếp tục dùng WeChatQRCode hiện tại. Luồng song song được bỏ
khỏi danh sách phương án; API thử nghiệm `wechat_parallel` cũ vẫn được giữ để tương thích.

Thử nghiệm hai mặt gọi riêng `POST /api/qr/experiment/pair`, gửi multipart
`front_image`, `back_image`, `method` và `execution=parallel|sequential`
(mặc định `parallel`). Hai bộ đọc/model độc lập được giữ lại cho hai mặt,
giúp chạy đồng thời mà không dùng chung detector. Chế độ tuần tự dùng chính
hai bộ đọc đó để so sánh sau khi đã nạp model. Một mặt lỗi vẫn giữ kết quả
mặt còn lại; hai mặt đều lỗi trả 429/503. Kiểm tra cả hai ảnh trước khi quét.

Response có `sides.front` / `sides.back`: thông tin CCCD, trạng thái, khung QR,
thời gian nạp model, đọc QR và tổng xử lý từng mặt. `schedule_ms` trong từng
mặt ghi thời điểm bắt đầu/kết thúc tính từ lúc chạy hai mặt. `timings_ms`
cấp ngoài có `image_decode`, `queue_wait`, `execution`, `sum_side_processing`
và `total`; tổng thời gian từng mặt có thể chồng nhau khi chạy song song,
không phải thời gian thực tế người dùng chờ. Giao diện đo thêm tổng chờ
trình duyệt và hiển thị kết quả/khung QR của từng mặt. JSON tải xuống giữ đủ
metadata này. QR trùng nhau được gộp ở kết quả tổng; nhiều nội dung khác nhau
trả `multiple_qr`, để trống thông tin tổng và vẫn giữ kết quả riêng từng mặt.
Worker phục hồi GPU có thể xếp hàng, nên chạy song song không bảo đảm nhanh
gấp đôi. Nên chạy lại cùng ảnh sau khi nạp model rồi so sánh hai chế độ.
Tính năng này chỉ nằm trong tab/API thử nghiệm QR.

Ba phép thử mới dùng cùng QRDet small (confidence 0.5). Ảnh có cạnh trên
1600 px được tìm thêm trên bốn vùng chồng lấn rộng/cao 60%, rồi loại các
phát hiện trùng. Cắt vùng từ ảnh gốc bằng bounding box có viền 12%; giữ hình
chữ nhật thay vì nắn trước khi phục hồi để tránh thêm nội suy. Tối đa bốn
vùng có độ tin cậy cao nhất, mỗi vùng giới hạn cạnh 384 px. Nếu bị giới hạn
số vùng, kết quả ghi `restoration.truncated: true` và giao diện hiển thị rõ.
Ảnh sau SwinIR có cạnh tối đa 1536 px. Chỉ giải mã vùng đã phục hồi, không
lấy kết quả ảnh gốc thay thế. Các khung được quy đổi về tọa độ ảnh gốc.

Model dùng trọng số chính thức, không huấn luyện lại:

- Restormer: **single-image defocus deblurring**, khoảng 100 MiB, dùng cho
  mất nét. Không phải checkpoint motion deblurring.
- SwinIR: **SwinIR-S lightweight DIV2K x4**, khoảng 16.4 MiB, bản tăng độ
  phân giải nhẹ. Đây là phép thử trên QR, không phải model đã huấn luyện riêng
  cho QR hay bảo đảm giải mã tốt hơn.

Kiến trúc upstream được pin commit và giữ license trong
`identification_demo/restoration_models/SOURCES.md`. Checkpoint được tải lúc
build GPU image, kiểm tra SHA-256 và lưu ở `/opt/qr-restoration`. Không tải
model lúc nhận ảnh. Có thể tải riêng bằng:

```bash
python scripts/setup-qr-restoration.py --directory /path/to/qr-restoration
```

Đặt `QR_RESTORATION_MODEL_DIR` trên worker nếu đổi đường dẫn. Dịch vụ web dùng
`QR_RESTORATION_URL`, mặc định lấy `GPU_OCR_URL`; cùng Bearer token
`GPU_OCR_API_KEY`. Với Docker Compose:

```bash
docker compose --profile gpu up -d --build
```

Phần phục hồi chạy trên GPU, chia tile chồng lấn (Restormer 256 px, SwinIR
128 px). Dùng chung hàng đợi model với OCR GPU để không chạy hai tác vụ nặng
đồng thời. Model nạp khi dùng lần đầu và dùng lại cho cả ba phép thử. Khi
thiếu GPU/model, lựa chọn đó báo lỗi rõ ràng; ba bộ đọc ban đầu vẫn dùng được.

Giao diện hiển thị ảnh trước/sau phục hồi của vùng đầu tiên, số vùng tìm được
và đã xử lý, kích thước trước/sau, thời gian tìm vùng, phục hồi (gồm gọi GPU),
thời gian từng model, thời gian giải mã, thời gian nạp và VRAM PyTorch tối đa
trong worker. Số VRAM gồm cả các model OCR đang nạp. Các nhóm thời gian nằm
trong nhau, không cộng tổng các cấp. Ảnh preview chỉ giữ trên trang và trong
JSON tải xuống; không lưu ảnh hoặc kết quả vào database.

Mỗi phương án chạy độc lập, không chuyển sang phương án khác và không dùng
OCR. Giao diện giữ các kết quả cạnh nhau, hiển thị vùng QR (xanh: đã giải mã,
vàng: chưa giải mã), số nội dung đã giải mã, các trường CCCD và bảng đối chiếu.
WeChatQRCode chỉ trả khung cho mã đã giải mã được; số khung không phải một
chỉ số detection có thể so sánh trực tiếp giữa các phương án. Mã có cấu trúc khác
CCCD được báo là đã giải mã nhưng không gán vào các trường CCCD.

Thời gian nạp bộ đọc/model, đọc QR, xử lý server và tổng chờ được hiển thị
riêng. Tổng xử lý server đã gồm giải nén ảnh, chờ khóa, nạp model và đọc QR;
không cộng các số này vào nhau. Lần suy luận đầu vẫn có chi phí warmup:
nên chạy lại cùng ảnh trước khi so sánh tốc độ. Model chỉ nạp khi chọn phương
án đó; kết quả so sánh chỉ giữ trong bộ nhớ trình duyệt, có thể sao chép/tải
JSON. Thay ảnh hoặc bỏ ảnh sẽ xóa toàn bộ kết quả thử nghiệm.

Docker và `scripts/setup.sh` cài thư viện, tải sẵn model QR. Với môi trường
đã cài sẵn, bổ sung bằng:

```bash
python -m pip install -r requirements-qr.txt
python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
python scripts/setup-qr-models.py
```

Các biến thể OpenCV dùng chung module `cv2`, vì vậy cần cài bản contrib sau
cùng để giữ WeChatQRCode. Model mặc định nằm ở `.vendor/qr-models`; đặt
`QR_MODEL_DIR` để dùng thư mục khác và truyền cùng đường dẫn qua
`scripts/setup-qr-models.py --directory /path/to/models`. Docker dùng
`/opt/qr-models`. Thiếu model hoặc thư viện sẽ trả lỗi rõ ràng, không tự tải
model trong yêu cầu xử lý ảnh. Ba bộ đọc ban đầu chạy CPU; ba phép thử phục hồi cần GPU.

API: `POST /api/qr/experiment`, multipart `file` và `method` bắt buộc là
`zxing`, `wechat`, `qrdet`, `wechat_restormer`, `wechat_swinir` hoặc `wechat_both`. Kết quả gồm `method`, `status`, `fields`,
`decoded_count`, `regions`, kích thước `image`, `cold_start`, `timings_ms`
(`image_decode`, `initialize`, `detect_decode`), `processing_time_ms` và
`request_id`; không trả QR thô. `GET /api/health` bổ sung `qr_experiments`
để xem thư viện/model có sẵn và bộ đọc đã nạp hay chưa. API và luồng trích
xuất CCCD hiện tại dùng WeChatQRCode; ZXing và QRDet vẫn chạy riêng khi chọn
trong tab/API thử nghiệm. Các phương án phục hồi thêm object `restoration`
gồm `mode`, `device`, `detected_regions`, `processed_regions`, `region_limit`,
`truncated`, `crops`, `timings_ms` và `previews` (ảnh JPEG data URL trước/sau,
hoặc `null` nếu không tìm thấy QR). Mỗi `crops[]` có model, thời gian từng
bước, kích thước, tọa độ, trạng thái giải mã, thời gian nạp/chờ queue và
VRAM. Không trả QR thô. Contract của `/api/cccd/extract` và `/api/qr/cccd` giữ nguyên.

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
python -m pip install -r requirements.txt -r requirements-qr.txt
python -m pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
python scripts/setup-qr-models.py
OCR_SOURCE_DIR=/absolute/path/to/vietnam-cccd-ocr \
  python -m uvicorn identification_demo.web:app --host 0.0.0.0 --port 8888
```

`OCR_DEVICE` mặc định `cpu`, `OCR_MODEL` mặc định `vgg_transformer`.
Có thể chọn `vgg_seq2seq` và đặt `OCR_WEIGHTS=/absolute/path/seq2seqocr.pth`
nếu đã có weight tương ứng. Chạy một worker để tránh tải nhiều bản model.

## Bật mức cao với GPU

```bash
nvidia-smi
docker compose --profile gpu up -d --build
```

Ứng dụng public cổng 8888; luồng chính `high` gọi worker **`namlhn-gpu:8082`**.
Worker dùng detector Paddle `PP-OCRv6_medium_det`, bộ đọc DocU `vgg_seq2seq`
và parser namlhn (`strategy: namlhn_docu_v1`). CPU vẫn dùng worker riêng.

Biến cấu hình cho GPU chính: `NAMLHN_GPU_OCR_URL`, `NAMLHN_GPU_OCR_API_KEY`,
`NAMLHN_GPU_MODEL`, `NAMLHN_GPU_PIPELINE`, `NAMLHN_GPU_TEXT_ENHANCEMENT`.
Để chạy trên máy GPU khác, build `Dockerfile.namlhn-gpu` trên máy đó rồi đặt
`NAMLHN_GPU_OCR_URL=http://GPU_HOST:8082` cho ứng dụng. Endpoint worker là
`GET /api/v1/health/ready` và `POST /api/v1/extract`.

`GET /api/health` trả trạng thái tại `ocr_profiles.high`: provider namlhn,
thiết bị `cuda:0`, `recognition_model: vgg_seq2seq`, `strategy: namlhn_docu_v1`.
Cần NVIDIA driver hoạt động trên host và Docker được cấp quyền GPU.
Các worker hiện dùng chung một GPU; request được xếp hàng theo từng worker.

Worker `ocr-gpu` của DocU, `ocr-gpu-regions` và các biến `GPU_OCR_URL`,
`GPU_OCR_RECOGNITION_MODEL`, `GPU_OCR_AUTHORITY_MODEL` phục vụ lựa chọn DocU
ở tab thử nghiệm. Cấu hình và giới hạn của chúng được mô tả trong
[thử nghiệm OCR](docs/OCR_EXPERIMENT.vi.md).

### Điều chỉnh trường cơ quan cấp và làm rõ ảnh

Để xuất từng **ảnh crop mặt trước cùng text OCR tương ứng**, dùng
[`scripts/export-front-ocr.py`](scripts/export-front-ocr.py). Script tạo báo cáo
HTML, ảnh đánh số vùng, PNG/text từng vùng, JSON và file ZIP; xem
[hướng dẫn xuất crop](docs/OCR_CROP_EXPORT.vi.md).

GPU hiện [tự crop vùng thẻ và hiệu chỉnh ảnh khi cần](docs/CARD_PREPROCESSING.vi.md).
Số căn cước đọc thừa/thiếu ký tự được giữ trong `id_number` kèm trạng thái
chưa xác nhận, thay vì bị xóa vì không đủ 9/12 chữ số. QR vẫn được ưu tiên.

Hai bộ xử lý hiện dùng model OCR và quy tắc tách trường, **không dùng prompt
LLM**. Quy tắc chuẩn hóa cơ quan cấp nằm trong
[`identification_demo/config/issuing_authority_rules.json`](identification_demo/config/issuing_authority_rules.json).
Compose mount cả thư mục cấu hình vào CPU và GPU; sửa file trên host được đọc
lại khi xử lý ảnh tiếp theo, kể cả khi editor thay file bằng thao tác atomic.

- `enabled`: bật/tắt chuẩn hóa tên cơ quan từ văn bản OCR.
- `min_similarity`: độ giống tối thiểu sau khi bỏ dấu, mặc định `0.9`; hỗ trợ
  `0.85` đến `1.0`.
- `authorities[].name`: tên cơ quan chuẩn muốn trả về, không gồm chức danh người ký.
- `authorities[].required_fragments`: các cụm phải thực sự có trong chữ OCR.
  Tất cả cụm đều phải khớp; tên chỉ được chuẩn hóa nếu đủ cụm và đủ độ giống.
- `image_enhancement`: bật/tắt bước đọc lại vùng cơ quan cấp bằng GPU.
- `recovery_crop`: vùng tìm bổ sung cơ quan cấp khi tên chưa đủ. Từ `0.6.6`,
  `min_width_ratio=0.66` (tối thiểu 66% chiều rộng ảnh thẻ), `above_lines=0.75`,
  `below_lines=6`, `min_below_height_ratio=0.16` (phần mở xuống tối thiểu 16%
  chiều cao ảnh). Crop được dịch vào trong ảnh khi gần mép để giữ chiều rộng.
  Có thể sửa các giá trị trong file cấu hình rồi gửi lại ảnh, không restart.
  Model tìm dòng trong vùng rộng rồi đọc riêng từng dòng bằng GPU.

Mục **Chữ OCR mặt sau** có thêm ảnh vùng tìm bổ sung khi bước này chạy, giúp
kiểm tra dòng bên dưới có nằm trong crop hay chưa. API bật `include_back_ocr`
trả ảnh JPEG preview và tọa độ tại `ocr.back.debug.search_crop`. Dòng đầu đã
đọc rõ vẫn được đọc bổ sung nếu toàn tên cơ quan chưa đủ; chỉ bỏ qua đọc lại
theo gợi ý dòng khi khối cơ quan đã đầy đủ. Không thay model GPU hoặc ưu tiên QR.

Không khớp đủ căn cứ thì giữ chữ đọc được, không điền một cơ quan mặc định.
Tên Công an tỉnh/thành và cơ quan khác không tự bị đổi thành Cục Cảnh sát.
Không dùng tên cơ quan chuẩn để điền vào ảnh không đọc được chữ.

Từ bản `0.6.0`, `issuing_authority` trả **tên cơ quan**, bỏ các tiền tố chức
vụ như `CỤC TRƯỞNG`, `PHÓ CỤC TRƯỞNG`, `GIÁM ĐỐC`, `KT.`. Ví dụ khối in:

```text
CỤC TRƯỞNG
CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH
VỀ TRẬT TỰ XÃ HỘI
```

trả `CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI`. Không suy ra tên
cơ quan từ riêng chữ `CỤC TRƯỞNG`, hoặc thay cơ quan địa phương bằng Cục.
Schema các trường và ưu tiên QR được giữ nguyên.

CPU/GPU dùng chung bộ chọn vùng dựa vào tọa độ và nội dung các dòng OCR.
Chức danh hoặc nhãn cơ quan cấp làm mốc; các dòng cùng cột, gần nhau được
nhóm thành khối. Các dòng ngày cấp, vân tay, tên người ký và MRZ được loại
khỏi khối. Dòng ở cột khác không làm đứt tên cơ quan hoặc bị ghép vào tên.
Nếu Paddle làm hỏng chữ ở dòng chức danh, bộ chọn xét thêm dòng ngay trên
trong cùng cột có bằng chứng tên cơ quan để VietOCR đọc bổ sung. Khi chỉ có
text không có box, dùng quy tắc thứ tự dòng làm fallback. Hai khối khác nhau
có điểm tương đương sẽ được xem là chưa xác định; không tự chọn một khối.

Với GPU, vùng cơ quan cấp được cắt rộng theo các dòng trong khối để tránh mất
đầu dòng. Chỉ đọc lại dòng chưa chắc chắn bằng bản crop rộng và bản tăng tương
phản/làm nét, rồi chọn kết quả có căn cứ và điểm nhận dạng tốt hơn. Dòng bị
cắt mất đầu tên cơ quan được mở rộng vùng đọc; phần mô tả như `VỀ TRẬT TỰ XÃ HỘI`
đứng riêng không được coi là tên cơ quan. Nếu detector bỏ sót dòng tỉnh sau
`CÔNG AN TỈNH`, GPU đọc thêm một dải nhỏ bên dưới và chỉ chấp nhận tên địa phương
đọc khớp chính xác danh mục trong checkpoint upstream. Không lấy tên người ký
để điền tỉnh, không tự bổ sung tên tỉnh theo cơ quan thường gặp.

Từ bản `0.6.2`, GPU giữ lại phần tên cơ quan đã đọc được, kể cả khi mới có
`CỤC CẢNH SÁT`; trả `authority_region.status: "partial"` và cảnh báo cần đối
chiếu thay vì xóa giá trị. Kết quả chung là `partial` dù `missing_fields` có
thể rỗng. Việc đọc bổ sung gặp lỗi hoặc yếu hơn không được xóa kết quả ban
đầu. Crop đầu dòng được mở rộng cả khi đã có đầu tên
nhưng thiếu phần tiếp theo; bộ lọc cho phép kết quả dài hơn khi đọc được các
từ gốc và phần bổ sung từ ảnh. Nếu vẫn thiếu, GPU chạy thêm **một lượt** tìm
dòng ở độ phân giải cao trong vùng quanh chức danh/cơ quan và một batch đọc
tối đa 12 vùng chữ. Model được tái sử dụng. Tên đã đầy đủ bỏ qua lượt này.
Không phục hồi thêm được thì giữ phần tên đã đọc được, không điền tên từ cấu
hình. Khi ảnh đã đúng mặt và chỉ còn thiếu phần tên cơ quan cấp, sau
lượt thử ảnh gốc không tiếp tục xoay cả ảnh nhiều lần để đọc cùng trường.
CPU giữ luồng hiện tại; mức GPU không chuyển sang model CPU.

`ocr.back.enhancement.recovery_attempted` cho biết đã chạy lượt tìm bổ sung
này; `recovered_lines` là số dòng trong khối cơ quan được chấp nhận sau phục
hồi. Các lượt đọc thêm được cộng vào `work.reread_regions` và thời gian
`timings_ms.authority_refinement`.

Giữ nguyên ảnh gốc để đọc QR trước; không
dùng mô hình tạo ảnh để bù chi tiết thiếu. Metadata tùy chọn
`ocr.back.enhancement` cho biết số dòng đọc lại, số dòng thay đổi và các bước
làm rõ đã dùng; không trả ảnh hay văn bản OCR gốc.
`ocr.back.authority_region` cung cấp `method`, `anchor`, `bbox` (left, top,
right, bottom), `image_size` và `line_count` khi xác định được vùng. Tọa độ
thuộc ảnh thẻ đã crop/xoay của GPU hoặc ảnh đã tiền xử lý của CPU, **không**
phải mặc định tọa độ ảnh upload. Nếu chưa chọn được giữa các vùng, metadata
có `status: "ambiguous"`. Box theo dòng có thể bao gồm chức danh nằm trên
cùng dòng; giá trị trường luôn loại chức danh.

Chạy trực tiếp có thể đặt `ISSUING_AUTHORITY_RULES_FILE=/path/to/rules.json`
để dùng cấu hình khác. File lỗi định dạng thì giữ văn bản OCR và ghi log,
không làm hỏng toàn bộ yêu cầu.

```bash
docker compose --profile gpu logs -f ocr-gpu
docker compose exec identification-ocr python -c \
  "import httpx; print(httpx.get('http://ocr-gpu:8081/api/v1/health/ready').json())"
```

## Đọc mã QR

Trên giao diện, tải hai ảnh JPG/PNG rồi nhấn **Trích xuất thông tin**; hệ thống
tự đọc QR và bổ sung bằng OCR. Có thể dùng API `/api/qr/cccd` để đọc riêng QR.
Giới hạn 10 MiB và 20 megapixel, tự xoay ảnh theo EXIF. Nên chụp rõ toàn bộ mã
và viền xung quanh; có thể cắt riêng vùng QR khi ảnh toàn thẻ khó đọc.

Hỗ trợ tách cấu trúc văn bản 7 thành phần, phân cách bằng `|`:

```text
Số căn cước|Số CMND cũ|Họ tên|Ngày sinh|Giới tính|Nơi cư trú|Ngày cấp
```

Số CMND cũ có thể trống. Ngày dạng `DDMMYYYY` hoặc `DD/MM/YYYY` được kiểm tra
hợp lệ rồi hiển thị `DD/MM/YYYY`. Số giấy tờ được giữ dạng chuỗi để bảo toàn
số 0 đầu. Quốc tịch, quê quán và cơ quan cấp không được suy ra từ QR.
API QR riêng trả `unsupported_qr` cho cấu trúc khác và `multiple_qr` khi có nhiều
mã khác nhau. API chung bỏ qua cấu trúc chưa hỗ trợ để dùng OCR; nếu các mã QR
CCCD hợp lệ có thông tin xung đột thì trả HTTP 422, không ghép các thông tin đó.

Chỉ dùng QR thì không cần chạy `setup.sh`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn identification_demo.web:app --host 0.0.0.0 --port 8888
```

Nếu dịch vụ đã cài trước đó, cài lại `requirements.txt` và khởi động lại;
với Docker chạy `docker compose up -d --build` để có thư viện đọc QR mới.

## API

Contract cho API nhận hai ảnh mặt trước/mặt sau và trả JSON thông tin căn cước:
[docs/API_CONTRACT.vi.md](docs/API_CONTRACT.vi.md).

### API chung — ưu tiên QR, OCR bổ sung

```bash
curl --fail-with-body http://localhost:8888/api/cccd/extract \
  -F 'front_image=@/path/to/cccd-front.jpg' \
  -F 'back_image=@/path/to/cccd-back.jpg' \
  -F 'quality=high'
```

`POST /api/cccd/extract` yêu cầu hai multipart field `front_image`, `back_image`.
Tùy chọn `quality=medium|high`; mặc định do `OCR_DEFAULT_QUALITY` quyết định
(Compose: `high`; chạy ngoài Compose không đặt biến: `medium`). Giá trị khác
trả HTTP 422.
Mỗi ảnh tối đa 10 MiB, 20 megapixel, JPG/PNG. Cả hai ảnh được kiểm tra hợp lệ
trước khi gọi QR hoặc OCR. Đặt đúng mặt vào từng field; nếu nhãn OCR cho thấy
đã gửi nhầm mặt, API báo HTTP 422.

Quy tắc ghép:

1. Đọc QR trên cả hai ảnh. Các bản sao cùng mã được chấp nhận; các giá trị khác
   nhau của cùng một trường trong QR CCCD sẽ trả HTTP 422 để tránh trộn thẻ.
2. Giữ nguyên mọi trường QR đã đọc hợp lệ. Ngày QR thiếu hoặc không hợp lệ được
   để trống, cho phép OCR bổ sung. Không suy ra cơ quan cấp từ QR.
3. Chạy OCR mặt trước khi thiếu trường mặt trước, OCR mặt sau khi thiếu trường
   mặt sau. Chỉ điền vào trường trống, kể cả khi OCR trả giá trị khác với QR.
   Schema hiện có quốc tịch, quê quán và cơ quan cấp nên
   thông thường vẫn cần OCR cả hai mặt dù QR đầy đủ.
4. OCR một mặt lỗi vẫn giữ kết quả QR và OCR mặt còn lại. Trường chưa đọc được
   trả `""`; `field_sources` tương ứng là `null`.

Kết quả trả `source: "qr_ocr"` và `fields` gồm 12 key:
`id_number`, `old_id_number`, `name`, `date_of_birth`, `gender`, `nationality`,
`place_of_origin`, `place_of_residence`, `issue_date`, `issuing_authority`,
`identifying_features`, `expiry_date`. Ngày cấp và ngày hết hạn có dạng `DD/MM/YYYY`. Số CMND cũ chỉ lấy từ QR
và có thể không tồn tại. `identifying_features` luôn là `""`, nguồn `null` trong
API chung; không đọc bổ sung đặc điểm nhân dạng. Hai key này không bắt buộc
để đạt `complete`, nhưng vẫn có trong schema cố định.

Metadata:

- `quality`, `ocr_engine.provider`, `ocr_engine.device`: mức và bộ xử lý đã
  chọn. `ocr_engine.used` cho biết đã chạy OCR thành công trên ít nhất một mặt.
- `field_sources`: nguồn theo từng key — `qr_front`, `qr_back`, `ocr_front`,
  `ocr_back`, hoặc `null`.
- `missing_fields`: các trường còn thiếu, không tính CMND cũ và đặc điểm nhân dạng.
- `qr`: `status`, `recognized_fields`, trạng thái của từng ảnh tại `sides`.
- `ocr`: trạng thái và `filled_fields` của từng mặt (`front`, `back`).
  Khi lỗi có `error` để biết bộ xử lý chưa sẵn sàng hoặc đang bận.
- `warnings`: thông báo QR fallback hoặc bộ xử lý gặp lỗi, không chứa dữ liệu gốc.
- `expiry`: `status` (`found`, `not_found`, `conflict`), `expiry_type` (`date`, `unlimited` hoặc `null`), `label_found`, `side`.
- `recognized_fields`, `total_fields: 12`, `processing_time_ms`, `request_id`.

HTTP 200 có `status` là `complete`, `partial` hoặc `no_text`. `complete` có thể
đạt 10/12 khi không có CMND cũ và đã bỏ qua đặc điểm nhân dạng (9/12 nếu không thời hạn). Nếu chưa có dữ liệu và bộ xử lý gặp lỗi, trả 503;
nếu đang bận thì 429 với `Retry-After: 5`. Ảnh quá dung lượng trả 413; thiếu ảnh,
ảnh hỏng, sai mặt hoặc QR xung đột trả 422. Không trả `raw_text`, `raw_qr`,
`qr_codes`; không lưu ảnh hoặc kết quả vào database.

### Thời gian, hàng đợi và lượt OCR

Số đo trước/sau và giới hạn của bài thử: [Báo cáo tối ưu](docs/OPTIMIZATION_BENCHMARK.vi.md).

Response API chung có `timings_ms`: `image_decode`, `qr_front`, `qr_back`,
`ocr_front`, `ocr_back`, `merge`, `total` (mili giây). Đây là thời gian server
xử lý sau khi đã parse upload và lấy được luồng chạy pipeline;
`processing_time_ms` giữ nguyên ý nghĩa này để tương thích. Nó **không** phải
thời gian người dùng chờ từ lúc bấm nút. Queue của model đã nằm trong pipeline;
queue luồng HTTP nằm trước pipeline.

API chung bổ sung `request_timings_ms`:
`receive_upload` (server nhận body và parse multipart trong lúc nhận),
`prepare_request` (hoàn tất form, kiểm tra và đọc file), `dispatch_wait`
(chờ luồng HTTP trước khi vào pipeline), `pipeline`, `finish_result`,
`server_until_result` (tổng từ khi ứng dụng nhận request đến khi có kết quả).
Header `Server-Timing` có thời gian ứng dụng đến lúc gửi response headers.
Các số này không đo thời gian trước khi request đến ứng dụng, ví dụ proxy,
kết nối mạng phía người dùng, hoặc thời gian hiển thị trong trình duyệt.

Giao diện hiển thị **Tổng chờ** từ lúc bấm đến khi chuẩn bị hiển thị kết quả,
tách khỏi **QR/OCR** trên server. Bảng chi tiết có gửi ảnh/kết nối, chờ phản hồi,
nhận/đọc JSON, hiển thị và phần thời gian ngoài pipeline. Các khoảng đo client
và server chồng nhau, không cộng các cấp. Tiến độ upload được cập nhật khi gửi
ảnh. Số `N trường từ QR`, `M trường từ OCR GPU/CPU` là số trường thông tin;
không phải số lượt gọi model. Thời gian trình duyệt chỉ nằm trên giao diện,
không thay đổi JSON trả từ API hoặc file JSON xuất ra.

Mỗi `ocr.<side>` có thêm `timings_ms` và `work` khi engine hỗ trợ:

- `queue_wait`: chờ worker; `inference`: toàn bộ xử lý trong worker, gồm khởi
  tạo model nếu chưa warm. Thời gian OCR cấp ngoài đã bao gồm các số này.
- CPU: `model_init`, `preprocess`, `paddle_detect_recognize`, `vietocr_reread`,
  `metadata`. Tái sử dụng kết quả Paddle; VietOCR đọc theo batch các vùng chữ
  tự do cần bổ sung (tên/địa chỉ/cơ quan cấp) và vùng còn thiếu/độ tin cậy thấp.
  Không đọc lại toàn bộ ảnh chỉ vì một trường QR đã có bị OCR thiếu.
- GPU: `worker_image_decode`, `worker_round_trip`, `iqa`, `segmentation`,
  `rotation`, `detection`, `recognition`, `metadata`, `authority_refinement`,
  `recovery` khi có. `detection`/`recognition` cộng cả các lượt phục hồi;
  `recovery` là thời gian bao gồm các bước đó, không cộng lặp vào tổng.
- `work.primary_regions`: số vùng chữ đã nhận dạng ở các lượt chính;
  `reread_regions`: số crop đọc bổ sung, kể cả các biến thể tăng tương phản;
  `full_image_attempts`: lượt pipeline ban đầu cộng số lần thử ảnh gốc;
  CPU có thêm `reread_batches`; GPU có `skipped_duplicate_passes` khi tái sử
  dụng lượt đã đọc trên đúng ảnh gốc thay vì OCR ảnh đó lần nữa.

Mỗi engine có một worker và FIFO riêng, giữ model xuyên suốt. Khi mặc định
là `medium`, CPU khởi tạo model trong nền khi server bật, trên cùng thread
chạy inference. Với mặc định `high`, không khởi động model OCR CPU sẵn. Request
đồng thời xếp hàng; không nhân model theo số request. `OCR_QUEUE_SIZE=8`
quy định số ảnh đang chờ **ngoài** ảnh đang xử lý; `OCR_QUEUE_TIMEOUT=120`
giới hạn thời gian chờ bắt đầu. Đầy hàng đợi hoặc hết thời gian chờ trả lỗi bận;
API chung vẫn giữ dữ liệu từ QR/mặt còn lại nếu có. Khi inference đã bắt đầu,
worker xử lý xong trước khi giải phóng ảnh. `OCR_CPU_THREADS=4` giới hạn luồng
CPU của Torch/Paddle; có thể chỉnh theo máy. Runtime CPU được pin ở
`paddlepaddle==3.2.2` để bật oneDNN với các checkpoint hiện tại; Paddle 3.3.x
được tắt oneDNN tự động vì lỗi PIR đã được ghi nhận ở
[upstream Paddle](https://github.com/PaddlePaddle/Paddle/issues/77340).
Có thể đặt `OCR_ENABLE_MKLDNN=0` để tắt tăng tốc khi cần chẩn đoán. `/api/health` và readiness GPU
có `queue.capacity`, `pending`, `active`, `completed`.

Ảnh được giải mã một lần tại mỗi dịch vụ. Hai mặt dùng chung model nhưng chạy
tuần tự qua worker; QR cả hai ảnh vẫn được đọc trước để kiểm tra xung đột và
xác định trường thiếu. GPU tái sử dụng ảnh gốc khi thử xoay và chỉ thử tiếp khi
các trường **cần bổ sung** chưa đủ. Crop cơ quan cấp đã rõ được bỏ qua; crop
chưa rõ dùng hai biến thể trong một batch. HTTP client/kết nối GPU được tái
sử dụng, readiness cache 2 giây. Không cache ảnh hoặc thông tin cá nhân giữa
các request.

Giao diện có mục **Thời gian từng bước** và số vùng đọc bổ sung. Có thể đo tải
bằng ảnh kiểm thử của bạn; script chỉ in thống kê, không in thông tin căn cước:

```bash
python scripts/benchmark-pipeline.py --url http://localhost:8888 \
  --front /path/to/front.jpg --back /path/to/back.jpg \
  --quality medium --requests 5 --concurrency 2
```

### API QR riêng

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

### API OCR riêng

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
`identifying_features`. Cả hai mặt có thêm `expiry_date` và metadata `expiry`.
Kết quả có `side`, `detected_side`, `raw_text`,
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

## Thử nghiệm OCR đầy đủ

Tab **Thử nghiệm OCR** (`/#experiment`) thay tab Ngày hết hạn, so sánh
**namlhn CPU**, **namlhn GPU**, **DocU GPU gốc**, **DocU v2 theo vùng** và
**DocU v3 mở rộng box** trên cùng hai ảnh. Mặc định chọn namlhn GPU; có thể chạy
từng bộ, so sánh ba phiên bản DocU hoặc chạy cả năm lần lượt, ưu tiên QR
như API chính hoặc chỉ OCR.
API riêng `POST /api/cccd/experiment/extract` trả 12 trường, gồm ngày hết
hạn **chỉ từ mặt trước** và đặc điểm nhân dạng nếu đọc được. Giao diện
đối chiếu giá trị, nguồn, thời gian, chữ OCR và cho tải JSON.
Namlhn GPU mặc định kết hợp Paddle tìm vùng chữ với bộ đọc VietOCR Seq2Seq
của DocU (`strategy: namlhn_docu_v1`). Các dòng được cắt từ ảnh gốc, có
khoảng đệm của DocU, rồi đọc bằng một recognizer dùng chung.
Chữ Paddle không tham gia kết quả cuối. `NAMLHN_GPU_PIPELINE=legacy` cùng
`NAMLHN_GPU_MODEL=vgg_transformer` khôi phục cách đọc cũ.
Worker namlhn GPU và DocU thử nghiệm tách riêng. V2 đọc bổ sung vùng ngày;
v3 phát triển từ v2, mở rộng vùng đọc nhãn/ngày và giữ một kích thước tìm
chữ toàn ảnh. Cả hai
dùng YOLO + Seq2Seq và chia sẻ model/queue trên worker thử nghiệm.
Chi tiết v3 có ảnh vùng đọc rộng hơn cùng chữ nhận dạng để đối chiếu.
API chính cũng dùng namlhn GPU ở mức `high` và trả 12 trường gồm `expiry_date`.
Xem [contract và cách chạy worker thử nghiệm](docs/OCR_EXPERIMENT.vi.md).
Kết quả trước/sau và giới hạn của bản kết hợp nằm trong
[báo cáo namlhn + DocU](docs/NAMLHN_DOCU_COMBINATION.vi.md).
Namlhn GPU còn có [khử nhiễu/làm nét chữ có chọn lọc](docs/NAMLHN_TEXT_ENHANCEMENT.vi.md),
điều khiển bằng `NAMLHN_GPU_TEXT_ENHANCEMENT=adaptive|off`.

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

Test API chung kiểm tra QR ưu tiên OCR, QR trên mỗi mặt, ngày cấp thiếu hoặc
không hợp lệ được OCR bổ sung, QR lỗi chuyển sang OCR, lỗi một mặt vẫn giữ
phần kết quả còn lại, QR xung đột, hai ảnh bắt buộc, dọn ảnh tạm và không trả raw.
Test profile kiểm tra mặc định CPU, lựa chọn GPU, không tự đổi GPU sang CPU,
QR ưu tiên metadata GPU, mapping các key thực tế của DocU, chuẩn hóa ngày ISO,
cơ quan cấp lấy từ chữ in, readiness, HTTP timeout, token và dọn ảnh ở worker.

Chưa tự phát hiện/crop toàn bộ thẻ; parser dựa vào nhãn chữ và bố cục CCCD.
Dùng ảnh chụp thẳng, rõ chữ, thẻ chiếm phần lớn khung hình. Parser mặt sau ưu
tiên ngày cấp và cơ quan cấp trên CCCD thông thường. Trường chưa đọc được để
trống; văn bản OCR giúp kiểm tra khi ảnh có bố cục hoặc nhãn khác.

Service áp dụng compatibility patch trong `patches/` để xử lý đúng nhãn quốc tịch
song ngữ `Quốc tịch / Nationality` và trả văn bản nhận dạng cho parser mặt sau.
Patch được áp dụng trong Docker và `setup.sh`;
test regression chạy khi môi trường đã cài upstream. Weight VietOCR nằm trong
volume cache; file ảnh upload vẫn dùng thư mục `/tmp` riêng và được dọn sau xử lý.
