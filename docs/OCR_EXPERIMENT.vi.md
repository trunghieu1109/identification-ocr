> **Tài liệu lịch sử.** Các API thử nghiệm, worker DocU độc lập và pipeline legacy đã được gỡ ở bản 0.8.0. Các số liệu được giữ để đối chiếu; dùng [README hiện tại](../README.md) để triển khai.

# Tab Thử nghiệm OCR

Tab **Thử nghiệm OCR** tại `/#experiment` thay tab Ngày hết hạn. Dùng cùng
hai ảnh để chạy riêng từng bộ xử lý hoặc chạy cả năm lần lượt và so sánh
12 trường, nguồn QR/OCR, thời gian và chữ nhận dạng. Link `/#expiry` cũ
được chuyển sang tab mới.

| Giá trị `engine` | Bộ xử lý | Model nhận dạng mặc định |
| --- | --- | --- |
| `namlhn_cpu` | namlhn trên CPU hiện tại | `vgg_transformer` |
| `namlhn_gpu` | Paddle tìm vùng + bộ đọc DocU, parser namlhn | `vgg_seq2seq`, `strategy: namlhn_docu_v1` |
| `docu_gpu` | DocU GPU để đối chiếu | `vgg_seq2seq` |
| `docu_gpu_regions` | DocU GPU · đọc theo vùng | `vgg_seq2seq`, cùng model cho lượt đầu và đọc bổ sung |
| `docu_gpu_v3` | DocU GPU · v3 mở rộng box | `vgg_seq2seq`, nền v2 với vùng đọc ngày rộng hơn |

Từ 0.7.0, API chính cũng chọn namlhn GPU cho mức `high` và trả 12 trường, gồm
`expiry_date`. Tab này tiếp tục phục vụ so sánh các bộ đọc và hiển thị chữ OCR.

UI mặc định chọn **namlhn GPU**. Nút **So sánh DocU gốc, v2, v3**
chạy ba phiên bản lần lượt, giữ kết quả, thời gian server và tổng chờ cạnh nhau.
Bỏ chọn ưu tiên QR để đo khả năng OCR độc lập.

CPU vẫn theo `OCR_MODEL` (mặc định Transformer). GPU có cấu hình riêng:
`NAMLHN_GPU_PIPELINE=docu` và `NAMLHN_GPU_MODEL=vgg_seq2seq`. Bộ đọc lấy
đúng checkpoint/config cục bộ của DocU và cách cắt dòng của upstream;
không gọi worker DocU qua HTTP. Namlhn GPU giữ toàn ảnh, không nạp model
segmentation/crop thẻ; [hiệu chỉnh ánh sáng/nhiễu](CARD_PREPROCESSING.vi.md)
vẫn được giữ lại.
Paddle chỉ nạp detector `PP-OCRv6_medium_det`, không nạp recognizer Paddle.
Detector dùng tiền xử lý namlhn; tọa độ được quy đổi về ảnh upload trước khi
cắt, nên recognizer đọc pixel chưa qua tăng nét detector. Kết quả cuối có
tọa độ ảnh upload. Tất cả vùng chữ hợp lệ
được đọc; kết quả không trộn chữ Paddle. Tên cạnh nhãn và tên ở dòng kế
được gán theo bằng chứng đọc được, không thay bằng dòng giới tính.

`NAMLHN_GPU_TEXT_ENHANCEMENT=adaptive` (mặc định) thử khử nhiễu/làm nét
ở những crop chất lượng thấp sau lượt nhận dạng đầu. Chế độ `off`
chỉ bỏ lượt đọc bổ sung; hiệu chỉnh toàn ảnh có cấu hình riêng.
Xem [phép so sánh và giới hạn](NAMLHN_TEXT_ENHANCEMENT.vi.md).

Health/API trả `strategy: namlhn_docu_v1`; UI hiển thị **bộ đọc DocU**.
Tọa độ chữ của bản này thuộc ảnh gốc đã giải mã/chỉnh EXIF
(`coordinate_space: original_image`). GPU dùng FP32, tắt TF32 và không
tự chuyển sang CPU khi lỗi. Chế độ `NAMLHN_GPU_PIPELINE=legacy` giữ cách
đọc Paddle + VietOCR chọn vùng trước đây. Các số đo đối chiếu thiết bị
và tối ưu Transformer dưới đây thuộc chế độ legacy, không phải bản kết hợp.

Đối chiếu sau khi sửa: checkpoint Transformer CPU/GPU có cùng SHA-256
`380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59`;
cấu hình VietOCR gốc khớp nhau. Ba cặp ảnh tổng hợp (hạn ở mặt trước,
hạn chỉ ở mặt sau, cơ quan cấp nhiều dòng) chạy chỉ OCR cho đầu ra
12 trường khớp giữa CPU/GPU sau chuẩn hóa NFC, hoa/thường và khoảng trắng.
Đây là kiểm tra sự tương đương, không phải tỷ lệ đọc đúng ground truth.
Xem [kết quả đối chiếu thiết bị](benchmarks/gpu-2026-10-25/namlhn-device-parity.json).

API chính `/api/cccd/extract`, `/api/ocr/cccd`, `/api/qr/cccd` giữ nguyên
schema và cách chọn bộ xử lý. API chính hai mặt vẫn trả 11 trường và bỏ
qua đặc điểm nhân dạng như trước. Chỉ API thử nghiệm đọc thêm ngày hết
hạn và bổ sung đặc điểm nhân dạng nếu đọc được.

## API thử nghiệm

`POST /api/cccd/experiment/extract`, `multipart/form-data`:

| Field | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `front_image` | Có | Ảnh mặt trước |
| `back_image` | Có | Ảnh mặt sau của cùng thẻ |
| `engine` | Không | Một trong năm giá trị ở bảng trên; API vẫn mặc định `docu_gpu` |
| `use_qr` | Không | Boolean, mặc định `true`: ưu tiên QR rồi OCR bổ sung; `false` chỉ dùng OCR |
| `include_ocr_text` | Không | Boolean, mặc định `false`; thêm chữ và vị trí OCR hai mặt, bằng chứng ngày hết hạn |

Giới hạn ảnh giống API chính: mỗi ảnh JPEG/PNG tối đa 10 MiB, 20 triệu
pixel; kiểm tra cả hai ảnh trước khi gọi QR/OCR. Hai mặt chạy tuần tự cho
mỗi request. Tab gửi `include_ocr_text=true`; nút chạy cả năm gửi năm request
tuần tự trên cùng hai file. Đổi ảnh hoặc đổi chế độ QR xóa kết quả cũ;
chạy lại một bộ xử lý giữ các kết quả còn lại để đối chiếu.

```bash
curl --fail-with-body http://localhost:8888/api/cccd/experiment/extract \
  -F 'front_image=@/path/to/front.jpg' \
  -F 'back_image=@/path/to/back.jpg' \
  -F 'engine=namlhn_gpu' \
  -F 'use_qr=true' \
  -F 'include_ocr_text=true'
```

`GET /api/cccd/experiment/health` trả `engines` với năm key engine, trạng
thái thư viện/model/worker và tên model. Không nạp CPU chỉ để kiểm tra
health. Health chính vẫn giữ cấu trúc cũ.

## Response

Giữ các nhóm giống API chính: `success`, `source`, `quality`, `status`,
`fields`, `field_sources`, `missing_fields`, `recognized_fields`,
`total_fields`, `qr`, `ocr`, `ocr_engine`, `warnings`, `timings_ms`,
`processing_time_ms`, `request_id`, `request_timings_ms`.
Thêm `experiment: "full_ocr"`, `engine`, `use_qr` và metadata `expiry`.

- `fields`/`field_sources` có đủ 12 key: 11 key của API chính và
  `expiry_date`. Các giá trị field đều là string, chưa đọc được trả `""`.
- `old_id_number` chỉ lấy từ QR như trước. `identifying_features` được
  thử đọc ở mặt sau nhưng không bắt buộc để đạt `complete`.
- `expiry_date` chỉ lấy từ **mặt trước**, theo nhãn Có giá trị đến/Date of
  expiry và ngày bên phải hoặc ngay bên dưới. Không lấy ngày ở mặt sau,
  không suy từ ngày sinh/ngày cấp, không cần chạy thêm lượt OCR toàn ảnh.
- `expiry.expiry_type` là `date`, `unlimited` hoặc `null`;
  `expiry.side` luôn là `front`. `expiry.status` là `found`, `not_found`
  hoặc `conflict`; `label_found` cho biết có nhận ra nhãn hay chưa.
- Với không thời hạn: `fields.expiry_date` giữ `""`,
  `expiry.expiry_type: "unlimited"`; không đưa expiry vào `missing_fields`.
  `recognized_fields` vẫn đếm các field khác rỗng, không cộng metadata này.
- Nếu các dòng mặt trước cho ngày khác nhau, giữ expiry trống, báo
  `conflict`, `partial` và cảnh báo. Khi bật `include_ocr_text`,
  `expiry.candidates` có giá trị, cách ghép và các dòng bằng chứng.
- `complete` yêu cầu đủ 9 trường bắt buộc cũ và ngày hết hạn hoặc metadata
  không thời hạn; không bắt buộc CMND cũ và đặc điểm nhân dạng. Cơ quan
  cấp mới đọc được một phần vẫn làm trạng thái chung `partial`.
- `ocr.<side>.filled_fields` chỉ ghi trường thực sự bổ sung; nguồn QR
  hợp lệ được giữ. Mỗi mặt có timing, khối lượng xử lý và metadata tương
  ứng nếu engine cung cấp. Khi bật text, có `debug.text_lines` gồm
  `number`, `text`, `is_expiry`, `bbox` nếu có; mặt sau có thêm
  `debug.authority`. Tọa độ thuộc ảnh đã xử lý của từng engine.
- `use_qr=false`: `source: "ocr"`, `qr.status` và trạng thái từng mặt
  là `skipped`; không gọi QR reader. Dùng để so sánh khả năng OCR độc lập.

Một bộ xử lý lỗi vẫn giữ QR và dữ liệu mặt còn lại. Khi chưa đọc được gì
và OCR lỗi, trả 503 hoặc 429 khi bận (`Retry-After: 5`). Ảnh sai/mặt sai
hoặc QR xung đột trả 422; quá giới hạn dung lượng trả 413. Mỗi request
độc lập, không tự dùng kết quả của engine khác.

`Server-Timing` và timing HTTP cùng định nghĩa như API chính. Thời gian
model, chờ queue và khởi tạo lần đầu nằm trong thời gian server. Giao diện
đo tổng chờ riêng, không sửa JSON API để thêm timing trình duyệt. File tải
xuống chứa `results` theo từng engine, gồm cả thông báo nếu một engine lỗi.

## DocU v3: mở rộng vùng đọc từ nền v2

`docu_gpu_v3` phát triển từ `docu_gpu_regions` v2, dùng cùng checkpoint
YOLO + VietOCR Seq2Seq, cùng hàng đợi và model đã nạp. Lượt tìm chữ toàn
ảnh giữ kích thước 640 như v2; đã bỏ bước chạy toàn ảnh ở 960 rồi hợp nhất.
Các bước đọc vùng khi thiếu trường và thử hướng ảnh vẫn theo cơ chế v2.

Khi thiếu ngày, vùng tìm bổ sung được mở rộng theo chiều ngang của ảnh
đã xử lý, giữ giới hạn chiều cao quanh nhãn/ngày như v2. Nếu mặt trước
chỉ có một ngày đứng riêng chưa thuộc trường khác, thêm tối đa ba dải
đọc: dải ngang chứa ngày, dải phía trên, dải tiếp theo phía trên. Các dải
này được đọc trực tiếp, không phụ thuộc detector tìm lại được nhãn.

Các vùng phát hiện và dải mở rộng dùng **chung một batch nhận dạng vùng**,
tổng tối đa 12 crop như v2. Không thêm lượt nhận dạng riêng sau bước này.
Chỉ dải bổ sung được cắt bớt lề trống theo pixel để chữ nhỏ không bị thu
nhỏ bởi một khoảng trống dài; không ghép cả khối nhiều dòng vào một ảnh
nhận dạng. Đọc lại đã có đủ nhãn/ngày thì bỏ qua. Không chạy namlhn.

Nếu crop rộng đọc được cả nhãn và ngày trùng ngày đã có, giữ dòng mới làm
bằng chứng thay vì bỏ mất nhãn vì ngày bị xem là trùng. Không tự chèn nhãn,
không thay ngày đã đọc bằng ngày khác, chỉ nhận dòng có confidence >= 0.6.
Ngày hết hạn vẫn chỉ lấy từ mặt trước. Lỗi hoặc không cải thiện giữ kết
quả trước đó. V2 và API chính giữ hành vi cũ.

API công khai vẫn chọn `engine=docu_gpu_v3`; strategy mới là
`docu_expanded_v3`. Nhánh worker nội bộ là `/expanded/api/v1/extract`.
Metadata `ocr.<side>.field_recovery` và thời gian `field_refinement` có
cùng cấu trúc v2. Bật `include_ocr_text=true` để xem
`ocr.<side>.debug.region_reads`: ảnh vùng tìm rộng hơn, các dải yêu cầu
đọc (`expanded_boxes`) và chữ đọc lại. UI hiển thị **Mở rộng box · Ngày hết
hạn** khi các dải bổ sung được dùng. Chế độ mặc định không trả ảnh/chữ debug.

Bản thử vẫn cần đối chiếu ảnh thực tế: ảnh có chữ nhỏ, mờ, lệch nhiều hoặc
nền nhiễu có thể chưa đọc được nhãn. Độ phân giải tìm chữ toàn ảnh không
tăng, nhưng một crop rộng có thể làm thời gian nhận dạng vùng thay đổi;
cần so sánh tổng thời gian trên cùng cặp ảnh với v2.

Kiểm tra GPU trên hai ảnh tổng hợp có nhãn 12px/16px ở xa bên trái ngày:
v2 chỉ đọc được ngày; bản mở rộng ghép được `20/10/2028` với nhãn. Mẫu
không có nhãn vẫn để ngày hết hạn trống. Ba lượt xen kẽ sau warmup trên
mẫu 12px: trung vị v2 1006ms, v3 1224ms. Mẫu ngày xuống dòng và cơ quan
cấp nhiều dòng vẫn giữ kết quả. 549 bài test qua; chưa xác nhận ảnh người
dùng. Xem [kết quả bản mở rộng từ v2](benchmarks/gpu-2026-10-25/docu-expanded-v3-smoke.json).

## DocU v2: đọc theo vùng

`docu_gpu_regions` dùng service `ocr-gpu-regions` riêng và cùng checkpoint
YOLO + VietOCR Seq2Seq như DocU gốc. Không gọi worker namlhn hoặc PaddleOCR.
Từ bản v2, bước đọc lại dùng tiền xử lý tương tự namlhn (resize có giới hạn,
bilateral, CLAHE, sharpen) và chừa lề theo pixel ở cả bốn phía. Detector và
recognizer vẫn là YOLO + Seq2Seq của DocU, không nạp model mới. Giữ các ô
chữ của lượt đầu nếu lượt tìm vùng bổ sung bỏ sót chúng.

Nếu thiếu ngày, worker đọc một vùng quanh nhãn. Nếu mặt trước chỉ có một
ngày đứng riêng chưa thuộc trường khác, đọc cả vùng phía trên/trái để tìm
lại nhãn. Chỉ nhận ngày khi đọc được bằng chứng nhãn phù hợp; không tự
coi ngày đứng riêng là ngày hết hạn. Nhãn Việt/Anh gần nhau và nhãn/ngày
bị tách thành nhiều ô được giữ để ghép theo tọa độ.
Mỗi mặt tối đa một lượt đọc vùng ngày. Ngày hết hạn chỉ
đọc mặt trước; ngày cấp chỉ bổ sung khi được yêu cầu ở mặt sau.

Đọc bổ sung chỉ cập nhật chữ khi ghép được một ngày hợp lệ với nhãn; giữ
nguyên chữ ban đầu khi lỗi, thiếu, xung đột hoặc confidence dưới 0.6.
Không có cả nhãn lẫn ngày ứng viên rõ ràng thì bỏ qua. Thiếu ngày hết hạn riêng lẻ
không kích hoạt các lượt xoay toàn ảnh. Phục hồi cơ quan cấp nhiều dòng
dùng cùng tiền xử lý và cách chừa lề mới trên khối cơ quan. Lượt đọc thất
bại hoặc không cải thiện vẫn giữ kết quả trước đó. Các vùng lặp lại không
được đọc lại; tối đa ba vùng bổ sung trên một ảnh trong bước phục hồi này.

Response có `ocr_engine.strategy: "docu_regions_v2"` và
`ocr.<side>.field_recovery` nếu lượt được chọn đã đánh giá vùng ngày:
`target`, `status`, `reason`, `reread_regions`, `elapsed_ms`, `bbox` khi có.
`ocr.<side>.timings_ms.field_refinement` đo tổng thời gian phục hồi ngày;
đây là một phần thời gian OCR, không cộng lại vào tổng server.
UI hiển thị phiên bản v2, lý do đọc/bỏ qua, thời gian bổ sung và số lượt
toàn ảnh. Khi `include_ocr_text=true`, `ocr.<side>.debug.region_reads` có
ảnh vùng đã đọc và các dòng đọc lại, kể cả khi không ghép được trường.
Chế độ mặc định không trả các ảnh/chữ chẩn đoán này.

Biến môi trường `DOCU_REGIONS_GPU_URL` mặc định là
`http://ocr-gpu-regions:8081`; `DOCU_REGIONS_GPU_API_KEY` mặc định theo
`GPU_OCR_API_KEY`. Worker không mở port host, có queue và cache riêng;
model được giữ trong bộ nhớ. Chạy thêm worker cần thêm bộ nhớ GPU.

```bash
docker compose --profile gpu up -d --build --no-deps ocr-gpu-regions identification-ocr
```

## Worker namlhn GPU

Service Compose `namlhn-gpu`, profile `gpu`, chỉ được gọi qua lựa chọn
thử nghiệm. Không mở port ra host; API nội bộ `/api/v1/extract` và health
`/api/v1/health/ready`. Model được nạp một lần và dùng hàng đợi FIFO riêng.
Worker kiểm tra Paddle lẫn PyTorch có CUDA, thực thi CUDA trước readiness,
dùng checkpoint/config VietOCR có sẵn trong image. CPU không làm fallback.

Mặc định chạy bản kết hợp DocU. Cập nhật riêng worker và ứng dụng:

```bash
docker compose --profile gpu up -d --build --no-deps namlhn-gpu identification-ocr
```

Quay lại Transformer và pipeline cũ:

```bash
NAMLHN_GPU_PIPELINE=legacy NAMLHN_GPU_MODEL=vgg_transformer \
  docker compose --profile gpu up -d --no-build --no-deps --force-recreate namlhn-gpu identification-ocr
```

Các số đếm `recognized_regions`/`recognition_batches` và timing
`paddle_detection`, `line_crops`, `docu_recognition` phân biệt tìm vùng và đọc chữ.
Không bảo đảm cải thiện chính tả trên mọi ảnh; cần đo trên ảnh có nhãn đúng,
giữ dấu tiếng Việt khi chấm và kiểm tra riêng ngày hết hạn.

### Tối ưu giải mã Transformer trên GPU

Trong chế độ `legacy`, `NAMLHN_GPU_FAST_DECODE=1` (mặc định) bật
`decoder: "cuda_batched_v1"` riêng worker namlhn GPU khi chọn Transformer.
Giữ nguyên checkpoint Transformer, FP32 và tắt TF32,
ảnh/vùng đọc và cách chọn vùng; CPU và DocU không dùng bộ giải mã này.

CNN/encoder vẫn nhận từng nhóm ảnh có đúng chiều rộng cũ. Các kết quả
encoder được gom để Transformer giải mã cùng lượt, với mask cho phần đệm
và giới hạn 12 mẫu khi gộp các nhóm. Mỗi nhóm gốc giữ mốc kết thúc EOS và
khoảng tính confidence cũ. Token nằm trên GPU trong vòng lặp; kết quả cuối
mới được chuyển về CPU. Nhóm cùng chiều rộng có sẵn không bị tách nhỏ.

Ba cặp ảnh giả lập, hai vòng xen kẽ sau warmup, đo cùng pipeline trên GPU
(có giải mã ảnh, không HTTP/QR): trung vị từng cặp giảm từ 1491/1566/2100 ms
xuống 783/781/1026 ms. Sáu batch, tổng 36 vùng đọc cho chữ khớp; chênh
confidence lớn nhất khoảng 5.5e-8. Các trường giữ nguyên trên ba cặp.
Đây là kiểm tra tương đương và tốc độ nhỏ, không đánh giá accuracy thực tế.
Xem [số đo chi tiết](benchmarks/gpu-2026-10-25/namlhn-decoder-optimization.json).

Health/API thử nghiệm trả `decoder`; UI hiển thị **batch GPU** cạnh model.
Đặt `NAMLHN_GPU_FAST_DECODE=0` và tạo lại service `namlhn-gpu` để dùng lại
bộ giải mã thư viện. Việc đổi cách batching có thể gây sai số số học nhỏ;
không cam kết đầu ra giống từng bit trên mọi ảnh.

Image thử kế thừa môi trường đã dùng trong
[benchmark GPU](GPU_ENGINE_BENCHMARK.vi.md), gồm Paddle GPU/cu129 và Torch/cu128.
Các khác biệt pin thư viện CUDA trong image benchmark vẫn áp dụng; chỉ
dùng cho lựa chọn thử nghiệm, không thay image của hai engine chính.

```bash
# Nếu máy chưa có base image benchmark, build nó trước.
docker build -f Dockerfile.benchmark-gpu -t identification-ocr-benchmark-gpu:experiment .
docker compose --profile gpu up -d --build
```

Biến môi trường mới: `NAMLHN_GPU_OCR_URL` (mặc định nội bộ
`http://namlhn-gpu:8082`), `NAMLHN_GPU_OCR_API_KEY`,
`NAMLHN_GPU_MODEL` (mặc định `vgg_seq2seq`) và `NAMLHN_GPU_PIPELINE`
(mặc định `docu`, có thể chọn `legacy`). Cache GPU và Paddle của namlhn
tách riêng. Lần đầu cần tải model Paddle nếu chưa có trong cache.

Ảnh không lưu vào database, file tạm được dọn sau xử lý. Request bật text
và JSON tải từ tab có thể chứa thông tin cá nhân đọc được; response dùng
`Cache-Control: no-store`.
