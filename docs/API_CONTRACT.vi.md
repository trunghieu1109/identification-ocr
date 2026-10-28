# API contract: đọc thông tin từ hai mặt căn cước

Đối chiếu với mã nguồn ứng dụng phiên bản **0.9.0**, ngày **28/10/2026**.

Từ 0.9.0, `quality=high` có công tắc `use_mrz`: bật dùng QR/MRZ/OCR song song và căn chỉnh; tắt dùng GPU QR + OCR cũ. Compose mặc định bật qua `GPU_MRZ_ENABLED=1`. Schema 12 trường, `source="qr_ocr"` và trạng thái HTTP được giữ; xem [hướng dẫn bật/tắt](GPU_MRZ_SWITCH.vi.md).

Từ 0.7.0, mức `high` dùng namlhn GPU. Cả hai mức thêm `expiry_date` vào
`fields` và `field_sources`; `total_fields` tăng từ 11 lên 12.

API nhận ảnh mặt trước và mặt sau của cùng một căn cước, trả một JSON chứa
thông tin đọc được của người trên thẻ. Đây là contract hiện tại để hệ thống
tích hợp tiếp tục sử dụng khi thay đổi bộ xử lý bên trong.

Bản 0.8.0 giữ contract API chính, mức CPU `medium` và GPU `high`; chỉ gỡ các API thử nghiệm riêng.

## 1. Giao tiếp HTTP

| Thuộc tính | Contract |
| --- | --- |
| Endpoint | `POST /api/cccd/extract` |
| Base URL mặc định khi chạy Docker Compose | `http://<HOST>:8888` |
| Request | `multipart/form-data` |
| Response | JSON, `Content-Type: application/json` |
| Cách xử lý | Đồng bộ: nhận kết quả trong chính request gửi hai ảnh |
| Xác thực tại ứng dụng hiện tại | Không yêu cầu API key hoặc Bearer token |
| Cache response | `Cache-Control: no-store` |

Mỗi lần gọi độc lập. `request_id` là mã đối chiếu lần xử lý, không phải mã
phiên để truy vấn lại kết quả. Ứng dụng không lưu ảnh hoặc kết quả vào database;
file tạm phục vụ OCR được dọn khi xử lý kết thúc.

## 2. Request

| Form field | Kiểu | Bắt buộc | Ý nghĩa |
| --- | --- | --- | --- |
| `front_image` | File nhị phân | Có | Ảnh mặt trước căn cước |
| `back_image` | File nhị phân | Có | Ảnh mặt sau của cùng căn cước |
| `quality` | string | Không | `medium` hoặc `high`; mặc định theo `OCR_DEFAULT_QUALITY` (Compose: `high`); phân biệt chữ hoa/chữ thường |
| `include_back_ocr` | boolean | Không | Mặc định `false`; `true` trả thêm chữ OCR mặt sau để đối chiếu |
| `use_mrz` | boolean | Không | Chỉ áp dụng `high`: `true` chạy QR/MRZ/OCR song song; `false` chạy GPU QR + OCR cũ; bỏ qua dùng `GPU_MRZ_ENABLED` |

Mỗi ảnh phải không rỗng, có nội dung JPEG hoặc PNG hợp lệ, tối đa
**10 MiB = 10.485.760 byte** và **20.000.000 pixel**. Giới hạn áp dụng riêng
cho từng ảnh. Máy chủ kiểm tra nội dung ảnh và xử lý hướng EXIF.

Gửi đúng mặt vào từng field. Nếu bước OCR nhận diện được nhãn cho thấy gửi
nhầm mặt, API trả HTTP 422. Việc không nhận diện được mặt thẻ không tự động
có nghĩa là hai ảnh đã được xác nhận thuộc cùng người.

`medium` và `high` là hai mức xử lý có cùng schema đầu ra. Cấu hình hiện tại
dùng CPU cho `medium`, GPU cho `high`. Chọn `high` khi GPU chưa sẵn sàng vẫn
có thể nhận kết quả một phần từ QR; hệ thống hiện không tự chuyển sang CPU.

Từ `0.6.4`, Compose đặt `OCR_DEFAULT_QUALITY=high`. Request bỏ qua trường
`quality` dùng GPU cho cả hai mặt và cơ quan cấp; không warm OCR CPU trong
cấu hình này. CPU chỉ chạy khi client chủ động chọn `medium` hoặc dùng API
OCR CPU riêng. Khi chạy ngoài Compose mà không đặt biến, mặc định vẫn là
`medium`; `GET /api/health.default_quality` cho biết cấu hình đang áp dụng.

Ví dụ:

```bash
curl --fail-with-body http://localhost:8888/api/cccd/extract \
  -F 'front_image=@/path/to/front.jpg' \
  -F 'back_image=@/path/to/back.jpg' \
  -F 'quality=medium'
```

Client dùng `FormData` hoặc `curl -F` cần để thư viện tự tạo multipart boundary.
Hai field ảnh nhận file, không nhận URL, đường dẫn trên máy client hoặc chuỗi
base64 thay thế.

## 3. Response HTTP 200

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `success` | boolean | Luôn `true` với kết quả HTTP 200; yêu cầu đã được xử lý |
| `source` | string | `"qr_ocr"` |
| `quality` | string | Mức xử lý được chọn: `medium` hoặc `high` |
| `status` | string | `complete`, `partial` hoặc `no_text` |
| `fields` | object | Luôn có đủ 12 key thông tin tại mục 4 |
| `field_sources` | object | Có cùng 12 key; giá trị là nguồn dữ liệu hoặc `null` |
| `missing_fields` | string[] | Tên các trường chưa đọc được, không tính `old_id_number` và `identifying_features` |
| `recognized_fields` | integer | Số trường có giá trị khác `""`, có tính `old_id_number`; từ 0 đến 11 (key đặc điểm nhân dạng được giữ trống) |
| `total_fields` | integer | Luôn là `12` |
| `expiry` | object | `status`: `found`, `not_found`, `conflict`; `expiry_type`: `date`, `unlimited` hoặc `null`; `label_found`: boolean; `side`: `front`, `back` hoặc `null` |
| `qr` | object | Trạng thái đọc QR, số trường lấy từ QR và trạng thái từng ảnh |
| `ocr` | object | Trạng thái OCR và các trường được bổ sung cho từng mặt |
| `ocr_engine` | object | Metadata bộ xử lý OCR |
| `warnings` | string[] | Thông báo bổ sung; có thể là mảng rỗng |
| `processing_time_ms` | integer | Thời gian pipeline QR/OCR trên server; gồm queue model, không gồm upload/parse multipart hoặc chờ luồng HTTP |
| `request_id` | string | Mã lần gọi do server sinh |
| `timings_ms` | object | Mili giây từng bước: `image_decode`, `qr_front`, `qr_back`, `ocr_front`, `ocr_back`, `merge`, `total` |
| `request_timings_ms` | object | Mili giây xử lý HTTP trên ứng dụng: `receive_upload`, `prepare_request`, `dispatch_wait`, `pipeline`, `finish_result`, `server_until_result` |
| `gpu_pipeline` | object | Khi `high`: `requested`, `used` là `classic/parallel`; `fallback` báo tự dùng GPU cũ khi worker MRZ lỗi kết nối/contract |
| `mrz`, `alignment`, `execution` | object | Khi dùng `parallel`: trạng thái MRZ/checksum và căn chỉnh, chính sách song song; không chứa ảnh hoặc MRZ thô |
| `review_fields` | string[] | Khi dùng `parallel`: các trường cần đối chiếu, có thể khiến `status=partial` dù không còn trường thiếu |
| `stage_intervals_ms` | object | Khi dùng `parallel`: mốc bắt đầu/kết thúc công đoạn để đối chiếu các phần chồng lấp |

Quy ước trạng thái:

- `complete`: đủ 10 trường ngoài `old_id_number` và `identifying_features`.
  Thẻ không thời hạn không bắt buộc có giá trị `expiry_date`.
  `recognized_fields` có thể là **9, 10 hoặc 11**; tổng key là 12.
- `partial`: có ít nhất một trường đọc được và còn thiếu trường bắt buộc,
  hoặc cơ quan cấp mới đọc được một phần (`authority_region.status: "partial"`).
  Trường hợp thứ hai vẫn giữ giá trị và nguồn OCR; `missing_fields` có thể rỗng.
- `no_text`: không có trường nào đọc được, `recognized_fields = 0`;
  `fields` vẫn chứa đủ 12 key với giá trị `""`.

`request_timings_ms.receive_upload` đo đến khi ứng dụng đã nhận hết body;
bao gồm thời gian parse các chunk multipart trong lúc nhận.
`prepare_request` bao gồm xử lý form còn lại và đọc/kiểm tra file;
`dispatch_wait` là chờ luồng HTTP trước khi chạy pipeline. `pipeline`
tương ứng thời gian QR/OCR (có cả queue model); `finish_result` là hoàn tất
kết quả sau pipeline. Tổng các bước này bằng `server_until_result` trong
sai số làm tròn. Tổng này chưa tính serialize JSON, truyền response và UI.
Header `Server-Timing` cung cấp `server` (đến khi phát response headers),
`receive`, `prepare`, `dispatch`, `pipeline` khi đã đo được. Hai nhóm timing
chồng nhau, không cộng tổng HTTP vào tổng pipeline.

Giao diện đo tổng thời gian chờ ở trình duyệt riêng; không ghi số đo client
vào JSON API/file tải xuống. Nhãn số QR/OCR là **số trường**, không phải lượt
nhận dạng. Chỉ đo pipeline có thể thấp hơn đáng kể thời gian người dùng chờ
khi upload lớn, mạng chậm hoặc đang chờ luồng HTTP.

HTTP 200 và `success: true` không đồng nghĩa thông tin đã đầy đủ.
Client kiểm tra `status` và `missing_fields` để quyết định bước tiếp theo.
`complete` mô tả độ đầy đủ của dữ liệu trích xuất, không phải kết luận
xác thực danh tính hoặc tính thật của giấy tờ.

## 4. Các trường thông tin

Tất cả giá trị trong `fields` đều là **string**. Trường chưa đọc được trả
`""`, không bỏ key và không trả `null`. Số giấy tờ là chuỗi để giữ số 0 ở đầu.

| Key | Ý nghĩa |
| --- | --- |
| `id_number` | Số căn cước |
| `old_id_number` | Số CMND cũ; có thể không tồn tại; hiện chỉ lấy từ QR |
| `name` | Họ và tên |
| `date_of_birth` | Ngày sinh |
| `gender` | Giới tính đọc được, ví dụ `Nam`, `Nữ` |
| `nationality` | Quốc tịch |
| `place_of_origin` | Quê quán |
| `place_of_residence` | Nơi thường trú/cư trú |
| `issue_date` | Ngày cấp, dạng `DD/MM/YYYY`; không đọc được trả `""` |
| `issuing_authority` | Tên cơ quan cấp đọc được trên thẻ; từ 0.6.0 bỏ chức danh người ký như Cục trưởng/Giám đốc/KT. |
| `identifying_features` | Đặc điểm nhân dạng; API chung luôn trả `""`, nguồn `null`, không yêu cầu OCR bổ sung |
| `expiry_date` | Ngày hết hạn dạng `DD/MM/YYYY`, đọc từ nhãn và ngày tương ứng trên một trong hai mặt; chưa đọc được hoặc mâu thuẫn trả `""` |

`id_number` từ OCR giữ cả ứng viên sai độ dài/ký tự; không tự cắt về 12 chữ số.
`ocr.front.id_number_validation` phân biệt `format_valid`, `unverified`,
`ambiguous`; kiểm tra định dạng không chứng minh OCR đúng. Metadata
`ocr.<side>.image_preprocessing` mô tả hiệu chỉnh ảnh thực sự đã dùng;
namlhn GPU giữ toàn ảnh và trả `crop.status: "off"`.
Xem [chi tiết hành vi mới](CARD_PREPROCESSING.vi.md).

Ngày hết hạn được lấy trong cùng lượt OCR hai mặt, không gọi thêm lượt OCR riêng.
Không dùng ngày sinh/ngày cấp hoặc một ngày đứng riêng để điền `expiry_date`.
Nếu hai mặt hoặc các lượt đọc cho kết quả khác nhau, `expiry.status: "conflict"`,
nguồn là `null` và trường nằm trong `missing_fields`. Nếu thẻ ghi không thời hạn,
`expiry.expiry_type: "unlimited"`, giá trị ngày là `""` và không tính là thiếu.

`date_of_birth` từ QR và bộ xử lý GPU được chuẩn hóa thành `DD/MM/YYYY`.
Nhánh OCR CPU hiện giữ chuỗi ngày sinh do bộ nhận dạng trả về; client cần
lưu ý điểm này khi phân tích ngày. `gender` là chuỗi, hiện chưa có enum chung
được kiểm tra cho mọi bộ xử lý.

Mỗi giá trị trong `field_sources` thuộc một trong:
`qr_front`, `qr_back`, `ocr_front`, `ocr_back` hoặc `null` ở luồng cũ/CPU. Luồng song song thêm `mrz_back` và tên nguồn chỉ rõ ảnh, như `qr_front_original`, `qr_back_aligned`, `ocr_front_original`, `ocr_back_aligned`, cùng biến thể fallback.
`null` tương ứng trường chưa có dữ liệu. Các key JSON không có ý nghĩa về thứ tự.

## 5. Quy tắc kết hợp dữ liệu

1. Kiểm tra hợp lệ cả hai ảnh trước khi gọi QR/OCR.
2. Đọc QR trên cả hai ảnh. QR trùng nhau được chấp nhận; các giá trị có dữ liệu
   của cùng một trường trong những QR căn cước xung đột nhau làm request bị
   từ chối bằng HTTP 422.
3. Ưu tiên giữ dữ liệu QR hợp lệ; OCR chỉ bổ sung các trường còn trống.
   Không có QR phù hợp thì dùng OCR hai mặt.
4. Nếu một bước xử lý lỗi nhưng đã đọc được thông tin từ QR hoặc mặt còn lại,
   vẫn trả HTTP 200 với dữ liệu giữ được; trường còn thiếu để `""`.
5. Nếu chưa đọc được dữ liệu nào và có lỗi bộ xử lý, trả lỗi HTTP theo mục 8.
   Nếu các bước chạy xong mà không đọc được trường nào, trả HTTP 200,
   `status: "no_text"`.

QR căn cước được hỗ trợ hiện chứa tối đa 7 trường: `id_number`,
`old_id_number`, `name`, `date_of_birth`, `gender`,
`place_of_residence`, `issue_date`. Các trường còn lại cần OCR bổ sung.

## 6. Metadata xử lý

| Trường | Kiểu và giá trị |
| --- | --- |
| `qr.status` | `complete`, `partial`, `no_qr`, `unsupported_qr` hoặc `error` |
| `qr.recognized_fields` | integer; số trường lấy được từ QR sau khi kết hợp, trước OCR |
| `qr.sides.front`, `qr.sides.back` | Trạng thái QR của từng ảnh, dùng cùng tập giá trị với `qr.status` |
| `ocr.front`, `ocr.back` | object; kết quả bước OCR tương ứng |
| `ocr.<side>.status` | `complete`, `partial`, `unstructured`, `no_text`, `skipped` hoặc `error` |
| `ocr.<side>.filled_fields` | string[]; chỉ những key thực sự bổ sung vào kết quả chung |
| `ocr.<side>.error` | string, chỉ xuất hiện khi bước đó lỗi |
| `ocr_engine.provider` | string; tên bộ xử lý được chọn |
| `ocr_engine.device` | string; hiện là `cpu` hoặc `cuda` |
| `ocr_engine.used` | boolean; ít nhất một bước OCR chạy xong mà không lỗi hoặc bị bỏ qua, kể cả không đọc được trường nào |

`unstructured` nghĩa là OCR có văn bản nhưng chưa tách được trường.
`skipped` nghĩa là đã có đủ các trường của mặt đó nên bỏ qua OCR.
`qr.status: "complete"` và `ocr.<side>.status: "complete"` chỉ mô tả bước
tương ứng; trạng thái chung của kết quả luôn nằm ở `status` cấp ngoài cùng.

Hiện cả `medium` và `high` trả `ocr_engine.provider: "namlhn/vietnam-cccd-ocr"`;
thiết bị lần lượt là `cpu` và `cuda`. Các giá trị provider, thời gian,
nội dung thông báo và kết quả đọc là dữ liệu của từng lần xử lý, có thể thay đổi
khi cải thiện bộ xử lý. Client dùng các trạng thái và key đã mô tả để xử lý.

Mức GPU dùng worker namlhn: Paddle tìm dòng và DocU VietOCR Seq2Seq đọc chữ,
`recognition_model: "vgg_seq2seq"`, `strategy: "namlhn_docu_v1"`.
Không gọi luồng CPU để đọc cơ quan cấp.
`GET /api/health` có `ocr_profiles.high.recognition_model`, `strategy` và
`text_enhancement`; response có model và chiến lược thực tế trong `ocr_engine`.
`recognition_provider` và `recognition_device` mô tả bộ nhận dạng thực tế;
detector/crop và ưu tiên QR giữ nguyên.

Các metadata tùy chọn đã có trong implementation:

- Từ `0.6.5`, `include_back_ocr=true` thêm `ocr.back.debug` khi OCR mặt sau
  chạy thành công, kể cả không đọc đủ trường. Gồm `initial_lines` và
  `final_lines`: mỗi dòng có `number`, `text`, `in_authority`; thêm
  `authority_candidate` (chữ ghép trước chuẩn hóa) và `ambiguous`. Hai danh
  sách thuộc cùng lượt/vùng ảnh được chọn trả về, trước/sau đọc bổ sung.
  Giao diện gửi cờ này để hiển thị và sao chép text. Không phát sinh OCR mới,
  không thay đổi ưu tiên QR; request không gửi cờ vẫn không có text debug.
  JSON tải từ giao diện có chứa text này khi đã yêu cầu.
  Từ `0.6.6`, khi GPU đã chạy tìm bổ sung vùng cơ quan cấp, debug có thêm
  `search_crop`: `image` (JPEG data URL, preview tối đa 1000×700), `bbox`
  `[left, top, right, bottom]`, `image_size` của ảnh thẻ đã xử lý. Đây là
  pixel của vùng đã tìm, không phải box của kết quả ghép cuối cùng.
  Request không bật cờ không nhận ảnh/text debug. Vùng tìm mặc định mở
  xuống 6 chiều cao dòng, tối thiểu 16% chiều cao ảnh, rộng tối thiểu 66%
  ảnh thẻ; có thể chỉnh `recovery_crop` trong file quy tắc cơ quan cấp.

- `ocr.back.authority_region`: vị trí khối cơ quan cấp khi engine xác định
  được: `method` (`layout`/`text_order`), `anchor` (`title`/`label`/`institution`),
  `bbox` `[left, top, right, bottom]`, `image_size` `[width, height]`, `line_count`.
  Tọa độ thuộc ảnh thẻ đã crop/xoay (GPU) hoặc ảnh tiền xử lý (CPU), không
  mặc định là ảnh upload. Metadata không chứa chữ OCR hay dữ liệu cá nhân.
  Khi các khối khó phân biệt: `{"status": "ambiguous", "method": "layout"}`;
  giá trị `issuing_authority` được để trống. Cần đối chiếu bằng `missing_fields`.
  Một phần mô tả không có đầu tên cơ quan, hoặc `CÔNG AN TỈNH` thiếu tên tỉnh,
  cũng được để trống. Box có thể bao gồm phần mở rộng crop đã giúp đọc thêm chữ.
  Ngưỡng tương đồng chỉ dùng chuẩn hóa khi có đủ chữ thực sự đọc từ ảnh.

- Từ `0.6.2`, GPU xem tên cụt như `CỤC CẢNH SÁT` hoặc một phần tên cơ quan
  đã biết là chưa đủ. Sau khi mở rộng crop, nếu cần, tìm thêm các dòng trong
  một vùng nhỏ bằng model GPU đang có (tối đa một lượt tìm và 12 crop/lượt).
  Không đọc đủ thì giữ phần `issuing_authority` đã đọc được, gán
  `ocr.back.authority_region.status: "partial"`, trả trạng thái chung `partial`
  và cảnh báo cần đối chiếu. Không xóa phần đã đọc chỉ vì tên chưa đầy đủ.
  Chưa nhận ra được cơ quan nào thì vẫn trả chuỗi trống như trước.
  `ocr.back.enhancement.recovery_attempted` báo đã thử tìm bổ sung;
  `recovered_lines` cho biết số dòng cơ quan được chấp nhận sau phục hồi.
  Công việc này nằm trong `timings_ms.authority_refinement` và được cộng vào
  `work.reread_regions`; không có model CPU chạy thay cho GPU. Luồng CPU và
  ưu tiên QR giữ nguyên.

- `ocr.<side>.timings_ms`: `queue_wait`, `inference`; CPU có `model_init`,
  `preprocess`, `paddle_detect_recognize`, `vietocr_reread`, `metadata`.
  GPU có `worker_image_decode`, `worker_round_trip`, `iqa`, `segmentation`,
  `rotation`, `detection`, `recognition`, `metadata`, `authority_refinement`,
  `recovery` khi áp dụng. Tất cả là số mili giây không âm.
- `ocr.<side>.work`: `primary_regions`, `reread_regions`, `full_image_attempts`;
  CPU có `reread_batches`, GPU có thể có `skipped_duplicate_passes`.
  `full_image_attempts` tính cả lượt pipeline ban đầu, không tính lượt đã tái sử dụng.
- Thời gian OCR cấp ngoài gồm chờ hàng đợi và xử lý; `inference` gồm các công
  đoạn chi tiết; `recovery` cũng gồm detection/recognition phục hồi. Không cộng
  các cấp vào nhau. `processing_time_ms` là `timings_ms.total` làm tròn số nguyên.
- Mỗi engine có FIFO giới hạn, model được tái sử dụng; không cache dữ liệu thẻ.
  Cấu hình `OCR_QUEUE_SIZE` (mặc định 8 ảnh chờ), `OCR_QUEUE_TIMEOUT` (120 giây
  chờ bắt đầu), `OCR_CPU_THREADS` (4). Queue đầy/hết hạn chờ được xử lý như
  OCR bận: trả dữ liệu một phần nếu có, hoặc HTTP 429 nếu chưa có dữ liệu.

- `ocr.<side>.error_code: "multiple_documents"` và `detected_regions`
  (integer hoặc `null`) khi chưa chọn được một vùng thẻ duy nhất.
- `ocr.<side>.recovery` khi bộ xử lý cung cấp thông tin phục hồi, gồm
  `method` (`segmented` hoặc `full_image`), `detected_regions`
  (integer hoặc `null`), `rotation_degrees` (`0`, `90`, `180`, `270`
  hoặc `null`) và `attempted_rotations` (mảng các góc thuộc tập trên,
  không gồm `null`).

## 7. Ví dụ response đầy đủ

Dữ liệu người và thời gian dưới đây là minh họa.

```json
{
  "success": true,
  "source": "qr_ocr",
  "quality": "medium",
  "ocr_engine": {
    "provider": "namlhn/vietnam-cccd-ocr",
    "device": "cpu",
    "used": true
  },
  "status": "complete",
  "fields": {
    "id_number": "001099000001",
    "old_id_number": "012345678",
    "name": "NGUYỄN VĂN THỬ",
    "date_of_birth": "01/02/1999",
    "gender": "Nam",
    "nationality": "Việt Nam",
    "place_of_origin": "Hà Nội",
    "place_of_residence": "Phường Thử, Hà Nội",
    "issue_date": "15/03/2021",
    "issuing_authority": "CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
    "identifying_features": "",
    "expiry_date": "25/10/2036"
  },
  "field_sources": {
    "id_number": "qr_front",
    "old_id_number": "qr_front",
    "name": "qr_front",
    "date_of_birth": "qr_front",
    "gender": "qr_front",
    "nationality": "ocr_front",
    "place_of_origin": "ocr_front",
    "place_of_residence": "qr_front",
    "issue_date": "qr_front",
    "issuing_authority": "ocr_back",
    "identifying_features": null,
    "expiry_date": "ocr_front"
  },
  "missing_fields": [],
  "qr": {
    "status": "complete",
    "recognized_fields": 7,
    "sides": {
      "front": "complete",
      "back": "no_qr"
    }
  },
  "ocr": {
    "front": {
      "status": "complete",
      "filled_fields": ["nationality", "place_of_origin", "expiry_date"]
    },
    "back": {
      "status": "complete",
      "filled_fields": ["issuing_authority"]
    }
  },
  "expiry": {"status": "found", "expiry_type": "date", "label_found": true, "side": "front"},
  "warnings": [],
  "recognized_fields": 11,
  "total_fields": 12,
  "processing_time_ms": 1234,
  "timings_ms": {"image_decode": 40, "qr_front": 15, "qr_back": 5, "ocr_front": 700, "ocr_back": 470, "merge": 4, "total": 1234},
  "request_id": "0123456789abcdef0123456789abcdef"
}
```

Response API này không kèm `raw_text`, `raw_qr` hoặc `qr_codes`.

## 8. Lỗi HTTP

| HTTP | Trường hợp |
| --- | --- |
| `413` | Một trong hai file vượt 10 MiB |
| `422` | Thiếu file, `quality` không hợp lệ, file rỗng/hỏng, định dạng không hỗ trợ, quá giới hạn pixel, nhận diện nhầm mặt hoặc QR xung đột; cũng áp dụng khi không chọn được vùng thẻ duy nhất và chưa đọc được dữ liệu nào |
| `429` | Chưa có dữ liệu và bộ xử lý đang bận; có header `Retry-After: 5`, đơn vị giây |
| `503` | Chưa có dữ liệu và bộ xử lý chưa sẵn sàng hoặc xử lý lỗi |

Lỗi do ứng dụng xử lý có body:

```json
{
  "success": false,
  "error": "Mặt trước: ảnh đang trống. Hãy chọn lại ảnh CCCD.",
  "request_id": "0123456789abcdef0123456789abcdef"
}
```

**Ngoại lệ đang có trong contract:** thiếu form field bắt buộc hoặc giá trị
`quality` không hợp lệ bị FastAPI từ chối trước khi vào handler, trả HTTP 422
với body dạng `{"detail": [...]}`. Mỗi phần tử `detail` có thông tin như
`loc`, `msg`, `type` và có thể có `input`, `ctx`; không bảo đảm có
`success`, `error` hoặc `request_id` trong nhánh này.

Client cần đọc cả hai dạng lỗi hiện có. Nội dung `error`, `warnings` và
`detail[].msg` phục vụ hiển thị, không phải mã cố định để rẽ nhánh.

## 9. Yêu cầu giữ nguyên khi thay bộ xử lý bên trong

- Giữ method, path, tên form field, kiểu upload, tập giá trị `quality` và
  giá trị mặc định để client tiếp tục gửi cùng request.
- Giữ schema, tên key, kiểu dữ liệu, enum trạng thái, ý nghĩa các bộ đếm,
  quy ước `""`/`null` và hai dạng body lỗi hiện tại.
- Giữ quy tắc QR ưu tiên, OCR bổ sung, trả dữ liệu một phần và các mã HTTP.
- Bộ xử lý mới chuyển dữ liệu riêng của nó sang schema này trước khi trả
  response. Các thay đổi model, phát hiện vùng thẻ, xoay/cắt ảnh hoặc cách chạy
  CPU/GPU được triển khai bên trong và tiếp tục đáp ứng các điều kiện trên.
- Kết quả đọc cụ thể, thời gian và metadata bộ xử lý phản ánh lần chạy thực tế.
  Client không cần đổi cách gửi request hoặc đọc response khi cải thiện OCR.

Các kiểm tra HTTP hiện có cho luồng hai ảnh nằm tại
[tests/test_pipeline.py](../tests/test_pipeline.py) và
[tests/test_profiles.py](../tests/test_profiles.py).
