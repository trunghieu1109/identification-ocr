# Đo pipeline OCR sau tối ưu — 24/10/2026

Đo trên cùng host bằng ảnh CCCD tổng hợp chứa dữ liệu kiểm thử; CPU và RTX 5060 8 GB. Trước/sau dùng cùng file ảnh, cùng checkpoint, cùng API hai mặt. Bản sau: `0.5.0`, được kiểm tra tại cổng 8888. Không dùng thẻ thật trong bài đo này.

Các số dưới là wall time của từng request localhost sau khi đã khởi tạo model, mỗi trường hợp một mẫu; dùng để kiểm tra thay đổi, chưa phải thống kê p50/p95 của tập ảnh thực tế. Mức giảm CPU gồm cả tối ưu runtime Paddle/oneDNN và giảm các lượt VietOCR.

| Trường hợp | Trước (ms) | Sau (ms) | Giảm |
| --- | ---: | ---: | ---: |
| CPU, có QR | 37758.76 | 5437.10 | 85.6% |
| CPU, không QR | 32423.31 | 5479.96 | 83.1% |
| GPU, có QR | 1543.82 | 1108.52 | 28.2% |
| GPU, không QR | 1726.03 | 1347.88 | 21.9% |

Cả sáu request so sánh (gồm lượt đầu) đều giữ kết quả 9/9 trường bắt buộc như trước sau khi chuẩn hóa khoảng trắng và hoa/thường. Ngày cấp từ QR vẫn được ưu tiên. Chưa thể suy ra độ chính xác trên ảnh thực tế từ bộ ảnh tổng hợp này.

## Tách thời gian từng phần khi có QR

Các số là ms phía server, từ request đã warm trong bảng trên. Các hàng cấp ngoài cộng thành tổng; chi tiết worker nằm trong thời gian OCR tương ứng.

| Bước | CPU | GPU |
| --- | ---: | ---: |
| Giải mã ảnh | 30.94 | 32.26 |
| QR mặt trước | 8.20 | 8.14 |
| QR mặt sau | 4.45 | 4.46 |
| OCR mặt trước, gồm chờ queue | 2920.35 | 455.92 |
| OCR mặt sau, gồm chờ queue | 2469.32 | 604.06 |
| Tổng hợp kết quả | 0.18 | 0.17 |
| Tổng pipeline | 5433.44 | 1105.01 |

CPU: PaddleOCR mất 2619,10 ms ở mặt trước và 1946,43 ms ở mặt sau; VietOCR bổ sung lần lượt 281,21 ms và 509,20 ms. GPU: nhận dạng chữ mất 324,28 ms ở mặt trước và 452,35 ms ở mặt sau (đã gồm nhận dạng phục hồi); đọc bổ sung cơ quan cấp mất 43,40 ms. Chi tiết đầy đủ có trong `timings_ms` và `ocr.<side>.timings_ms` của response.

## Các lượt đã giảm

- CPU có QR: Paddle tìm/đọc 19 vùng chữ. Bản trước tiếp tục đọc cả 19 bằng VietOCR; bản sau chỉ đọc lại 3 vùng (1 quê quán, 2 dòng cơ quan cấp) bằng 2 batch. Giảm 84,2% số crop VietOCR trong trường hợp này; không đồng nghĩa giảm 84,2% toàn bộ số lượt nhận dạng.
- CPU không QR: VietOCR đọc bổ sung 6/19 vùng trên cặp ảnh kiểm thử; fallback rộng vẫn được giữ khi không định vị được trường thiếu.
- GPU: bỏ lượt đọc baseline crop lặp cho cơ quan cấp; chỉ dòng chưa rõ cần 2 biến thể trong một batch. Đủ trường cần bổ sung thì dừng, không xoay/thử lại để tìm đặc điểm nhân dạng hoặc ngày cấp QR đã có.
- Thử ảnh gốc khi model cắt/xoay sai vẫn cần thiết: ảnh mặt sau của bài đo GPU còn một lượt phục hồi. Ảnh gốc trùng chính xác với vùng đã OCR sẽ tái sử dụng kết quả; số lượt bỏ được thể hiện trong `work.skipped_duplicate_passes`.
- Hai ảnh được giữ ở dạng pixel trong request; không giải mã và ghi PNG tạm lại ở API chính. HTTP client và model được tái sử dụng, không cache dữ liệu cá nhân giữa request.

## Hàng đợi và xác minh

Chạy đồng thời 3 request CPU và 4 request GPU trên bản đã triển khai. Cả 7 trả HTTP 200, `complete`, không có stage OCR lỗi. Queue chờ từng ảnh; mỗi model chỉ xử lý một ảnh tại một thời điểm. Mức tải này là smoke test hàng đợi, chưa xác định công suất tối đa.

| Engine | Request đồng thời | Wall time từng request (giây, sắp tăng dần) |
| --- | ---: | --- |
| CPU | 3 | 11.68, 14.31, 16.93 |
| GPU | 4 | 2.91, 3.65, 4.40, 5.13 |

- 216 test tự động đã qua trên image đóng gói; `pip check` không phát hiện dependency lỗi.
- Trình duyệt: CPU không QR, GPU có QR, GPU không QR; JSON/download, bảng thời gian, layout mobile và không có lỗi JavaScript.
- Chạy model GPU thật với segmentation bị ép trả 2 vùng trắng sai: phục hồi từ ảnh gốc, lấy được ngày cấp và cơ quan cấp. Kiểm tra bảo vệ nhiều thẻ xung đột vẫn được giữ trong regression tests.
- Test riêng FIFO, queue đầy, hết hạn chờ, lỗi model rồi retry và không truy cập lại ảnh của job đã hủy.

## Đo lại

Script không in trường thông tin cá nhân. Request đầu sau restart có thể gồm chờ warmup; nên chạy một lượt chuẩn bị trước khi thu mẫu.

```bash
python scripts/benchmark-pipeline.py --url http://localhost:8888 \
  --front /path/to/front.jpg --back /path/to/back.jpg \
  --quality medium --requests 5 --concurrency 1
```

Dùng `--quality high` cho GPU và tăng `--concurrency` để đo chờ hàng đợi. `OCR_QUEUE_SIZE` mặc định 8 ảnh chờ mỗi engine, `OCR_QUEUE_TIMEOUT` mặc định 120 giây chờ bắt đầu. Tham khảo [API contract](API_CONTRACT.vi.md) và [README](../README.md) để đọc định nghĩa timing và trạng thái.

## Kiểm tra thời gian HTTP và giao diện — bản 0.5.1

Nhãn trước đây chỉ dùng `processing_time_ms` của pipeline. Bản 0.5.1 đo thêm
HTTP và đo tổng chờ trên trình duyệt, tách số trường QR/OCR thành nhãn riêng.
Thử Chromium với cùng hai ảnh tổng hợp khoảng 151 KiB và GPU đã tải model:

| Điều kiện | Pipeline (ms) | Tổng chờ trình duyệt (ms) | Ngoài pipeline (ms) |
| --- | ---: | ---: | ---: |
| Kết nối bình thường | 1196 | 1218.00 | 22.00 |
| Giả lập upload 30.000 byte/giây, latency 80 ms | 1200 | 6398.10 | 5198.10 |

Thử mạng chậm cho thấy thời gian upload không nằm trong số pipeline. Đây là
giả lập để kiểm chứng cách đo; chưa xác định đường truyền của người dùng thực
tế. `request_timings_ms` giúp tách thời gian server nhận ảnh, chuẩn bị và chờ
luồng HTTP; `Server-Timing` có thể xem trong DevTools. Thời gian chờ model
vẫn nằm trong pipeline. JSON xuất ra giữ nguyên kết quả API, không trộn số đo
trình duyệt.

Đã qua 219 tests, gồm giả lập upload body theo từng chunk chậm và trì hoãn
dispatch; browser kiểm tra upload progress, tổng chờ, số trường, JSON/download
và layout mobile.

## Chọn vùng cơ quan cấp — bản 0.6.0

Chạy lại API sau khi triển khai với model CPU và GPU thật. Ba ảnh mặt sau tổng
hợp gồm: dòng chức danh ghép với cơ quan; chức danh riêng, cơ quan nhiều dòng
có chữ vân tay ở cột bên cạnh; và cơ quan Công an tỉnh có tên tỉnh ở dòng dưới.
Cả hai engine trả đúng tên cơ quan, bỏ chức danh và tên người ký. Trường hợp
GPU cắt đầu dòng Cục hoặc bỏ sót tên tỉnh được phục hồi bằng crop vùng nhỏ.
Không có tên cơ quan mặc định để điền khi đọc thiếu chữ.

| Engine | Ảnh thử | Kết quả cơ quan cấp | Pipeline (ms) |
| --- | --- | --- | ---: |
| high | cccd-back-synthetic.png | CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI | 2469 |
| high | authority-layout-cuc.png | CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI | 1082 |
| high | authority-layout-province.png | CÔNG AN TỈNH HÀ NAM | 977 |
| medium | cccd-back-synthetic.png | CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI | 9745 |
| medium | authority-layout-cuc.png | CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI | 8699 |
| medium | authority-layout-province.png | CÔNG AN TỈNH HÀ NAM | 5614 |

Đây là smoke test chức năng, không phải benchmark độ chính xác trên thẻ thực.
Mỗi request giữ nguyên ngày cấp từ QR, schema 11 key và đặc điểm nhận dạng
được bỏ qua. `ocr.back.authority_region` có tọa độ vùng theo ảnh đã xử lý.

246 tests trên image đóng gói đã qua, gồm ghép khối theo tọa độ khi dòng bị
xáo trộn/có cột xen kẽ, không lấy tiêu đề Bộ Công an hoặc chức danh riêng,
không nhận phần đuôi cơ quan là tên đầy đủ, giữ trống khi hai khối mơ hồ, và
không lấy tên người ký để bổ sung tên tỉnh bị thiếu.

## Phục hồi tên cơ quan bị cụt trên GPU — bản 0.6.1

Luồng GPU trước đây có thể xem riêng `CỤC CẢNH SÁT` là đủ trường và dừng.
Đọc lại chỉ bám các box đã có; bộ lọc tương đồng cũng có thể loại một kết quả
đầy đủ vì chuỗi ban đầu quá ngắn. Bản 0.6.1 mở rộng crop đầu tên còn cụt,
chấp nhận phần chữ bổ sung khi có bằng chứng từ ảnh, và khi cần tìm thêm dòng
trong một vùng nhỏ bằng chính detector/recognizer GPU đang tải. Chỉ thử một
lượt tìm vùng với tối đa 12 crop, một batch, cho mỗi ảnh thẻ đang xét. Tên đã
đầy đủ bỏ qua lượt này; không chuyển sang CPU và không đổi ưu tiên QR.

Đã chạy 7 trường hợp bằng model GPU thật trên 3 ảnh tổng hợp: 3 lượt bình
thường, 3 lượt chủ động bỏ dòng tiếp nối khỏi kết quả nhận dạng ban đầu,
và 1 lượt chỉ giữ `CỤC CẢNH SÁT` trong khối tên cơ quan. Cả 7 trả đúng tên
cơ quan; Công an tỉnh Hà Nam vẫn giữ đúng tỉnh. Riêng ca chỉ còn đầu tên,
GPU đọc được 2 dòng cơ quan từ vùng bổ sung, tổng 7 crop đọc lại (gồm 2 biến
thể trước đó và 5 crop tìm bổ sung); `full_image_attempts: 1`. Thời gian xử
lý riêng ảnh mặt sau đã warm trong lượt thử này khoảng 555 ms, trong đó
`authority_refinement` khoảng 212 ms. Đây không phải thời gian toàn request
hay benchmark trên ảnh CCCD người dùng.

325 tests đã qua, gồm tên mất phần giữa phải báo thiếu, chấp nhận phần bổ
sung từ crop rộng, không ghép tên người ký/cột khác, giữ nguyên ngày cấp,
không nhận kết quả xác suất thấp, trả lại cấu hình model khi lỗi, và dừng
xoay toàn ảnh khi đã đúng mặt và chỉ còn trường cơ quan cấp chưa đọc đủ.

## Sửa việc xóa cơ quan cấp đã đọc được — bản 0.6.2

Bản 0.6.1 có hồi quy: tên chưa đủ như `CỤC CẢNH SÁT` bị đổi thành chuỗi
trống ở bước chuẩn hóa kết quả GPU. Các bài thử ảnh tổng hợp đọc được tên
đầy đủ không phát hiện tác động đó trên ảnh người dùng. Bản 0.6.2 bỏ việc
xóa giá trị: giữ phần đọc được, thêm `authority_region.status: "partial"`,
trả trạng thái chung `partial` và cảnh báo cần đối chiếu. `missing_fields`
vẫn chỉ liệt kê trường trống nên có thể rỗng trong trường hợp này.

Trước bước đọc bổ sung, worker giữ bản kết quả ban đầu; lỗi hoặc kết quả yếu
hơn không được làm mất cơ quan đã đọc được. Khi chọn giữa các crop/lượt
xoay có số trường đầy đủ bằng nhau, ưu tiên lượt còn phần tên cơ quan hơn
lượt chỉ có nhiều dòng chữ không liên quan. Không thay đổi luồng CPU hoặc
ưu tiên dữ liệu QR.

335 tests đã qua, gồm đường HTTP đầy đủ từ adapter GPU qua QR/merge, giữ
tên cụt và nguồn OCR khi không đọc bổ sung được, bảo toàn ngày cấp từ QR,
rollback khi enhancement lỗi/xóa/làm yếu chữ và không chọn kết quả trắng
thay phần cơ quan đã đọc. Bảy bài thử model GPU trên ảnh tổng hợp vẫn qua;
chưa có ảnh người dùng báo lỗi để đối chiếu độ chính xác trực tiếp.

## Bộ đọc riêng cho cơ quan cấp trên GPU — bản 0.6.3

Đối chiếu checkpoint bằng SHA-256 xác nhận file `vgg_transformer.pth`
trong image GPU giống file CPU đang dùng. Trước đây nhánh GPU chỉ dùng
`vgg_seq2seq`. Thử cùng crop tổng hợp, Seq2Seq đọc `TỤC CẢNH SÁT QUẢN LÝ
HÀNH CHÍNH`, Transformer đọc đúng `CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH`.
Đây là khác biệt bộ đọc chữ, không phải bằng chứng GPU kém chính xác hơn
CPU khi dùng cùng model và cùng pixel.

Bản 0.6.3 nạp Transformer một lần trên `cuda:0` và chỉ dùng cho bước đọc
bổ sung cơ quan cấp, gồm crop đã tìm và crop phục hồi vùng nhỏ. Seq2Seq vẫn
đọc chữ thông thường, YOLO vẫn tìm vùng thẻ/vùng chữ. Không gọi Paddle hoặc
luồng CPU. Chạy Transformer cho toàn bộ ảnh trong thử nghiệm làm tăng thời
gian nhận dạng khoảng 1,3–1,7 giây trên mỗi ảnh mặt sau tổng hợp, nên không
chọn cấu hình đó làm mặc định.

Cấu hình cuối đã qua 7 lượt dùng model GPU thật trên 3 ảnh tổng hợp, gồm
bỏ dòng tiếp nối và chỉ giữ đầu tên `CỤC CẢNH SÁT`. Cả 7 trả đúng cơ quan;
ca chỉ còn đầu tên có thời gian inference riêng ảnh mặt sau khoảng 1.505 ms,
trong đó bước cơ quan cấp khoảng 1.198 ms. Đây là chi phí đọc bổ sung bằng
Transformer, cao hơn bộ Seq2Seq trước đó; không phải thời gian toàn request.
343 tests đã qua, gồm chọn đúng bộ đọc riêng cho cơ quan cấp và giữ giá trị
ban đầu khi đọc bổ sung lỗi. Chưa có ảnh người dùng đang gặp lỗi, nên chưa
thể kết luận thay đổi này giải quyết mọi trường hợp đọc thiếu dòng của YOLO.

## Dùng chung bộ đọc GPU cho cơ quan cấp — bản 0.6.4

Theo cấu hình sử dụng mới, GPU dùng `vgg_seq2seq` cho chữ thông thường và
cơ quan cấp, tái sử dụng cùng instance trên `cuda:0`; không nạp Transformer
riêng. Compose đặt `OCR_DEFAULT_QUALITY=high`, giao diện và API không gửi
`quality` mặc định chọn GPU. Không warm OCR CPU trong cấu hình này; mức
`medium` chỉ chạy khi được chọn rõ ràng. QR vẫn được ưu tiên trước OCR.

371 tests đã qua, gồm mặc định GPU không warm/call OCR CPU và giữ QR khi
GPU không sẵn sàng. Bảy lượt model thật trên ảnh tổng hợp đều qua; đã kiểm
tra trực tiếp hai bước dùng cùng instance `vgg_seq2seq` trên `cuda:0`.
Health sau restart báo `default_quality: high`, hai tên bộ đọc đều là
`vgg_seq2seq`, model CPU chưa nạp và queue CPU có 0 lượt xử lý.

## Mở rộng vùng tìm cơ quan cấp — bản 0.6.6

Vùng tìm bổ sung mở xuống 6 chiều cao dòng (tối thiểu 16% chiều cao ảnh),
rộng tối thiểu 66% ảnh thẻ; gần mép thì dịch vùng vào trong ảnh để giữ chiều
rộng. Tham số nằm ở `recovery_crop` trong file quy tắc, có thể sửa mà không
restart. Tính vùng tìm theo box trước khi mở rộng crop từng dòng để tránh
mở rộng lặp. Dòng có điểm cao cũng không bị bỏ qua đọc bổ sung khi cả tên
cơ quan còn thiếu.

Thử nghiệm phát hiện mở rộng crop nhận dạng từng dòng theo toàn vùng tìm
làm chữ nhỏ đi và Seq2Seq đọc sai. Cấu hình cuối dùng vùng rộng để tìm dòng,
giữ crop đọc riêng từng dòng sát chữ (đệm ngang 6%). Không thay model GPU,
không phát sinh luồng OCR CPU. Debug khi yêu cầu có ảnh preview của chính
vùng tìm tại `ocr.back.debug.search_crop`, kèm bbox và kích thước ảnh đã xử lý.

379 tests và 7 lượt model GPU thật trên ảnh tổng hợp đã qua. Ca chỉ còn
đầu tên phục hồi đủ tên cơ quan; một lượt kiểm tra preview riêng xác nhận
vùng `[566, 228, 1754, 781]` trên ảnh `1800×1100` chứa cả hai dòng tên cơ
quan. Chưa có ảnh người dùng đang gặp lỗi để kết luận trực tiếp về ảnh đó.
