# Namlhn GPU kết hợp bộ đọc DocU

Lựa chọn `namlhn_gpu` hiện dùng `strategy: namlhn_docu_v1`:

1. [Tách vùng thẻ và hiệu chỉnh ảnh có chọn lọc](CARD_PREPROCESSING.vi.md),
   sau đó tiền xử lý namlhn và detector `PP-OCRv6_medium_det` tìm các dòng chữ.
2. Quy đổi box về vùng thẻ đang đọc, cắt bằng
   `VietOCRRecognizer.crop_oriented_box` của DocU, với khoảng đệm trên/dưới/ngang
   0.04/0.01/0.02 như upstream.
3. Nhận dạng mọi dòng hợp lệ bằng checkpoint/config `vgg_seq2seq` của DocU.
   Không nạp recognizer Paddle, không trộn chữ Paddle vào kết quả cuối.
4. Sắp box theo hàng chữ và chiều trái sang phải, có xét độ nghiêng; không
   chia tọa độ Y thành các dải cố định. Các mảnh số căn cước chỉ được ghép
   khi nằm cạnh nhau trên cùng hàng, giữ số 0 đầu và ưu tiên đủ 9 hoặc
   12 chữ số. Số sai định dạng vẫn được trả kèm trạng thái chưa xác nhận.
   Không ghép chữ số từ ngày sinh/ngày hết hạn hay các hàng khác.
   Nếu một box đã có đủ số, giữ riêng bằng chứng đó kể cả khi nhãn `Số/No.`
   bị nhận dạng sai, dính vào số hoặc có box chữ số thừa bên cạnh. Chỉ ghép
   các mảnh còn thiếu; không loại số hoàn chỉnh vì cả hàng có ký tự nhiễu.
5. Với mặt trước có nhãn và box, gán giá trị bằng vị trí quanh nhãn thay vì
   lấy phần tử kế tiếp trong mảng OCR. Xét bên phải/cùng hàng và phía dưới,
   độ thẳng cột, nhãn trường chắn giữa và kiểu dữ liệu. Một box giá trị được
   chọn cho một trường; chỉ sắp chữ trong vùng đã chọn. Parser namlhn giữ
   vai trò dự phòng khi thiếu box/nhãn. Cả CPU và namlhn GPU dùng chung bước này.

Tên, số căn cước, ngày sinh, giới tính, quốc tịch, quê quán và nơi thường trú
được gán độc lập. Nhãn song ngữ và giá trị có thể tách thành nhiều box; tên
hoặc địa chỉ có thể xuống dòng. Ngày phải hợp lệ và gần nhãn ngày sinh;
không lấy ngày hết hạn ở cột bên cạnh. Nhãn ngày sinh chen giữa nhãn tên và
`HOÀNG THÀNH VINH` không làm mất tên. Một box chứa cả hai nhãn như
`Giới tính: Nam Quốc tịch: Việt Nam` được tách theo nhãn.

Khi bật `include_ocr_text=true` trong API thử nghiệm, xem
`ocr.front.debug.field_mapping`: mỗi trường có `label_rows`, `value_rows`,
`bbox`, `method`, `status` và giá trị. Số dòng tham chiếu chính các dòng trong
`debug.text_lines`, giúp đối chiếu box nào đã được chọn. API không bật debug
không trả phần chẩn đoán này. Việc gán trường không sửa chữ/số mà OCR đọc sai.
Nếu chưa ghép được số căn cước theo vị trí nhưng OCR đã có một số hoàn chỉnh
hợp lệ, giữ số đó và ghi `method: observed_number`; không xóa vì thiếu liên
kết với nhãn. Trường ghép bằng vị trí ghi `method: spatial`.

Quê quán và nơi thường trú được nhóm từ nhãn song ngữ cùng các dòng giá
trị lân cận, dùng vị trí khi có box. Nhãn tiếng Anh bị đọc sai như
`Quê quán 1 Place of rrgin` không được dùng làm địa chỉ. Nhãn có thể nằm
ở cùng dòng, nhiều box hoặc nhiều dòng; địa chỉ được ghép tới trước nhãn
trường tiếp theo. Giữ nguyên chữ, dấu câu và số nhà đã đọc được.

Chỉ dùng một bản VietOCR, chạy CUDA/FP32, không gọi worker DocU qua HTTP.
Model segmentation DocU cục bộ tách thẻ khỏi nền trước detector; khi không
crop/hiệu chỉnh, nhận dạng vẫn dùng pixel gốc chưa qua tăng nét detector.
Tọa độ kết quả cuối được quy đổi về ảnh upload. Ngày hết hạn tiếp tục được
ghép từ nhãn và giá trị trên mặt trước, không lấy ngày đứng riêng.

Lượt đọc đầu dùng ảnh sau bước crop/hiệu chỉnh toàn thẻ. Namlhn GPU có thêm
[khử nhiễu/làm nét chữ có chọn lọc](NAMLHN_TEXT_ENHANCEMENT.vi.md), thử đọc
bổ sung các vùng chất lượng thấp rồi đối chiếu với kết quả ban đầu.

## Kết quả kiểm tra ngày 25/10/2026

So sánh HTTP trước/sau trên đúng hai cặp ảnh giả lập của bộ benchmark GPU,
ở phiên bản trước khi thêm crop và hiệu chỉnh toàn thẻ,
QR tắt, mỗi cấu hình một lượt warmup và hai lượt đo mỗi cặp. Chấm đúng
toàn bộ giá trị sau chuẩn hóa Unicode NFC, hoa/thường và khoảng trắng;
**giữ nguyên dấu tiếng Việt**. Chỉ chấm 9 trường bắt buộc chung của API;
ngày hết hạn kiểm tra riêng bằng nhãn và bằng chứng vị trí.

| Cặp ảnh | Namlhn cũ: số trường đúng | Kết hợp: số trường đúng | Trung vị cũ | Trung vị kết hợp |
| --- | ---: | ---: | ---: | ---: |
| `expiry-front` | 8/9 | 9/9 | 808 ms | 650,5 ms |
| `expiry-back` | 8/9 | 9/9 | 807 ms | 610,5 ms |

Kết quả lặp lại giống nhau ở hai lượt. Ở hai cặp này, tên cũ bị gán thành
`Giới Tính/Sex Nam`; bản kết hợp lấy đúng `NGUYỄN VĂN THỬ`. Phép đo này
đổi cả nhận dạng, vùng cắt và cách lấy tên, nên không dùng để kết luận
riêng model Seq2Seq chính xác hơn Transformer.

Kiểm tra trực tiếp GPU thêm hai tình huống: nhãn hết hạn nhỏ và ngày không
có nhãn. Bản kết hợp đọc được nhãn nhỏ cùng `25/10/2036`; tình huống
không có nhãn không tạo ra ngày hết hạn. Tuy nhiên, ở hai ảnh bổ sung này
vẫn đọc `THỬ` thành `THỨ` trong tên. Chưa có ảnh căn cước thực của người
dùng để đánh giá lỗi chính tả; các số liệu trên không phải accuracy vận hành.

- Kiểm thử hồi quy: **554 passed, 6 skipped**. Các URL dịch vụ và mặc định
  `OCR_DEFAULT_QUALITY` của Compose được bỏ khỏi môi trường chạy test để
  các test dùng backend giả lập của chúng.
- Trình duyệt thật: mặc định chọn namlhn GPU, hiển thị **bộ đọc DocU**,
  gửi hai ảnh đến API, nhận đúng tên/ngày, không lỗi JavaScript và không
  tràn trang ở viewport 390 px.
- Readiness worker: `recognition_model: vgg_seq2seq`,
  `strategy: namlhn_docu_v1`, `recognition_provider: docu`,
  `pipeline_ready: true`.

Dữ liệu thô chỉ chứa ảnh/chữ giả lập:

- [HTTP trước thay đổi](benchmarks/namlhn-docu-2026-10-25/legacy-http.json)
- [HTTP bản kết hợp](benchmarks/namlhn-docu-2026-10-25/hybrid-http.json)
- [Kiểm tra GPU, gồm nhãn nhỏ và không có nhãn](benchmarks/namlhn-docu-2026-10-25/hybrid-gpu-smoke.json)

## Chạy và quay lại bản cũ

Compose mặc định `NAMLHN_GPU_PIPELINE=docu`,
`NAMLHN_GPU_MODEL=vgg_seq2seq`. Cập nhật worker và ứng dụng:

```bash
docker compose --profile gpu up -d --build --no-deps namlhn-gpu identification-ocr
```

Tab `/#experiment` mặc định chọn namlhn GPU. API dùng giá trị
`engine=namlhn_gpu` như trước; metadata trả chiến lược thực tế.

Quay lại pipeline cũ và Transformer:

```bash
NAMLHN_GPU_PIPELINE=legacy NAMLHN_GPU_MODEL=vgg_transformer \
  docker compose --profile gpu up -d --no-build --no-deps --force-recreate namlhn-gpu identification-ocr
```

[Script kiểm tra HTTP](../scripts/check-namlhn-docu.py) nhận manifest ảnh
giả lập có nhãn đúng, lưu cả so sánh nguyên văn và so sánh đã chuẩn hóa:

```bash
python scripts/check-namlhn-docu.py \
  --dataset /tmp/gpu-engine-benchmark/dataset/dataset.json \
  --output /tmp/namlhn-docu-results.json --label namlhn_docu_v1 --repeats 2
```
