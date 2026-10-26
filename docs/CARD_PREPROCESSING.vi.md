# Giữ số căn cước OCR và tập trung vào vùng thẻ

Luồng GPU chính vẫn dùng namlhn GPU với VietOCR `vgg_seq2seq`. Hai thay đổi:

- Giữ ứng viên số căn cước đọc được ngay cả khi OCR làm thừa/thiếu ký tự.
- Tách thẻ khỏi nền, nắn phối cảnh khi cần, rồi hiệu chỉnh ánh sáng và nhiễu.

## Số căn cước

Trước đây, helper chỉ giữ số đúng 9 hoặc 12 chữ số; các chuỗi 10/11/13 ký tự
bị bỏ nên `id_number` có thể rỗng dù dòng OCR có chữ số. Nay giá trị gắn với
nhãn Số/No. được giữ lại; không tự cắt chuỗi về 12 chữ số, không tự đổi O/I
thành 0/1. Khoảng trắng/dấu phân nhóm số được bỏ như trước.

Ví dụ `Số: 1001099000007` trả `id_number: "1001099000007"`, cùng metadata:

```json
{
  "status": "unverified",
  "format_valid": false,
  "observed_length": 13,
  "ambiguous": false
}
```

Metadata nằm ở `ocr.front.id_number_validation` trong API hai mặt/thử nghiệm;
API một ảnh đặt ở `id_number_validation`. `format_valid` chỉ kiểm tra chuỗi
có 9/12 chữ số, không khẳng định OCR đúng hoặc giấy tờ hợp lệ. Nếu nhiều ứng
viên còn mâu thuẫn, vẫn trả ứng viên quan sát được và ghi `ambiguous: true`.
Không dùng số thuộc nhãn ngày sinh, ngày hết hạn, quê quán, địa chỉ để bù ID.

Nếu provider trả ID rỗng, lớp xử lý chung cũng thử lấy lại từ text của lượt
OCR đó. QR vẫn được ưu tiên; ID OCR chưa xác nhận không ghi đè ID đã có từ QR.

## Crop và tiền xử lý

Mặc định trên worker namlhn GPU:

```text
NAMLHN_GPU_CARD_CROP=auto
NAMLHN_GPU_CARD_ENHANCEMENT=adaptive
```

Model segmentation DocU đã có trong Docker nhận diện vùng thẻ; chỉ bước tách
thẻ được dùng thêm, bộ đọc chữ giữ nguyên. Khi nền chiếm nhiều diện tích,
crop có một ít đệm quanh mép để giữ chữ sát cạnh rồi nắn phối cảnh. Ảnh đã
sát thẻ (thẻ chiếm ít nhất 85% diện tích) giữ nguyên pixel để tránh nội suy
không cần thiết. Không thấy vùng đủ tin cậy hoặc localizer lỗi thì đọc ảnh gốc.
Hai thẻ có diện tích tương đương được báo lỗi `multiple_documents`.

Sau đó đo độ sáng, độ tương phản và nhiễu trên vùng đọc. Chỉ tăng sáng gamma
khi thiếu sáng; dùng CLAHE khi tương phản thấp; giảm nhiễu bằng bilateral
trước khi tăng tương phản. Ảnh sạch/đủ sáng không bị áp dụng các bước này.
Lượt tăng cường từng dòng theo `NAMLHN_GPU_TEXT_ENHANCEMENT` vẫn độc lập.

`ocr.front.image_preprocessing` và `ocr.back.image_preprocessing` báo crop,
polygon, kích thước và các bước thực sự đã áp dụng. Tọa độ dòng chữ, mapping
và vùng cơ quan cấp được quy đổi về ảnh gốc (`original_image`). Vì thế
debug/export tiếp tục đối chiếu được với ảnh upload, kể cả khi có phối cảnh.

Muốn tắt crop và hiệu chỉnh toàn thẻ để đối chiếu pixel với luồng cũ:

```bash
NAMLHN_GPU_CARD_CROP=off NAMLHN_GPU_CARD_ENHANCEMENT=off \
  docker compose --profile gpu up -d --no-deps namlhn-gpu identification-ocr
```

## Xem ảnh trước/sau

```bash
python3 scripts/export-front-ocr.py front.jpg --docker \
  --card-crop auto --card-enhancement adaptive
```

Ngoài ảnh/text từng dòng, báo cáo có `card.png` (vùng thẻ trước hiệu chỉnh)
và `prepared_card.png` (ảnh thực tế dùng đọc chữ). `source.png` luôn là ảnh
upload đã chỉnh EXIF. `--card-crop off --card-enhancement off` tắt hai bước
này trong lần export mà không thay cấu hình worker phục vụ API.

Đã kiểm tra crop nền và ảnh thiếu sáng/nhiễu tạo từ ảnh đã export trước đó;
748 kiểm thử đạt, 6 bỏ qua. Đây là kiểm tra hành vi và hồi quy, không phải
chứng minh độ chính xác OCR trên một tập CCCD đại diện.
