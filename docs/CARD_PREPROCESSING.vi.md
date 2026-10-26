# Giữ số căn cước OCR và hiệu chỉnh ảnh

Luồng GPU chính vẫn dùng namlhn GPU với VietOCR `vgg_seq2seq`. Hai thay đổi:

- Giữ ứng viên số căn cước đọc được ngay cả khi OCR làm thừa/thiếu ký tự.
- Giảm nhiễu và hiệu chỉnh ánh sáng khi cần, giữ toàn ảnh upload.

Phần tự tìm thẻ/crop nền/nắn phối cảnh được thêm sau đã được gỡ theo yêu cầu.
Cách tìm và cắt từng dòng chữ sẵn có của namlhn/DocU vẫn giữ nguyên.

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

## Hiệu chỉnh ảnh, giữ nguyên vùng đọc

Mặc định trên worker namlhn GPU:

```text
NAMLHN_GPU_CARD_ENHANCEMENT=adaptive
```

Namlhn GPU không nạp model segmentation để tách thẻ. Ảnh upload giữ nguyên
kích thước, khung hình và tọa độ; detector tiếp tục tìm dòng chữ trên toàn ảnh.
Tùy chọn `NAMLHN_GPU_CARD_CROP` và `--card-crop` đã được bỏ.

Sau đó đo độ sáng, độ tương phản và nhiễu trên vùng đọc. Chỉ tăng sáng gamma
khi thiếu sáng; dùng CLAHE khi tương phản thấp; giảm nhiễu bằng bilateral
trước khi tăng tương phản. Ảnh sạch/đủ sáng không bị áp dụng các bước này.
Lượt tăng cường từng dòng theo `NAMLHN_GPU_TEXT_ENHANCEMENT` vẫn độc lập.

`ocr.front.image_preprocessing` và `ocr.back.image_preprocessing` báo
`crop.status: "off"`, kích thước toàn ảnh và các bước thực sự đã áp dụng.
Tọa độ dòng chữ, mapping và vùng cơ quan cấp tiếp tục thuộc ảnh upload
(`original_image`), không có thêm phép nắn hay biến đổi tọa độ thẻ.

Muốn tắt hiệu chỉnh toàn ảnh để đối chiếu pixel với luồng cũ:

```bash
NAMLHN_GPU_CARD_ENHANCEMENT=off \
  docker compose --profile gpu up -d --no-deps namlhn-gpu identification-ocr
```

## Xem ảnh trước/sau

```bash
python3 scripts/export-front-ocr.py front.jpg --docker \
  --card-enhancement adaptive
```

Ngoài ảnh/text từng dòng, báo cáo có `card.png` (toàn ảnh trước hiệu chỉnh)
và `prepared_card.png` (toàn ảnh thực tế dùng đọc chữ). `source.png` luôn là
ảnh upload đã chỉnh EXIF. `--card-enhancement off` tắt hiệu chỉnh trong lần
export mà không thay cấu hình worker phục vụ API. Các báo cáo đã export
trước khi gỡ crop vẫn được giữ nguyên để đối chiếu lịch sử.

Kiểm thử kiểm tra việc giữ toàn ảnh, giữ box/cách cắt dòng chữ và tiếp tục
giảm nhiễu, tăng sáng/tương phản. Đây là kiểm tra hành vi và hồi quy, không
phải chứng minh độ chính xác OCR trên một tập CCCD đại diện.
