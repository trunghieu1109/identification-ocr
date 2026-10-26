# Thử nghiệm ngày hết hạn

**Tài liệu API thử nghiệm cũ.** Tab Ngày hết hạn đã được thay bằng
[Thử nghiệm OCR đầy đủ](OCR_EXPERIMENT.vi.md) tại `/#experiment`, có năm
engine và chỉ lấy ngày hết hạn từ mặt trước. Endpoint mô tả bên dưới vẫn
được giữ để tương thích các lần thử trước; không được gọi từ tab mới.

Tab **Ngày hết hạn** tại `/#expiry` dùng API riêng:
`POST /api/cccd/expiry/experiment`. Từ 0.7.0, API chính `/api/cccd/extract` trả
12 trường, có `expiry_date` cho cả CPU và namlhn GPU.

Gửi `multipart/form-data`, gồm `front_image` và `back_image` bắt buộc,
`quality=medium|high` tùy chọn. Mặc định theo `OCR_DEFAULT_QUALITY`, giống
API chính. Mỗi ảnh JPEG/PNG tối đa 10 MiB và 20 triệu pixel. Cả hai ảnh được
kiểm tra trước khi gọi OCR. Mức `high` chỉ dùng GPU; không tự chuyển sang CPU.

```bash
curl --fail-with-body http://localhost:8888/api/cccd/expiry/experiment \
  -F 'front_image=@/path/to/front.jpg' \
  -F 'back_image=@/path/to/back.jpg' \
  -F 'quality=high'
```

Thử nghiệm chạy OCR trên **cả hai mặt**, không phụ thuộc QR hoặc số trường
hiện có. Tìm ngày cạnh nhãn như `Có giá trị đến`, `Ngày hết hạn`,
`Hạn sử dụng`, `Date of expiry`, `Valid until`. Hỗ trợ nhãn và giá trị trên
cùng dòng hoặc các dòng kế tiếp, nhãn Việt/Anh, ngày có dấu `/`, `-`, `.`,
ngày ISO và dạng `ngày … tháng … năm …`. Chỉ nhận ngày lịch hợp lệ và chuẩn
hóa thành `DD/MM/YYYY`. Không lấy một ngày không nhãn, không suy từ ngày
sinh/ngày cấp hoặc MRZ. Kết quả phụ thuộc chữ mà bộ OCR thực sự nhận ra.

Với dòng **Có giá trị đến**, thử nghiệm ghép ngày nằm cùng dòng bên phải
hoặc ngay bên dưới theo tọa độ ô OCR, kể cả khi bộ đọc trả ngày trước nhãn
hoặc xen kẽ các cột khác. Hỗ trợ nhãn/ngày bị tách thành nhiều ô. Khi không
có tọa độ, dùng thứ tự dòng làm fallback. Nhánh CPU đọc lại nhãn và các ô
gần đó bằng VietOCR; tái sử dụng cùng bộ chọn vùng như API chính.

Phần chính của response:

```json
{
  "success": true,
  "experiment": "expiry_date",
  "source": "ocr",
  "quality": "high",
  "status": "complete",
  "fields": {"expiry_date": "25/10/2036"},
  "expiry_type": "date",
  "field_sources": {"expiry_date": "ocr_back"},
  "sources": ["ocr_back"]
}
```

| Trường | Ý nghĩa |
| --- | --- |
| `fields.expiry_date` | Ngày `DD/MM/YYYY`; không xác định được hoặc không thời hạn trả `""` |
| `expiry_type` | `date`, `unlimited` (OCR có chữ không thời hạn), hoặc `null` |
| `field_sources.expiry_date` | `ocr_front`, `ocr_back`, hoặc `null`; khi hai mặt cùng giá trị, lấy nguồn đầu tiên |
| `sources` | Tất cả nguồn đồng ý với giá trị được chọn; rỗng khi chưa chọn được |
| `candidates` | Các kết quả đọc được, gồm `expiry_date`, `expiry_type`, `source`, `reading` (`primary`/`backup`), `method` (`inline`/`layout`/`text_order`) và `evidence` (số dòng, chữ OCR) |
| `ocr.front`, `ocr.back` | `status` (`found`/`not_found`/`error`), `expiry_label_found`, `text_lines` gồm `number`, `text`, `is_expiry` và `bbox` nếu có; có thể có `error` hoặc `backup_text_lines` |
| `message`, `warnings` | Thông báo để đối chiếu |
| `ocr_engine`, `timings_ms`, `processing_time_ms`, `request_id` | Bộ OCR, thời gian xử lý server và mã lần gọi |

| `status` chung | Ý nghĩa |
| --- | --- |
| `complete` | Có một giá trị thống nhất, hoặc có chữ không thời hạn; cả hai lượt OCR chạy xong |
| `not_found` | Hai lượt OCR chạy xong nhưng chưa tìm thấy giá trị phù hợp |
| `conflict` | Có ngày khác nhau hoặc ngày và không thời hạn cùng xuất hiện; giữ các ứng viên, không tự chọn |
| `partial` | Một mặt bị lỗi; giữ thông tin đọc được từ mặt còn lại, có thể vẫn chưa có ngày |

HTTP 200 nghĩa là đã xử lý được ít nhất một mặt. Hai mặt đều lỗi OCR trả
503, hoặc 429 khi có lỗi bận (kèm `Retry-After: 5`). Thiếu ảnh, ảnh rỗng,
ảnh không hợp lệ hoặc `quality` sai trả 422; ảnh quá 10 MiB trả 413.

Tab hiển thị kết quả, nguồn, ứng viên và chữ OCR của cả hai mặt; hỗ trợ sao
chép/tải JSON. JSON bao gồm chữ OCR có thể chứa thông tin cá nhân. Ảnh và
kết quả không được lưu vào database; file tạm được dọn sau xử lý. Response
có `Cache-Control: no-store`. Thời gian chờ trình duyệt hiển thị riêng,
không ghi vào JSON API.
