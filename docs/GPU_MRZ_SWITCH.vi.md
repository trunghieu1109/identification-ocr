# Bật/tắt MRZ trên luồng GPU chính

Từ 0.9.0, `POST /api/cccd/extract` có hai cách chạy cho `quality=high`:

| Chế độ | Xử lý |
| --- | --- |
| MRZ bật | DocAligner căn chỉnh; QR, MRZ và OCR chạy song song. Ghép QR → MRZ đã kiểm tra → OCR điền phần còn thiếu. |
| MRZ tắt | Chạy lại luồng GPU QR + OCR trước đó. Không gọi worker MRZ. |

Nếu QR và MRZ đều không đọc được, luồng mới vẫn đọc đầy đủ hai mặt bằng OCR. Model OCR/CUDA không thay đổi; việc chọn chế độ nằm ở điều phối request.

## Trên giao diện

Ở tab **Đọc căn cước**, chọn **Cao · GPU**, dùng công tắc **Bật MRZ và căn chỉnh ảnh**:

- Bật: luồng mới.
- Tắt: luồng GPU cũ, có hiệu lực ở lần bấm trích xuất tiếp theo.

Trình duyệt nhớ lựa chọn tại máy hiện tại. Khi chưa có lựa chọn lưu, dùng mặc định server từ `/api/health.gpu_mrz.default_enabled`. Công tắc không ảnh hưởng CPU `medium` hoặc tab thử nghiệm. Không cần rebuild, restart hoặc tải lại model khi đổi bằng giao diện.

## Qua API

```bash
# Luồng mới
curl --fail-with-body http://localhost:8888/api/cccd/extract \
  -F 'front_image=@/path/front.jpg' -F 'back_image=@/path/back.jpg' \
  -F 'quality=high' -F 'use_mrz=true'

# Luồng GPU cũ
curl --fail-with-body http://localhost:8888/api/cccd/extract \
  -F 'front_image=@/path/front.jpg' -F 'back_image=@/path/back.jpg' \
  -F 'quality=high' -F 'use_mrz=false'
```

Bỏ `use_mrz` để dùng mặc định server. Nếu `quality=medium`, `use_mrz` không thay đổi xử lý CPU.

## Mặc định server

Compose mặc định `GPU_MRZ_ENABLED=1`; profile `gpu` khởi chạy cả worker `cccd-experiment` dùng CPU và worker `namlhn-gpu` hiện có.

Đặt `GPU_MRZ_ENABLED=0` trong `.env` nếu muốn mặc định dùng luồng cũ lâu dài, rồi áp dụng:

```bash
docker compose up -d --no-build --no-deps identification-ocr
```

Hoặc đổi nhanh mặc định của container đang chạy:

```bash
# Mặc định dùng luồng cũ
GPU_MRZ_ENABLED=0 docker compose up -d --no-build --no-deps identification-ocr

# Mặc định dùng luồng mới và khởi chạy worker MRZ nếu chưa chạy
GPU_MRZ_ENABLED=1 docker compose --profile gpu up -d --no-build identification-ocr cccd-experiment
```

Biến đặt trước lệnh chỉ áp dụng lần gọi Compose đó; muốn giữ qua những lần triển khai sau thì đặt trong `.env`. `use_mrz` trên từng request và lựa chọn trình duyệt vẫn có thể ghi đè mặc định này. Chạy Python trực tiếp không đặt biến thì mặc định tắt để tương thích cài đặt cũ.

## Kết quả và fallback

Schema 12 trường, `source="qr_ocr"`, ba trạng thái `complete/partial/no_text` và cách gọi endpoint được giữ. `identifying_features` vẫn giữ trống theo contract cũ. Bổ sung:

```json
{"gpu_pipeline": {"requested": "parallel", "used": "parallel", "fallback": false}}
```

- `used="parallel"`: luồng QR/MRZ/OCR mới.
- `requested="classic", used="classic"`: đã chọn luồng cũ.
- `requested="parallel", used="classic", fallback=true`: worker MRZ không kết nối được hoặc không trả contract hợp lệ; request được đọc lại bằng GPU QR + OCR cũ, kèm cảnh báo. Không chuyển sang CPU.
- Hàng đợi MRZ bận trả 429; ảnh sai/xung đột căn cước trả 422. Những lỗi này không tự chạy một pipeline khác để tránh che lỗi hoặc tăng tải.

Nguồn từng trường có thể là `qr_front_original`, `qr_back_aligned`, `mrz_back`, `ocr_front_original`, `ocr_back_aligned` hoặc biến thể dự phòng. Khi tắt MRZ giữ tên nguồn cũ `qr_front/qr_back/ocr_front/ocr_back`.

API chính không gửi ảnh preview, QR/MRZ thô, danh sách ứng viên hay chữ OCR toàn ảnh. `include_back_ocr=true` vẫn cho phép chữ mặt sau để đối chiếu như trước; mặc định `false`. MRZ chỉ trả trạng thái, checksum, số lượt và các trường được sử dụng. Các trường cần kiểm tra lại được ghi trong `review_fields`, `status=partial` và `warnings`.

Tab thử nghiệm giữ chi tiết ảnh/chữ, độc lập với công tắc chính. Profile `gpu` và profile `experiment` dùng chung một worker MRZ; không nhân bản mô hình.

## Kiểm chứng triển khai

- 523 test qua, gồm bật/tắt mặc định server, ghi đè từng request, CPU không dùng MRZ, fallback kết nối lỗi và không fallback khi ảnh sai/hàng đợi bận.
- HTTP trên model thật: mặc định bật MRZ, tắt không tăng số job của worker MRZ; hai cặp ảnh ở chế độ tắt khớp toàn bộ 12 trường baseline trước đó.
- Làm mờ cả QR/MRZ: các trường đọc được vẫn do OCR hai mặt bổ sung; không mất đường OCR đầy đủ.
- Chrome: công tắc chạy đúng hai luồng và nhớ lựa chọn sau reload; nhãn/số trường MRZ đúng, chữ OCR mặt sau tùy chọn vẫn có, màn hình 390 px không tràn ngang.
- API MRZ chính không chứa ảnh/MRZ thô hoặc chữ OCR toàn ảnh; response của cặp giả lập khoảng 4,9 KB khi không yêu cầu chữ OCR mặt sau.
- Worker nhận dạng GPU và checkpoint giữ nguyên; chỉ cập nhật web và worker điều phối CPU. Tắt công tắc không khởi động lại model.

Các kiểm tra dùng ảnh giả lập; kết quả không thay thế đánh giá chất lượng trên ảnh căn cước thực tế.
