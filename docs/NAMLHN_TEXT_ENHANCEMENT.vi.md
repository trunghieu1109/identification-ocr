# Khử nhiễu và làm nét chữ trong namlhn GPU

Namlhn đã có resize, bilateral denoising, CLAHE và sharpen ở ảnh dùng cho
Paddle tìm vùng chữ. Bản kết hợp DocU trước đây nhận dạng từ pixel gốc.
Nay bộ đọc có thêm chế độ `NAMLHN_GPU_TEXT_ENHANCEMENT=adaptive`:

1. Đọc ảnh gốc như trước.
2. Với dòng có confidence dưới 0,9, kiểm tra độ tương phản, nhiễu nền và
   độ nét. Ảnh rõ được giữ ở một lượt đọc.
3. Tối đa 12 vùng mỗi mặt được khử nhiễu nhẹ bằng Non-local Means trên
   kênh sáng, tăng tương phản bằng CLAHE khi cần, rồi làm nét unsharp nhẹ.
   Kích thước vùng và tọa độ được giữ nguyên. Các phép lọc OpenCV chạy CPU;
   detector và bộ nhận dạng đã cấu hình tiếp tục được dùng, không nạp model mới.
4. Đọc lại các vùng này bằng cùng recognizer. Chỉ nhận chữ mới khi confidence
   ít nhất 0,6, tăng tối thiểu 0,015 và độ dài không thay đổi quá lớn.
   Những nhóm chữ số đã có phải giữ nguyên, để tránh vô tình đổi số thẻ/ngày.
   Đây là quy tắc chọn ứng viên, không chứng minh ứng viên mới đúng.
5. Lỗi ở phần lọc/đọc bổ sung giữ kết quả từ ảnh gốc. Mọi crop tạm được đóng.

Tham khảo thuật toán gốc: [Non-local Means của OpenCV](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)
và [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html).
Lọc quá mạnh có thể làm mất chi tiết dấu tiếng Việt; vì vậy ảnh gốc luôn
được đọc trước và kết quả không được thay chỉ vì ảnh trông nét hơn.

## Kết quả thử ngày 25/10/2026

Sáu cặp ảnh giả lập, gồm ảnh sạch, nhãn nhỏ, không có nhãn hết hạn và ba
biến thể giảm chất lượng có seed cố định. Mỗi chế độ một lượt warmup và
hai lượt đo xen kẽ trên cùng model GPU, không HTTP/QR. Chấm 9 trường chung,
chuẩn hóa Unicode/hoa thường/khoảng trắng nhưng giữ dấu tiếng Việt.

| Trường hợp | Tắt lọc: đúng | Chọn lọc: đúng | Tắt lọc: trung vị | Chọn lọc: trung vị | Vùng lọc/cặp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ảnh sạch | 9/9 | 9/9 | 595 ms | 601 ms | 0 |
| Nhãn nhỏ | 7/9 | 7/9 | 560 ms | 574 ms | 0 |
| Mờ nhẹ + nhiễu | 9/9 | 9/9 | 574 ms | 589 ms | 0 |
| Tương phản thấp | 9/9 | 9/9 | 550 ms | 1.142 ms | 13 |
| Nhiễu mạnh | 9/9 | 9/9 | 539 ms | 1.077 ms | 11 |
| Không có nhãn hết hạn | 8/9 | 8/9 | 525 ms | 529 ms | 0 |

Trong phép thử ban đầu áp dụng lọc cho mọi vùng, ảnh sạch giảm từ 9/9 xuống
6/9 trường đúng. Chế độ chọn lọc giữ được kết quả gốc ở các ca trên,
nhưng **chưa cải thiện số trường đúng**. Các lỗi tên và nhãn quê quán trong
ca chữ nhỏ vẫn tồn tại. Ngày hết hạn vẫn đọc được khi có nhãn và không
được tạo ra khi thiếu nhãn. Bộ ảnh nhỏ và giả lập, không đại diện cho
ảnh căn cước thực tế hoặc tải vận hành.

Dữ liệu chỉ chứa chữ/ảnh giả lập:

- [So sánh GPU có kiểm tra chất lượng](benchmarks/namlhn-text-enhancement-2026-10-25/gpu-comparison.json)
- [Thử lọc toàn bộ vùng](benchmarks/namlhn-text-enhancement-2026-10-25/full-filter-exploration.json)
- [Kiểm tra API sau cập nhật](benchmarks/namlhn-text-enhancement-2026-10-25/adaptive-http.json)

## Cấu hình và quan sát

Compose và image namlhn GPU mặc định bật `adaptive`. Health worker có
`text_enhancement: adaptive`. API OCR trả chế độ trong `ocr_engine`, và
mỗi mặt có metadata `text_enhancement` với số vùng đã thử/đã đổi chữ.
`work.enhanced_regions`, `work.enhancement_changed_regions` cùng timing
`text_enhancement`, `enhanced_recognition` thể hiện chi phí bổ sung.

Tắt riêng phần lọc chữ, giữ bản kết hợp DocU và parser ghép địa chỉ:

```bash
NAMLHN_GPU_TEXT_ENHANCEMENT=off \
  docker compose --profile gpu up -d --no-build --no-deps --force-recreate namlhn-gpu
```

Bật lại:

```bash
NAMLHN_GPU_TEXT_ENHANCEMENT=adaptive \
  docker compose --profile gpu up -d --no-build --no-deps --force-recreate namlhn-gpu
```

Thêm biến tương ứng vào `.env` để lưu lựa chọn cho các lần tạo container sau.
