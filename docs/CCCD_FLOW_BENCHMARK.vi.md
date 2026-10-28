# Thời gian GPU chuẩn và DocAligner + FastMRZ — 28/10/2026

Đây là số liệu **trước khi điều phối song song**. Bản thử nghiệm hiện tại đã cập nhật; xem [bài đo mới và kiểm chứng OCR dự phòng](CCCD_PARALLEL_BENCHMARK.vi.md).

Luồng thử nghiệm hiện mất **1,59–3,50 lần thời gian** của GPU chuẩn trên năm tình huống giả lập. Phần tăng chủ yếu nằm ở đọc MRZ/thử lại; căn chỉnh bốn góc chiếm 9–16 ms cho hai mặt.

## Cách đo

- Đo hai API đang chạy trên cùng host: `/api/cccd/extract` với `quality=high` và `/api/cccd/experiment/extract` với `align=true`.
- Cùng NVIDIA GeForce RTX 5060 8 GB, cùng worker Paddle PP-OCRv6_medium_det + DocU VietOCR vgg_seq2seq. DocAligner LCNet050 và FastMRZ/Tesseract chạy CPU, host AMD Ryzen 9 5950X.
- Mỗi tình huống dùng đúng cùng hai file cho cả hai luồng. Front 1200 × 760; back 1200 × 760 hoặc 1800 × 1200 khi có nền/phối cảnh. Ảnh mờ dùng Gaussian radius 1,0 px.
- Mỗi luồng làm nóng một lần cho mỗi cặp ảnh, loại khỏi thống kê. Sau đó đo 5 lần/luồng/cặp; tổng **50 lượt đo và 10 lượt làm nóng**. Gọi tuần tự, đảo thứ tự hai luồng sau mỗi vòng.
- Thời gian bảng chính là **trung vị wall time HTTP localhost**, từ gửi request đến nhận hết response. Gồm upload, hàng đợi, xử lý, JSON/ảnh preview. Chưa tính đường truyền từ máy người dùng qua Internet.
- Đo từ 08:47:43 đến 08:48:54 UTC. Thời gian chờ GPU trung bình 0,08–0,14 ms, không có dấu hiệu tranh hàng đợi đáng kể trong bài đo.
- Toàn bộ ảnh dùng dữ liệu giả lập. Không khởi động lại hay thay đổi pipeline/model để đo. Báo cáo JSON không lưu thông tin căn cước, chữ OCR hoặc ảnh preview.

## Kết quả

| Tình huống | GPU chuẩn | Luồng mới | Tăng | Tỷ lệ thời gian mới/chuẩn |
| --- | ---: | ---: | ---: | ---: |
| Ảnh rõ, có MRZ, không QR | 0,657 s | 1,239 s | +0,583 s | 1,89× |
| Hai mặt mờ nhẹ, có MRZ, không QR | 0,693 s | 1,105 s | +0,411 s | 1,59× |
| Mặt sau nghiêng/có nền, không QR | 0,818 s | 2,861 s | +2,043 s | 3,50× |
| Không có MRZ, không QR | 0,619 s | 1,501 s | +0,881 s | 2,42× |
| QR đọc được, ảnh có MRZ | 0,666 s | 1,759 s | +1,094 s | 2,64× |

Khoảng min–max của năm lần đo:

| Tình huống | GPU chuẩn | Luồng mới |
| --- | ---: | ---: |
| Ảnh rõ/MRZ | 0,649–0,672 s | 1,212–1,296 s |
| Mờ nhẹ | 0,688–0,705 s | 1,078–1,114 s |
| Nghiêng/có nền | 0,796–0,829 s | 2,826–2,885 s |
| Không MRZ | 0,612–0,621 s | 1,485–1,536 s |
| QR đọc được | 0,664–0,701 s | 1,738–1,842 s |

## Phần xử lý thêm nằm ở đâu

Dưới đây là **trung bình từng nhóm công đoạn phía server**, đơn vị ms. Đây là phép tổng hợp khác với trung vị wall time phía trên.

| Tình huống | OCR GPU chuẩn | OCR luồng mới | DocAligner hai mặt | MRZ + dự phòng |
| --- | ---: | ---: | ---: | ---: |
| Ảnh rõ/MRZ | 620,65 | 663,13 | 8,88 | 504,24 |
| Mờ nhẹ | 654,35 | 706,95 | 9,08 | 313,50 |
| Nghiêng/có nền | 745,64 | 720,88 | 16,45 | 2025,18 |
| Không MRZ | 584,22 | 626,59 | 9,29 | 799,68 |
| QR đọc được | 641,54 | 677,39 | 8,79 | 1016,60 |

- Cả hai luồng đều chạy hai lượt OCR, một lượt mỗi mặt; không có OCR fallback bổ sung trong năm tình huống này. MRZ hiện được cộng thêm vào luồng thử nghiệm, chưa thay thế lượt OCR nào.
- Luồng mới vẫn chạy MRZ khi QR đã đọc được và vẫn thử vùng cuối ảnh/hướng khác khi không đọc ra MRZ hợp lệ.
- Với mặt sau nghiêng, MRZ đọc ảnh căn chỉnh rồi thử lại ảnh gốc: khoảng 1 giây mỗi lượt. Kết quả cuối chỉ dùng số căn cước MRZ có kiểm tra hợp lệ; ngày MRZ chưa đạt không được đưa vào kết quả từ MRZ.
- QR chiếm khoảng 11–21 ms, không phải phần lớn thời gian của bài đo này.
- Response thử nghiệm khoảng **180–250 KB**, so với **3,3–3,5 KB** của luồng chính, do kèm ảnh xem trước và chữ OCR. Khi dùng qua mạng chậm, chênh lệch người dùng cảm nhận có thể cao hơn đo localhost.

## Đối chiếu kết quả và giới hạn

Tất cả 50 lượt trả HTTP 200 và không có stage OCR lỗi. Luồng mới có thêm một trường không rỗng (`identifying_features`), do luồng chính bỏ qua trường tùy chọn này. Số trường không rỗng không đồng nghĩa với độ chính xác tốt hơn.

Đối chiếu các trường có đáp án trong manifest: tình huống ảnh rõ/MRZ đạt 9/10 ở **cả hai** luồng, cùng lệch `issue_date`; ba tình huống mờ nhẹ/nghiêng/không MRZ đạt 10/10; trường hợp QR đạt 11/11. Đây là kiểm tra dữ liệu của fixture, chưa phải benchmark độ chính xác trên ảnh thật. MRZ được kiểm tra đầy đủ ở ảnh rõ/mờ nhẹ, chỉ kiểm tra được một phần ở ảnh nghiêng, không được sử dụng ở hai tình huống còn lại.

Năm mẫu mỗi tình huống đủ đối chiếu độ trễ ở tải thấp; chưa đo p95 tải thực tế, thời gian khởi động model, ảnh điện thoại 12–20 MP hoặc nhiều người dùng đồng thời. Không suy ra tỷ lệ cải thiện nhận dạng ảnh mờ thực tế từ bộ này.

## Dữ liệu và chạy lại

- [Tất cả mẫu và thống kê](benchmarks/cccd-flow-comparison-2026-10-28/results.json): thời gian, hash hai ảnh đầu vào, số trường đối chiếu, trạng thái MRZ/căn chỉnh và hàng đợi.
- [Phần cứng, image và health](benchmarks/cccd-flow-comparison-2026-10-28/environment.json).
- Script [benchmark-cccd-flows.py](../scripts/benchmark-cccd-flows.py) chỉ dùng Python standard library, đo hai API trên cùng ảnh và không lưu nội dung thông tin cá nhân.

Tạo manifest cho bộ ảnh của bạn (đường dẫn tương đối theo vị trí manifest hoặc đường dẫn tuyệt đối):

```json
{"synthetic": false, "cases": [{"name": "cap_anh_1", "front": "/path/front.jpg", "back": "/path/back.jpg"}]}
```

```bash
python scripts/benchmark-cccd-flows.py \
  --dataset /path/cases.json --requests 5 --warmup 1 \
  --output /tmp/cccd-flow-results.json
```
