# QR, MRZ và OCR song song — 28/10/2026

Đã [đo lại 100 lượt trên cùng bộ ảnh](benchmarks/cccd-parallel-repeat-2026-10-28/README.md): trung bình GPU chuẩn **0,699 s**, luồng thử nghiệm **0,816 s**, tăng **0,117 s (+16,74%)** cho một cặp ảnh hai mặt. Kết quả gần lần đo đầu bên dưới; các bảng bên dưới dùng trung vị, còn lần đo lại báo cáo trung bình cộng.

**Nếu QR và MRZ không đọc được, vẫn đọc đầy đủ hai mặt bằng OCR.** Thứ tự QR → MRZ → OCR áp dụng khi chọn dữ liệu từng trường; không chờ hai nguồn đầu thất bại mới khởi động OCR. Bỏ các lượt MRZ thử thêm không làm mất nhánh OCR.

Trên năm cặp ảnh giả lập cũ, luồng thử nghiệm sau điều phối song song mất **0,75–0,88 giây**, trước đó **1,10–2,86 giây**. GPU chuẩn trong cùng lần đo mới mất 0,62–0,84 giây. Model và xử lý GPU chuẩn giữ nguyên.

## Số liệu

Mỗi ô là trung vị 5 lượt HTTP localhost sau làm nóng, đơn vị giây; gọi tuần tự và đảo thứ tự hai luồng. Cùng RTX 5060 8 GB, cùng worker CUDA. Hash từng ảnh đầu vào khớp chính xác bài đo trước. Bài đo cũ lúc 08:47–08:48 UTC; bài mới lúc 09:28:05–09:28:52 UTC, mỗi bài 50 lượt đo + 10 warm-up.

| Tình huống | GPU chuẩn cùng lần đo mới | Thử nghiệm trước | Thử nghiệm song song |
| --- | ---: | ---: | ---: |
| Ảnh rõ, có MRZ | 0,687 | 1,239 | **0,786** |
| Hai mặt mờ nhẹ | 0,737 | 1,105 | **0,844** |
| Mặt sau nghiêng/có nền | 0,842 | 2,861 | **0,882** |
| Không có MRZ | 0,623 | 1,501 | **0,818** |
| QR đọc được | 0,668 | 1,759 | **0,753** |

Đây là hai lần đo riêng trên cùng máy; so sánh GPU chuẩn trong từng lần giúp nhìn được dao động tải/tần số. Không coi chênh lệch với bài cũ là phép A/B ba nhánh chạy cùng lúc.

## Cách giảm thời gian

- QR, MRZ và OCR khởi chạy trên ba nhánh. Nhận dạng ảnh in bắt đầu ngay, không chờ QR/MRZ hoàn tất. Các nguồn đã xong được dùng để cập nhật danh sách cần bổ sung cho lượt OCR tiếp theo.
- GPU hiện vẫn đọc toàn bộ dòng của mỗi ảnh được gửi. Do quê quán/cơ quan cấp vẫn cần chữ in, cả năm tình huống đều có hai lượt OCR. Lần sửa này không thêm lọc vùng theo từng trường vào model GPU.
- Kết quả OCR được giữ trong request: QR có giá trị thì ưu tiên QR, kế đến trường MRZ vượt kiểm tra, sau đó dùng OCR điền trường còn thiếu. Tên có dấu tương đương tên MRZ và đối chiếu số căn cước được giữ.
- MRZ dừng ngay khi có trường hữu ích vượt kiểm tra, thay vì tiếp tục đọc vì một trường khác chưa đạt. Tối đa ba lượt cho cả ảnh căn chỉnh/ảnh gốc/hướng xoay, ngưỡng thử tiếp 900 ms. Ngưỡng này không cắt ngang một lượt đã chạy.
- Khi QR xác nhận danh tính và OCR đã bổ sung đủ các trường MRZ, dừng thử thêm MRZ. Luôn chạy lượt MRZ đầu tiên. Nếu không có QR, chỉ riêng việc OCR trả các chuỗi không rỗng không tự hủy phục hồi MRZ.
- Ảnh nghiêng trước đây mất khoảng **2,03 giây riêng MRZ**, gồm đọc lại ảnh gốc. Nay MRZ khoảng **0,57 giây**, chạy chồng lấp với khoảng **0,78 giây OCR**. Tổng pipeline không phải tổng cộng các khoảng này.

Một lượt `FastMRZ` trong báo cáo gồm tìm vùng và đọc ký tự bằng Tesseract; các lượt `bottom_region` đọc riêng dải cuối ảnh. Đây là các lần nhận dạng thật, không phải đọc lại JSON hoặc truy vấn cache. Giao diện hiển thị số lượt, thời gian và lý do dừng; JSON có `stage_intervals_ms` để kiểm chứng chồng lấp.

## Kiểm chứng dự phòng và chất lượng

- **504 test qua**, gồm kiểm tra ba nhánh chồng lấp bằng đồng bộ sự kiện, thứ tự ưu tiên không phụ thuộc nhánh nào xong trước và QR/MRZ đều lỗi thì OCR vẫn đọc đầy đủ hai mặt.
- Kiểm tra model khi không có mạng: MRZ ảnh rõ, mờ nhẹ và xoay 180° đều giữ được số căn cước có kiểm tra hợp lệ trong tối đa ba lượt.
- Làm mờ riêng **cả QR lẫn MRZ**, giữ phần chữ in: không nguồn mã nào được sử dụng; cả hai mặt vẫn OCR, điền 11 trường không rỗng. Tất cả 10 trường bắt buộc trong fixture khớp kết quả GPU chuẩn. Trong đó 9/10 khớp đáp án, cùng sai `issue_date` như GPU chuẩn: đầy đủ trường không đồng nghĩa tất cả chính xác.
- Ảnh mặt sau ngược: nếu MRZ xác định hướng sau khi OCR đã chạy, đọc lại ảnh đã xoay khi còn thiếu trường bắt buộc. Bài kiểm tra giữ được số căn cước MRZ và các trường cơ quan/ngày cấp từ OCR.
- 50 lượt benchmark không có OCR stage lỗi. Số trường khớp đáp án trên năm cặp ảnh giữ như trước: ảnh rõ 9/10 ở cả hai luồng; mờ/ảnh nghiêng/không MRZ 10/10; QR đọc được 11/11.
- Chrome: ảnh bốn góc, MRZ, xuất JSON, màn hình 390 px và quay lại form chính đều qua. Không gọi API thử nghiệm khi chưa mở tab.
- Hash tám file xử lý/JS chính khớp container trước thay đổi; worker GPU và checkpoint không thay đổi. Chỉ triển khai mã thử nghiệm và giao diện tương ứng.

Chưa đánh giá trên ảnh căn cước thật bị mờ/lóa hoặc tải nhiều người dùng. Dữ liệu giả lập xác nhận đường dự phòng và đối chiếu độ trễ, không chứng minh độ chính xác trên mọi ảnh.

## Artefact

- [50 mẫu đo và timeline](benchmarks/cccd-parallel-2026-10-28/results.json).
- [Kiểm chứng cả QR/MRZ không đọc được và ảnh ngược](benchmarks/cccd-parallel-2026-10-28/fallback-verification.json), không lưu giá trị căn cước/ảnh trong báo cáo.
- [Bài đo tuần tự trước đó](CCCD_FLOW_BENCHMARK.vi.md).
- [Hướng dẫn chạy](CCCD_EXPERIMENT.vi.md).

Dùng lại `scripts/benchmark-cccd-flows.py` với cùng manifest ảnh. Thời gian từng công đoạn chồng lấp không được cộng trực tiếp; script tính phần ngoài các stage bằng hợp các khoảng thời gian.
