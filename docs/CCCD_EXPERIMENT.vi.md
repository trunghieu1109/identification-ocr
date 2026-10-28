# Thử nghiệm DocAligner + FastMRZ

Tab **Thử nghiệm · MRZ & căn chỉnh** có API chẩn đoán riêng. Từ 0.9.0, API GPU chính cũng dùng worker này khi bật **MRZ và căn chỉnh ảnh**; tắt để dùng lại QR + OCR cũ. Xem [công tắc giao diện, API và server](GPU_MRZ_SWITCH.vi.md). API chính chỉ trả kết quả gọn; ảnh/MRZ thô vẫn ở tab thử nghiệm.

## Chạy

```bash
docker compose --profile gpu --profile experiment up -d --build
```

Truy cập cổng `8888`, mở tab thử nghiệm, chọn hai ảnh cùng thẻ, bấm **Chạy thử nghiệm**. Có thể tắt DocAligner rồi chạy lại để so sánh. Dùng ô xoay ảnh với ảnh mặt trước ngang/ngược; EXIF được áp dụng trước. MRZ mặt sau được thử thêm hướng ngược lại.

Chỉ tắt worker thử nghiệm:

```bash
docker compose --profile experiment stop cccd-experiment
```

Khi worker này dừng, GPU chính đang bật MRZ tự dùng lại QR + OCR cũ và ghi cảnh báo. Khi tắt MRZ, GPU chính không gọi worker này.

## Xử lý

1. Xác thực cả hai ảnh (JPG/PNG, tối đa 10 MB và 20 megapixel mỗi ảnh), áp dụng EXIF và góc xoay đã chọn.
2. DocAligner dự đoán bốn góc. Chỉ warp nếu điểm tìm tài liệu đạt ngưỡng, tứ giác hợp lệ, đủ lớn và nằm trong ảnh. Warp lấy pixel ảnh gốc; trường hợp không đạt giữ nguyên ảnh.
3. Khởi chạy **QR, MRZ và OCR song song**. QR đọc ảnh gốc của cả hai mặt, thử ảnh căn chỉnh khi ảnh gốc chưa đọc được.
4. FastMRZ đọc MRZ mặt sau với tối đa **ba lượt nhận dạng cho cả request**, dùng chung ngân sách thử 900 ms. Dừng khi đã có trường vượt kiểm tra; nếu không tìm thấy vùng MRZ, ưu tiên ảnh gốc/hướng ngược thay vì đọc đi đọc lại cùng một vùng. QR đã xác nhận danh tính và OCR đã bổ sung đủ các trường MRZ thì bỏ các lượt MRZ còn lại. Luôn giữ lượt MRZ đầu tiên để đối chiếu danh tính. Ngưỡng thời gian được kiểm tra trước khi bắt đầu lượt tiếp theo; không cắt ngang một lần nhận dạng đang chạy.
5. OCR bằng **worker namlhn-gpu hiện có** chạy ngay cho quê quán/cơ quan cấp và dự phòng. Model GPU hiện vẫn nhận dạng toàn bộ dòng của mỗi ảnh được gửi; danh sách `requested_fields` mô tả phần cần bổ sung, chưa có lọc crop theo từng trường. Nếu ảnh căn chỉnh cho quá ít trường thì đọc ảnh gốc dự phòng. Nếu MRZ xác định ảnh bị ngược sau khi OCR chạy, chỉ đọc lại ảnh đã xoay khi còn thiếu trường bắt buộc.
6. Khi ghép, chọn **QR → MRZ đã kiểm tra → OCR**, độc lập với thứ tự các nhánh hoàn tất. OCR chỉ điền chỗ trống; dùng lại tên in có dấu nếu tương đương tên MRZ và giữ đối chiếu số căn cước giữa các mặt. Hai số căn cước QR/MRZ khác nhau chặn ghép tự động.

**QR và MRZ đều không đọc được thì OCR vẫn đọc đầy đủ cả hai mặt.** Không có nhánh trả sớm làm mất OCR. Giới hạn/dừng sớm chỉ áp dụng cho các lượt thử MRZ bổ sung. Khi không có QR xác nhận danh tính, việc OCR trả đủ chuỗi không rỗng không tự hủy phục hồi MRZ; MRZ vẫn được dùng hết số lượt/ngân sách cho phép.

Thời gian từng công đoạn trong `timings_ms` có phần chồng lấp khi chạy song song; không cộng tất cả để suy ra tổng. `stage_intervals_ms` ghi mốc bắt đầu/kết thúc để kiểm chứng, `execution` ghi chính sách điều phối. Hai lượt OCR GPU vẫn được worker hiện có xử lý qua hàng đợi an toàn, không nhân bản model.

Giao diện hiển thị ảnh dùng cho OCR, chữ MRZ, từng checksum, nguồn từng trường, cảnh báo và JSON tải xuống. Ảnh upload/căn chỉnh không được lưu trên server; ảnh xem trước chỉ nằm trong response của request thử nghiệm.

## Phân tích MRZ Việt Nam

FastMRZ được gọi với `ignore_parse=True`: phần đọc ảnh/segmentation vẫn dùng FastMRZ. Lớp phân tích riêng chỉ nhận TD1 ba dòng, mỗi dòng đúng 30 ký tự, nước phát hành/quốc tịch VNM, bố cục số căn cước 12 chữ số trong optional data đã hỗ trợ.

- Kiểm tra checksum số tài liệu, số căn cước, ngày sinh, ngày hết hạn và checksum tổng hợp. Kiểm tra tính hợp lệ ngày tháng.
- Không sửa O thành 0, thêm ký tự hoặc thử số đến khi checksum khớp.
- Không gán số tài liệu chín ký tự thành **CMND cũ**; chỉ QR cung cấp trường CMND cũ.
- Năm sinh hai chữ số chưa xác định được thế kỷ được giữ để đối chiếu QR/chữ in, không dùng ngưỡng năm cố định để đoán.
- Tên MRZ không có dấu; tên, giới tính, quốc tịch không được checksum tổng hợp bảo vệ. Giá trị MRZ này cần đối chiếu khi được chọn làm kết quả.
- Checksum kiểm tra tính nhất quán dữ liệu đọc, không xác thực giấy tờ hay danh tính.

Mẫu MRZ khác chỉ hiển thị để đối chiếu. Với MRZ đọc thiếu một phần, chỉ dùng trường có checksum riêng hợp lệ khi số căn cước, số tài liệu và checksum tổng hợp đều khớp; tên/giới tính/quốc tịch chưa được tự điền từ MRZ một phần. Chưa đánh giá độ chính xác trên bộ ảnh căn cước thực tế chụp mờ/lóa.

## API và tài nguyên

- `GET /api/cccd/experiment/health`: kiểm tra riêng worker thử nghiệm.
- `POST /api/cccd/experiment/extract`: multipart `front_image`, `back_image`, `align` (mặc định true), `front_rotation`, `back_rotation` (0/90/180/270).
- Worker `cccd-experiment:8083` chỉ trong mạng Docker, không publish cổng, hàng đợi riêng tối đa hai request đang chờ. Cùng dùng hàng đợi OCR GPU hiện có khi đọc chữ.
- `CCCD_EXPERIMENT_URL`, `CCCD_EXPERIMENT_API_KEY`: cấu hình HTTP nội bộ. Main health không probe worker; giao diện chỉ kiểm tra worker khi mở tab thử nghiệm.
- DocAligner LCNet050 ≈ 4.9 MB, Tesseract MRZ ≈ 11.4 MB, segmentation trong wheel FastMRZ ≈ 5.3 MB. Không có Torch/CUDA trong image thử nghiệm, không cần volume model mới.
- Model tải khi build và kiểm tra SHA-256 trong `scripts/setup-experiment-models.py`; runtime không cần tải model qua mạng.

## Nguồn tham khảo

- [FastMRZ](https://github.com/sivakumar-mahalingam/fastmrz), package `2.1.2`, AGPL-3.0. Model Tesseract từ revision `bb1d4ee38eb1eaa88cd80e1eba88bacc1842cd76`.
- [DocAligner](https://github.com/DocsaidLab/DocAligner), revision `3275b0f07f8e99d8c01cb0774dea2549be1416b6`, Apache-2.0. Dùng model point LCNet050 chính thức qua ONNXRuntime CPU, cùng quy ước BGR/NCHW/[0,1] và output điểm chuẩn hóa; không cài bộ Capybara.
- [Demo_CCCD_OCR](https://github.com/KhongMinhCuong/Demo_CCCD_OCR), revision `f45ceef63925b2cc44e27c27b9a661a41de4c9c4`: tham khảo cách tách crop, đọc MRZ riêng, đối chiếu nguồn và giữ ảnh gốc khi crop không chắc. Viết adapter mới, không sao chép mã của repository chưa công bố giấy phép.
- [ICAO Doc 9303 Part 5](https://www.icao.int/sites/default/files/publications/DocSeries/9303_p5_cons_en.pdf): cấu trúc TD1 và checksum.

Giữ bản giấy phép upstream tại `docs/licenses/` và trong image thử nghiệm. OCRDocs chưa được tích hợp.

## Kiểm chứng ngày 28/10/2026

Các số liệu bên dưới ghi lần tích hợp thử nghiệm trước 0.9.0. Khi đưa vào luồng GPU chính, [kiểm chứng công tắc bật/tắt](GPU_MRZ_SWITCH.vi.md) thay thế giả định API chính không gọi worker MRZ.

- 493 test qua trên image web triển khai, gồm kiểm tra không gọi worker thử nghiệm từ API chính, checksum/ngày MRZ, ảnh crop không hợp lệ và xung đột danh tính.
- FastMRZ thực đọc đúng MRZ trên ảnh giả lập rõ, mờ Gaussian nhẹ và xoay 180 độ khi chạy không có mạng.
- Ảnh giả lập có nền/phối cảnh được DocAligner căn chỉnh; số căn cước MRZ vượt kiểm tra được sử dụng dù ngày hết hạn MRZ đọc sai. Ngày sai không được đưa vào kết quả từ MRZ.
- API chính trả nguyên toàn bộ 12 trường so với baseline trên hai cặp ảnh giả lập.
- Chrome: upload, ảnh bốn góc, ảnh căn chỉnh, JSON tải xuống, giữ trạng thái form chính, viewport 390 px không tràn ngang; chưa mở tab thì không phát sinh request API thử nghiệm.
- Runtime web đang triển khai giữ nguyên các layer thư viện/model cũ, chỉ thêm layer mã ứng dụng. Worker GPU chính giữ nguyên image và không khởi động lại. Image CPU thử nghiệm khoảng 726 MB; image/cache trung gian của lần build đã được dọn.

Các kết quả trên xác nhận tích hợp và regression; không thay thế benchmark bằng ảnh căn cước thực tế.

So sánh độ trễ với GPU chuẩn: [bài đo song song mới và kiểm chứng OCR dự phòng](CCCD_PARALLEL_BENCHMARK.vi.md); [bài đo tuần tự trước đó](CCCD_FLOW_BENCHMARK.vi.md).
