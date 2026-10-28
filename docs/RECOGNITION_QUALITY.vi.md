> **Tài liệu lịch sử.** Các API thử nghiệm, worker DocU độc lập và pipeline legacy đã được gỡ ở bản 0.8.0. Các số liệu được giữ để đối chiếu; dùng [README hiện tại](../README.md) để triển khai.

# Ghi nhận thử nghiệm bộ nhận dạng

**AsterNet đã được gỡ theo yêu cầu người dùng vì kết quả trên ảnh thực tế
kém hơn. Mặc định hiện tại là VietOCR `vgg_seq2seq`.** Detector, crop,
parser, API và script export tiếp tục dùng luồng namlhn + DocU VietOCR.
Không còn checkpoint, ONNX Runtime riêng hoặc cấu hình chọn AsterNet trong
worker được build từ `Dockerfile.namlhn-gpu`.

Các số liệu cũ chỉ là lịch sử thử nghiệm trên ảnh tổng hợp. Chúng không
chứng minh chất lượng trên ảnh căn cước thực tế của người dùng:

- [Thử nghiệm ban đầu](benchmarks/recognition-quality-2026-10-25/development.json).
- [Thử nghiệm 96 crop tổng hợp](benchmarks/recognition-quality-2026-10-25/holdout.json).
- [Manifest bộ ảnh tổng hợp](benchmarks/recognition-quality-2026-10-25/holdout-dataset.json).
- [API lúc thử AsterNet](benchmarks/recognition-quality-2026-10-25/deployed-http.json).
- [API chính lúc thử AsterNet](benchmarks/recognition-quality-2026-10-25/main-http.json).

Xuất crop và text bằng bộ đọc hiện tại:

```bash
python3 scripts/export-front-ocr.py front.jpg --docker --model vgg_seq2seq
```

Đối chiếu với VietOCR Transformer:

```bash
python3 scripts/export-front-ocr.py front.jpg --docker --model vgg_transformer
```

Mỗi lượt xuất dùng thư mục mới hoặc đang trống. Kết quả xuất trước đây
được giữ để đối chiếu.
