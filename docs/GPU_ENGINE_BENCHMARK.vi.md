> **Tài liệu lịch sử.** Các API thử nghiệm, worker DocU độc lập và pipeline legacy đã được gỡ ở bản 0.8.0. Các số liệu được giữ để đối chiếu; dùng [README hiện tại](../README.md) để triển khai.

# Thử nghiệm namlhn và DocU trên cùng GPU — 25/10/2026

Trên bộ ảnh giả lập này, **namlhn + VietOCR Seq2Seq trên GPU đạt trung vị
388 ms/cặp hai mặt**, so với **DocU + Seq2Seq 912 ms/cặp**, nhanh hơn khoảng
**2,35 lần**. Giữ Transformer của nhánh namlhn hiện tại và chuyển sang GPU
đạt **1.062 ms/cặp**, chậm hơn DocU khoảng **16,5%** trong phép đo này.

Đây là phép đo nhỏ của pipeline ứng dụng trên ảnh giả lập, chưa đủ để kết
luận về độ chính xác hoặc tốc độ trên tập căn cước thực tế. Không thay đổi
API, cấu hình model hoặc container đang phục vụ người dùng.

## Điều kiện đo

- NVIDIA GeForce RTX 5060, driver 580.178.04; CUDA capability 12.0.
- PyTorch 2.7.1+cu128, VietOCR 0.3.13. Nhánh namlhn dùng PaddlePaddle GPU
  3.2.2/cu129, PaddleOCR 3.7.0, detector/recognizer PP-OCRv6 medium.
- Hai cặp ảnh giả lập, mỗi mặt 1200×760: một cặp có ngày hết hạn ở mặt
  trước, một cặp có ngày hết hạn ở mặt sau. Sinh ảnh từ cùng một script;
  fingerprint SHA-256 của ảnh được lưu trong từng JSON kết quả.
- Mỗi cấu hình chạy trong một container mới, tuần tự; model được giữ
  trong bộ nhớ suốt lượt đo. Dịch vụ sản phẩm vẫn chạy, GPU rảnh trước
  thử nghiệm; không chạy đồng thời hai cấu hình benchmark.
- Đọc trực tiếp RGB đã giải mã sẵn, mặt trước rồi mặt sau. Bỏ HTTP,
  upload, giải mã ảnh và QR. Cùng yêu cầu 7 trường mặt trước và 2 trường
  mặt sau, tương đương 9 trường bắt buộc của API chính khi không có QR.
- Dùng pipeline namlhn đã tối ưu của ứng dụng: Paddle đọc lần đầu, VietOCR
  chỉ đọc lại các vùng cần thiết, theo batch. Không dùng nguyên luồng
  upstream vốn đọc lại từng vùng trong mọi trường hợp. DocU dùng worker
  hiện tại của ứng dụng, giữ cắt thẻ, chỉnh hướng, phục hồi và cơ quan cấp.
- Hai cấu hình Seq2Seq dùng **cùng checkpoint và cấu hình nhận dạng**;
  SHA-256: `0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db`.
- Tách lượt xử lý đầu; warm thêm 2 vòng trên toàn bộ ảnh, sau đó đo 5 vòng.
  Mỗi cấu hình có 10 mẫu cặp hai mặt, tương đương 20 lượt ảnh được đo.
  Gọi CUDA synchronize trước/sau để tránh chỉ đo thời gian đưa việc vào GPU.
- `pair_ms` là tổng thời gian hai lượt engine, gồm tiền xử lý và chuẩn hóa
  kết quả, không gồm các kiểm tra ground truth của script. `pair_wall_ms`
  giữ thêm thời gian các kiểm tra này để đối chiếu. Các timing chi tiết
  của worker có thể lồng nhau; không cộng tất cả thành tổng mới.

Script xác nhận PyTorch CUDA, Paddle GPU, `PaddleOCR.device=gpu:0` và tham
số model VietOCR trên `cuda:0`; không có nhánh CPU fallback trong phép đo.

## Thời gian khi model đã nóng

Đơn vị **ms cho một cặp mặt trước + mặt sau**; cùng trọng số cho hai loại cặp.

| Pipeline GPU | Model VietOCR | Trung vị | Trung bình | Nhỏ nhất–lớn nhất |
| --- | --- | ---: | ---: | ---: |
| namlhn | Transformer | 1.061,79 | 1.060,85 | 834,72–1.287,07 |
| namlhn | Seq2Seq | 388,29 | 387,90 | 349,42–425,92 |
| DocU | Seq2Seq | 911,78 | 916,72 | 770,34–1.086,90 |

Tách theo cặp, trung vị của 5 lần:

| Cặp ảnh | namlhn Transformer | namlhn Seq2Seq | DocU Seq2Seq |
| --- | ---: | ---: | ---: |
| Ngày hết hạn trên mặt trước | 1.283,23 | 424,19 | 1.055,81 |
| Ngày hết hạn trên mặt sau | 842,63 | 351,06 | 773,23 |

## Khởi tạo và lượt đầu

Đơn vị ms. Khởi tạo gồm lazy import framework/engine và nạp model sau khi
script đã import PyTorch, kiểm tra CUDA và mở context ban đầu. Model đã có
trên đĩa; không tính tải thư viện/checkpoint. Lượt đầu được đo **sau** khởi
tạo, không bao gồm thời gian khởi tạo. Đây không phải tổng cold start HTTP.

| Pipeline | Khởi tạo | Cặp đầu tiên sau khởi tạo |
| --- | ---: | ---: |
| namlhn Transformer | 6.912,52 | 2.327,06 |
| namlhn Seq2Seq | 6.538,20 | 1.355,15 |
| DocU Seq2Seq | 3.620,69 | 2.078,99 |

## Đối chiếu kết quả đọc

So sánh 9 trường được in trên ảnh với ground truth, bỏ khác biệt hoa/thường
và dấu cách hai đầu. Cả 5 vòng của mỗi loại cặp cho cùng kết quả:

| Pipeline | Cặp có hạn ở mặt trước | Cặp có hạn ở mặt sau | Ngày hết hạn |
| --- | --- | --- | --- |
| namlhn Transformer | Đúng chính xác 8/9 trường | Đúng chính xác 8/9 trường | Đúng ở cả hai vị trí |
| namlhn Seq2Seq | Đúng chính xác 8/9 trường | Đúng chính xác 8/9 trường | Đúng ở cả hai vị trí |
| DocU Seq2Seq | Đúng chính xác 7/9 trường | Đúng chính xác 8/9 trường | Đúng ở cả hai vị trí |

Cả ba chưa khớp chính xác trường tên trên ảnh giả lập; DocU còn thiếu nơi
thường trú ở cặp thứ nhất. Số trường có giá trị khác rỗng không phải số
trường đọc đúng. Không dùng các tỷ lệ này để suy ra accuracy của thẻ thật.

DocU phải phục hồi từ ảnh gốc sau bước chỉnh hướng trên bộ ảnh này; các
lượt đó vẫn được tính. Namlhn có tiền xử lý và chọn vùng đọc lại khác
DocU. Do đó, chênh lệch Seq2Seq phản ánh **toàn pipeline hiện tại**, không
phải kết luận detector hoặc model riêng lẻ nhanh hơn 2,35 lần.

## Môi trường thử và giới hạn

Image thử kế thừa worker DocU, thêm Paddle GPU và mã nguồn namlhn. Wheel
Paddle lấy từ [kho CUDA 12.9 chính thức](https://www.paddlepaddle.org.cn/packages/stable/cu129/paddlepaddle-gpu/).
Paddle/cu129 nâng các gói NVIDIA so với pin của Torch/cu128 trong base;
pip có báo khác pin. Đường dẫn cuSPARSELt đã được khai báo rõ để Torch
khởi động được. Cả hai framework đã thực thi tensor và inference GPU thật
thành công, cả ba cấu hình dùng cùng image cuối. Đây là image thử nghiệm,
chưa phải cấu hình dependency được kiểm định để thay dịch vụ sản phẩm.

Số mẫu nhỏ, ảnh tổng hợp rõ chữ, không có phản quang/ảnh chụp thật và chưa
đo tải đồng thời. p95 trong JSON chỉ là gần giá trị lớn nhất của 10 mẫu,
chưa đại diện cho p95 vận hành. Không so trực tiếp bảng này với các số
HTTP/QR trong báo cáo tối ưu trước đây vì phạm vi đo khác nhau.

## Tái lập và dữ liệu thô

- [Dockerfile thử nghiệm](../Dockerfile.benchmark-gpu)
- [Script sinh ảnh](../scripts/create-gpu-benchmark-fixtures.py)
- [Script đo](../scripts/benchmark-gpu-engines.py)
- [namlhn Transformer](benchmarks/gpu-2026-10-25/namlhn-vgg_transformer.json)
- [namlhn Seq2Seq](benchmarks/gpu-2026-10-25/namlhn-vgg_seq2seq.json)
- [DocU Seq2Seq](benchmarks/gpu-2026-10-25/docu-vgg_seq2seq.json)
- [Phiên bản thư viện và image](benchmarks/gpu-2026-10-25/runtime.json)
- [Manifest ground truth](benchmarks/gpu-2026-10-25/dataset.json)

Ví dụ chạy một cấu hình, từ thư mục repo; thay `--engine`, `--model` và
tên output cho cấu hình còn lại. Các base image sản phẩm cần có sẵn.

```bash
docker build -f Dockerfile.benchmark-gpu -t identification-ocr-benchmark-gpu:experiment .
python scripts/create-gpu-benchmark-fixtures.py --output /tmp/gpu-engine-benchmark/dataset
docker cp identification-ocr-identification-ocr-1:/root/.paddlex /tmp/gpu-engine-benchmark/paddlex-cache
docker run --rm --gpus all --shm-size=1g \
  -v /tmp/gpu-engine-benchmark:/benchmark \
  -v /tmp/gpu-engine-benchmark/paddlex-cache:/root/.paddlex \
  identification-ocr-benchmark-gpu:experiment \
  --engine namlhn --model vgg_seq2seq \
  --dataset /benchmark/dataset/dataset.json \
  --output /benchmark/namlhn-vgg_seq2seq.json --repeats 5 --warmup 2
```

Nếu cache Paddle chưa tồn tại, framework cần tải model trước; không đưa
thời gian tải vào số inference. Có thể chỉ định dataset khác để đo trên
ảnh được phép sử dụng; JSON đo không lưu chữ OCR hoặc giá trị trường,
chỉ lưu các kiểm tra khớp và ngày hết hạn đã nhận dạng.
