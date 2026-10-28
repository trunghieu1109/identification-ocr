# identification-ocr — OCR và QR căn cước

Dịch vụ nhận ảnh hai mặt của cùng một căn cước, đọc QR bằng WeChatQRCode,
rồi dùng OCR để bổ sung các trường trống. Giao diện tiếng Việt và API FastAPI
được phục vụ cùng một địa chỉ. Phiên bản hiện tại: **0.9.0**.

```text
Trình duyệt / API client
  → identification-ocr:8080 (cổng ngoài mặc định 8888)
    → high, MRZ bật: cccd-experiment:8083 điều phối QR + MRZ + OCR song song
        → DocAligner căn chỉnh; FastMRZ/Tesseract đọc MRZ trên CPU
        → OCR qua namlhn-gpu:8082; ghép QR → MRZ → OCR
      high, MRZ tắt: QR + OCR qua namlhn-gpu như trước
        → hiệu chỉnh ảnh khi cần
        → Paddle PP-OCRv6_medium_det tìm dòng chữ
        → DocU VietOCR vgg_seq2seq đọc từng dòng trên CUDA/FP32
        → gán trường theo nhãn và vị trí
      medium: namlhn CPU trong dịch vụ web
    → ghép dữ liệu, giữ giá trị QR hợp lệ, trả JSON
```

Profile `gpu` có **ba service**: `identification-ocr`, `namlhn-gpu` và worker CPU
`cccd-experiment` điều phối MRZ/căn chỉnh. Bỏ chọn **Bật MRZ và căn chỉnh ảnh**
trong tab chính để dùng lại luồng GPU QR + OCR cũ cho request tiếp theo.
Worker GPU có Dockerfile độc lập; không cần image benchmark hoặc worker DocU cũ.
Toàn bộ suy luận chạy trên máy chủ; ảnh và kết quả không lưu vào database.

## Chạy bằng Docker

Mặc định dùng GPU; host cần NVIDIA driver và NVIDIA Container Toolkit:

```bash
docker compose --profile gpu up -d --build
```

Mở `http://localhost:8888`, Swagger tại `/docs`. Đổi cổng qua `OCR_PORT`.
Chạy CPU khi không có GPU:

```bash
OCR_DEFAULT_QUALITY=medium docker compose up -d --build identification-ocr
```

Compose mặc định `high`; chạy ngoài Compose không đặt `OCR_DEFAULT_QUALITY`
thì mặc định `medium`. GPU lỗi sẽ báo lỗi OCR và giữ dữ liệu đã đọc được;
không tự chuyển sang CPU.

```bash
docker compose logs -f identification-ocr namlhn-gpu
docker compose down
```

Mã upstream được pin commit trong Dockerfile. Checkpoint DocU Seq2Seq lấy từ
image nguồn đã pin digest, chỉ sao chép đúng file `vgg_seq2seq.pth` vào image cuối.
Paddle tải detector lần đầu và giữ trong volume. WeChat có đủ bốn file model
ngay khi build. CPU chỉ khởi tạo khi chọn `medium` hoặc gọi API OCR riêng;
với mặc định `medium`, model CPU được khởi tạo trong nền khi ứng dụng bật.

## Model được giữ lại

| Luồng | Model |
| --- | --- |
| QR | WeChat detector và super-resolution (`detect.*`, `sr.*`) |
| GPU chính | Paddle `PP-OCRv6_medium_det` và DocU `vgg_seq2seq.pth` |
| CPU `medium` | Paddle detector/recognizer của namlhn và VietOCR `vgg_transformer` mặc định |

Không còn QRDet, Restormer, SwinIR, YOLO detector/segmentation, model xoay thẻ,
GPU Transformer legacy, hoặc checkpoint ImageNet VGG19 riêng trong image GPU.
DocU recognizer nạp toàn bộ trọng số từ checkpoint Seq2Seq, không cần tải backbone riêng.

## Cấu hình

| Biến | Mặc định / ý nghĩa |
| --- | --- |
| `OCR_PORT` | `8888`, cổng public |
| `OCR_DEFAULT_QUALITY` | Compose: `high`; trực tiếp: `medium` |
| `GPU_MRZ_ENABLED` | Compose: `1`; trực tiếp: `0`. Mặc định chọn luồng GPU song song; `0` dùng GPU QR + OCR cũ. Request `use_mrz` có thể chọn riêng. |
| `CCCD_EXPERIMENT_URL` | Worker MRZ/căn chỉnh, Compose: `http://cccd-experiment:8083` |
| `CCCD_EXPERIMENT_API_KEY` | Bearer token nội bộ cho worker MRZ nếu được đặt |
| `NAMLHN_GPU_OCR_URL` | Compose: `http://namlhn-gpu:8082` |
| `NAMLHN_GPU_OCR_API_KEY` | Bearer token giữa API và worker nếu được đặt |
| `GPU_OCR_TIMEOUT` | `120` giây, timeout HTTP của worker |
| `NAMLHN_GPU_TEXT_ENHANCEMENT` | `adaptive`, hoặc `off` |
| `NAMLHN_GPU_CARD_ENHANCEMENT` | `adaptive`, hoặc `off` |
| `OCR_QUEUE_SIZE` | `8` ảnh chờ ngoài tác vụ đang xử lý |
| `OCR_QUEUE_TIMEOUT` | `120` giây chờ bắt đầu inference |
| `OCR_CPU_THREADS` | `4` |
| `OCR_MODEL` | Model CPU, mặc định `vgg_transformer` |
| `OCR_ENABLE_MKLDNN` | `1`; đặt `0` để chẩn đoán CPU |

GPU cố định dùng `vgg_seq2seq` và chiến lược `namlhn_docu_v1`. Các biến chọn
pipeline legacy, DocU worker cũ và phục hồi QR đã được bỏ khỏi Compose.
Mỗi engine giữ model trên một thread và có FIFO trong bộ nhớ; hai mặt được OCR
tuần tự. Queue đầy/hết thời gian chờ trả lỗi bận, không nhân bản model theo request.

Quy tắc chuẩn hóa tên cơ quan cấp nằm trong
`identification_demo/config/issuing_authority_rules.json`. Chỉ chuẩn hóa dựa
trên chữ đọc được; không điền một cơ quan mặc định.

## API

[Contract đầy đủ của API chính](docs/API_CONTRACT.vi.md).

```bash
curl --fail-with-body http://localhost:8888/api/cccd/extract \
  -F 'front_image=@/path/to/front.jpg' \
  -F 'back_image=@/path/to/back.jpg' \
  -F 'quality=high'
```

Mỗi ảnh JPG/PNG tối đa 10 MiB và 20 megapixel. Hai ảnh được kiểm tra trước
khi xử lý. QR trên hai mặt xung đột hoặc gửi sai mặt được báo HTTP 422.

Response có schema cố định 12 trường: `id_number`, `old_id_number`, `name`,
`date_of_birth`, `gender`, `nationality`, `place_of_origin`, `place_of_residence`,
`issue_date`, `issuing_authority`, `identifying_features`, `expiry_date`.
CMND cũ chỉ lấy từ QR; `identifying_features` giữ trống để tương thích contract.
Ngày được chuẩn hóa `DD/MM/YYYY`. Ngày hết hạn phải gắn với nhãn; thẻ không
thời hạn được ghi rõ trong `expiry.expiry_type`.

Metadata gồm `field_sources`, `missing_fields`, trạng thái từng mặt QR/OCR,
`warnings`, `ocr_engine`, `expiry`, `timings_ms`, `request_timings_ms` và
`request_id`. API chính không trả QR thô hoặc toàn bộ chữ OCR. Giao diện gửi
`include_back_ocr=true` để hiển thị chữ mặt sau của cùng lượt đọc, không phát
sinh lượt OCR riêng.

| Endpoint | Mục đích |
| --- | --- |
| `POST /api/cccd/extract` | Hai mặt, QR ưu tiên và OCR bổ sung |
| `POST /api/qr/cccd` | Đọc QR một ảnh, multipart `file` |
| `POST /api/ocr/cccd` | OCR CPU một ảnh, `file`, `side=auto/front/back` |
| `GET /api/health` | Trạng thái QR, CPU, GPU và hàng đợi |
| Worker: `GET /api/v1/health/ready` | Readiness GPU |
| Worker: `POST /api/v1/extract` | OCR một mặt, dùng nội bộ |

HTTP 200 có trạng thái `complete`, `partial` hoặc `no_text` ở API chính.
Lỗi ảnh trả 413/422; queue bận trả 429 với `Retry-After: 5`; bộ xử lý lỗi và
chưa có dữ liệu trả 503. Một mặt lỗi vẫn giữ thông tin từ QR/mặt còn lại.

## Thử nghiệm MRZ và căn chỉnh bốn góc

DocAligner + FastMRZ đã được đưa vào API GPU chính qua công tắc **Bật MRZ và căn chỉnh ảnh**, mặc định bật trong Compose. Tắt công tắc để dùng GPU QR + OCR trước đó; trình duyệt nhớ lựa chọn. API có `use_mrz=true/false`; mức CPU `medium` không dùng MRZ. Worker MRZ mất kết nối sẽ dùng lại GPU cũ và ghi rõ `gpu_pipeline.fallback=true`. Không chuyển sang CPU.

Xem [cách bật/tắt và quay lại luồng cũ](docs/GPU_MRZ_SWITCH.vi.md). Tab **Thử nghiệm · MRZ & căn chỉnh** vẫn giữ để xem ảnh bốn góc, MRZ thô và chi tiết đối chiếu, dùng chung worker. [Giới hạn MRZ và mô hình](docs/CCCD_EXPERIMENT.vi.md).

## Chạy trực tiếp và kiểm thử

Cần Git, Python 3.10–3.12 và thư viện hệ thống `libgomp1`, `libglib2.0-0`, `libgl1`.

```bash
bash scripts/setup.sh
.venv/bin/python -m uvicorn identification_demo.web:app --host 0.0.0.0 --port 8888
```

```bash
python -m pip install -r requirements-dev.txt -r requirements-qr.txt
python -m pytest -q
```

Đo luồng chính bằng ảnh của bạn, script chỉ in thống kê:

```bash
python scripts/benchmark-pipeline.py --url http://localhost:8888 \
  --front /path/to/front.jpg --back /path/to/back.jpg \
  --quality high --requests 5 --concurrency 2
```

Xuất crop và chữ từ đúng pipeline chính: [hướng dẫn](docs/OCR_CROP_EXPORT.vi.md).
Các báo cáo thử nghiệm cũ trong `docs/` và `docs/benchmarks/` được giữ làm bằng
chứng lịch sử; các API/tab thử nghiệm được mô tả trong đó đã gỡ khỏi bản 0.8.0.
