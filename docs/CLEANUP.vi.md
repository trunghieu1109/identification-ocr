# Thu gọn hệ thống OCR — 28/10/2026

Sau đợt thu gọn này, một worker/tab thử nghiệm **DocAligner + FastMRZ** được bổ sung theo yêu cầu riêng; xem [hướng dẫn](CCCD_EXPERIMENT.vi.md). Các worker/model thử nghiệm cũ bên dưới vẫn đã gỡ. Luồng chính giữ nguyên.

Bản 0.8.0 giữ luồng đọc QR bằng WeChatQRCode, sau đó bổ sung bằng namlhn GPU
(Paddle detector + DocU Seq2Seq) hoặc CPU `medium`. Contract của API chính
`POST /api/cccd/extract` và hai API QR/OCR riêng được giữ lại.

## Các thành phần đã gỡ

- Hai worker DocU độc lập `ocr-gpu`, `ocr-gpu-regions`, các biến thể DocU v2/v3,
  phục hồi/mở rộng vùng và những lượt thử xoay thẻ của chúng.
- API/tab so sánh OCR, so sánh QR, thử ngày hết hạn riêng và luồng WeChat song song cũ.
- Nhánh namlhn GPU legacy/Transformer và bộ giải mã CUDA riêng của nó.
- QRDet/ZXing, Restormer/SwinIR, model segmentation/detection YOLO và model xoay thẻ.
- Dockerfile GPU/benchmark cũ cùng sự phụ thuộc image benchmark của worker chính.
- Dependency, script và test chỉ dành cho các nhánh đã gỡ. Test parser ngày hết
  hạn, cơ quan cấp, số căn cước và ánh xạ trường được giữ lại.

## Model và image

GPU có Dockerfile độc lập từ Python slim, cài đúng Torch 2.7.1/CUDA 12.8,
Paddle GPU 3.2.2/CUDA 12.9 và PaddleOCR 3.7.0 như runtime trước.
Chỉ sao chép checkpoint `vgg_seq2seq.pth` từ image nguồn đã pin digest.
Mã DocU được giới hạn ở recognizer và các kiểu dữ liệu của nó; không import
pipeline YOLO qua `core.__init__`.

Config đặt `cnn.pretrained: false` vì toàn bộ CNN đã được ghi đè bằng checkpoint
Seq2Seq. Nhờ đó không tải/lưu thêm checkpoint ImageNet VGG19 khoảng 549 MiB.
Detector `PP-OCRv6_medium_det` vẫn được giữ trong volume Paddle. Hai recognizer
Paddle cũ trong volume GPU được gỡ vì luồng GPU không gọi chúng.

WeChat giữ bốn model detector/super-resolution; model QRDet không được tải lúc
build nữa. CPU `medium` giữ detector/recognizer Paddle và VietOCR Transformer.
Checkpoint Seq2Seq thử nghiệm trong checkout cục bộ cùng cache Restormer/SwinIR
đã được dọn. CPU image loại checkpoint và thư mục Git ngay trong lớp clone,
tránh giữ trọng số thừa trong các lớp image bên dưới.

Các image/cache Docker được dọn theo nhãn project hoặc ID đã xác định thuộc OCR.
Hai volume cache của worker thử nghiệm được gỡ. Các báo cáo/benchmark cũ trong
`docs/` được giữ và đánh dấu tài liệu lịch sử.

Image GPU chính giảm từ **20.096.633.012 byte (~20,1 GB)** xuống
**12.136.151.508 byte (~12,1 GB)**, giảm khoảng **39,6%**. Compose chỉ còn
hai service và bốn volume cache model CPU/Paddle. Image web/CPU giảm từ
**~3,70 GB** xuống **~3,30 GB**; lớp clone chỉ còn khoảng 134 kB và không
chứa checkpoint cũ.

## Xác minh

- 466 test hồi quy đạt trong môi trường OCR có Paddle/VietOCR.
- Hai cặp ảnh giả lập của benchmark: đủ 12 trường trong schema, trạng thái
  `complete`, các giá trị trả về khớp bản trước khi dọn.
- Worker GPU thu gọn xử lý bốn ảnh giả lập khi tắt mạng, không import pipeline
  DocU đầy đủ và không nạp backbone VGG19 riêng.
- OCR CPU `medium` thực tế xử lý thành công cả hai mặt.
- Trình duyệt thật: không lỗi JavaScript, hiển thị 12 trường/JSON, không tràn
  ở chiều rộng 390 px; chỉ gọi health và API trích xuất chính.
- API/tab thử nghiệm trả 404; health không kiểm tra worker thử nghiệm nữa.

Ảnh kiểm tra là dữ liệu giả lập; đây là kiểm tra hồi quy, không phải đánh giá
độ chính xác trên tập căn cước thực tế.

## Giải phóng dung lượng thực sau kiểm kê

Đợt kiểm kê bổ sung cùng ngày phát hiện các record BuildKit `pulled from`
của image DocU nguồn và `from local` của các image OCR cũ đều báo `0B`.
Những record này vẫn giữ các lớp CUDA/model trong `overlay2`, nên việc gỡ tag
image và cache build có dung lượng hiển thị chưa giải phóng hết đĩa.

Đã đối chiếu metadata snapshot/chain với image hiện tại, rồi dùng
`docker buildx prune --all` theo description nguồn DocU hoặc ID record OCR
cụ thể. Không xóa trực tiếp thư mục `overlay2`. Tổng cộng 55 record nguồn/cũ
được gỡ ở đợt này; các cache của dự án khác được giữ lại.

- Trước đợt bổ sung: khoảng **8,1 GiB trống**, sử dụng **95%**.
- Sau đợt bổ sung: khoảng **41,2 GiB trống**, sử dụng **70%**.
- Thu hồi thêm khoảng **33 GiB thực**, đo qua filesystem; số `Total` do
  Docker in ra khi prune record `0B` không phản ánh đầy đủ dung lượng này.
- Hai service chính vẫn healthy; API health xác nhận QR và GPU sẵn sàng.
