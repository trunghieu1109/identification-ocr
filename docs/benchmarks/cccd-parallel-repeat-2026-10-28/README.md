# Đo lại GPU chuẩn và luồng thử nghiệm song song

Đo ngày 28/10/2026, từ 10:00:00 đến 10:01:23 UTC. Không thay đổi code, model hay khởi động lại dịch vụ để đo.

Cùng 5 cặp ảnh giả lập của bài đo trước, hash đầu vào đã đối chiếu khớp. Mỗi trường hợp warm-up một lần mỗi luồng rồi đo 10 lần mỗi luồng, tổng **100 lượt đo + 10 warm-up**. Chạy tuần tự qua localhost, đảo thứ tự hai API mỗi vòng. Cùng RTX 5060 8 GB, DocAligner/FastMRZ CPU và worker OCR GPU hiện có.

Mỗi số dưới đây là **trung bình cộng thời gian HTTP cho toàn bộ cặp mặt trước + mặt sau**, gồm nhận hết response; không tính đường truyền Internet của người dùng.

| Tình huống | GPU chuẩn | Thử nghiệm song song | Tăng |
| --- | ---: | ---: | ---: |
| Ảnh rõ, có MRZ | 0,682 s | 0,776 s | 0,094 s |
| Hai mặt mờ nhẹ | 0,718 s | 0,825 s | 0,107 s |
| Mặt sau nghiêng/có nền | 0,813 s | 0,867 s | 0,053 s |
| Không có MRZ | 0,621 s | 0,834 s | 0,213 s |
| QR đọc được | 0,659 s | 0,775 s | 0,116 s |
| **Trung bình 50 lượt mỗi luồng** | **0,699 s** | **0,816 s** | **0,117 s (+16,74%)** |

Lần trước trung bình 25 lượt mỗi luồng: GPU chuẩn 0,713 s, thử nghiệm song song 0,823 s, tăng 0,110 s. Lần đo lại cho kết quả gần lần trước; độ trễ tăng khác nhau theo tình huống, cao nhất khi ảnh không có MRZ.

Tất cả 100 lượt đo trả HTTP 200, không có stage OCR lỗi. Thời gian chờ queue GPU trung bình 0,10 ms ở luồng chính và 0,19 ms ở luồng thử nghiệm. Số trường khớp đáp án giữ như lần trước: ảnh rõ 9/10 ở cả hai luồng (cùng lệch ngày cấp), các tình huống mờ/nghiêng/không MRZ 10/10, QR đọc được 11/11. Đây là ảnh giả lập, chưa suy ra độ chính xác hoặc độ trễ trên tập ảnh thật, tải nhiều người dùng và ảnh điện thoại kích thước lớn.

- [Tóm tắt số liệu](summary.json)
- [Toàn bộ mẫu đo và timeline](results.json)
- [Phần cứng, image và model readiness](environment.json)
- [Bài đo và giải thích điều phối trước đó](../../CCCD_PARALLEL_BENCHMARK.vi.md)

Lệnh chạy:

```bash
python scripts/benchmark-cccd-flows.py \
  --url http://127.0.0.1:8888 \
  --dataset /tmp/cccd-flow-benchmark/dataset.json \
  --requests 10 --warmup 1 \
  --output docs/benchmarks/cccd-parallel-repeat-2026-10-28/results.json
```
