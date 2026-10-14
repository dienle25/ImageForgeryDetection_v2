# CPU_BENCHMARK_VI — Thời gian chạy và bộ nhớ trên CPU

_Đo bằng chính mã ứng dụng (ONNX Runtime CPU, không GPU). “Phân tích” = đọc ảnh + 3 bộ phát hiện; “đầu-cuối” = phân tích + tạo mọi file tải về (PNG/NPY/JSON) như web/CLI; “web” = gửi ảnh qua HTTP tới máy chủ cục bộ đến khi nhận kết quả (và tải ZIP). Ảnh thử tổng hợp (không phải ảnh người dùng); thời gian không phụ thuộc nội dung ảnh, trừ bộ khuôn mặt (chạy theo số mặt, tối đa 3)._

## 1. Máy đo
| Máy | CPU | RAM | Hệ điều hành / Python | Bằng chứng |
|---|---|---|---|---|
| Máy build (cloud) | Intel Xeon @ 2,10 GHz, **2 vCPU** | 8 GB | Linux x86_64, CPython 3.13.16, onnxruntime 1.29.0 | `reports/benchmark/linux_cloud_2vcpu_v2.json`, `..._v1_baseline.json` |
| Laptop người dùng | AMD Ryzen 7 4800H, 8 nhân / 16 luồng | 7,6 GB | Windows 11, CPython 3.13.16 embeddable của gói | `reports/evidence/windows_selftest/` (bản RC) |

## 2. Kết quả v2 (cấu hình bàn giao)
| Chỉ số | Linux 2 vCPU (10 lần, sau 1 lần làm nóng) | Windows Ryzen 7 4800H (bản RC, 3 lần) |
|---|---|---|
| Nạp + kiểm SHA-256 mô hình | 8,4 s (đĩa đã có cache) | 17,0 s (lượt 2); 26,5 s (lượt 1, máy vừa giải nén) |
| Phân tích ảnh 0,25 MP | p50 1,70 s · p95 1,85 s | — |
| Phân tích ảnh 2 MP (1920×1080) | p50 1,90 s · p95 2,14 s | 2,67–3,03 s |
| Phân tích ảnh 12 MP (4000×3000) | p50 2,49 s · p95 2,80 s | 2,91–3,06 s |
| Ảnh chân dung có 1 mặt | p50 2,47 s (bộ khuôn mặt 0,78 s) | — |
| Đầu-cuối 2 MP (kèm file tải về) | p50 3,05 s · p95 3,29 s | — |
| Đầu-cuối 12 MP | p50 9,37 s · p95 10,20 s | — |
| Web: gửi ảnh → nhận kết quả, 2 MP | p50 3,28 s · p95 3,47 s (thêm tải ZIP: 3,93 s) | — |
| Web: gửi ảnh → nhận kết quả, 12 MP | p50 10,8 s · p95 12,2 s (thêm tải ZIP 78 MB: 14,5 s) | — |
| RAM tiến trình sau khi nạp mô hình | 1,7 GB | chưa đo (bản self-test cũ) |
| RAM đỉnh (đang xử lý ảnh 12 MP, giữ kết quả web) | 3,8 GB | chưa đo |

Phân bổ thời gian (2 MP, Linux): Mesorch 1,58 s · ảnh AI 0,21 s · tìm mặt (không có mặt) 0,06 s. Với ảnh lớn, phần lớn thời gian đầu-cuối là ghi PNG độ phân giải gốc (ảnh 12 MP sinh ~85 MB file); CLI có `--no-images` để chỉ ghi điểm khi sàng lọc hàng loạt.

**Đối chiếu mục tiêu `QUALITY_TARGETS.yaml`:** nạp mô hình ≤ 30 s — **đạt** (8,4 s Linux; 17–26,5 s Windows); ảnh 12 MP đầu-cuối p95 ≤ 60 s — **đạt** (10,2 s Linux 2 vCPU).

## 3. So với v1 (PSCC-Net) trên cùng máy Linux
| Ảnh | v1 phân tích p50 | v2 phân tích p50 (3 bộ phát hiện) |
|---|---|---|
| 0,25 MP | 3,01 s | 1,70 s |
| 2 MP | 2,72 s | 1,90 s |
| 12 MP | 3,67 s | 2,49 s |
v1 đo bằng mô hình PSCC-Net đã chuyển ONNX chạy trong ứng dụng v2 (`research/configs/config_v1_baseline.json`); bản v1 gốc dùng PyTorch + Streamlit, không đo lại.

## 4. Ghi chú
- Lần mở đầu tiên sau khi giải nén/khởi động máy chậm hơn (đọc ~1 GB mô hình từ đĩa, phần mềm diệt virus quét file mới).
- Máy 8 GB RAM: không chạy đồng thời `RUN_APP.bat` và `PHAN_TICH_THU_MUC.bat`/`KIEM_DINH_CONG_TY.bat` (mỗi cái nạp một bộ mô hình). Ứng dụng web giữ tối đa 8 kết quả gần nhất và ~600 MB trong RAM.
- Số luồng CPU: mặc định để ONNX Runtime tự chọn (`runtime.threads = 0`); có thể đặt `--threads N` cho CLI.
- Thời gian trong `reports/test/*` (final test) đo khi máy cloud đang chạy việc khác, chỉ để tham khảo; dùng bảng trên cho thời gian.
- Lệnh đo lại: `python tools/benchmark.py --runs 10 --http-runs 5 [--face-image ANH_CO_MAT.jpg]`; trên Windows: `SELF_TEST.bat` (mục `cpu_timing`, có RAM).
