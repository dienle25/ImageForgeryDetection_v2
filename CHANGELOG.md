# CHANGELOG

## 2.0.0 — 2026-10-12/13 (bàn giao sau audit)
### Thay đổi chất lượng (có bằng chứng, xem EXPERIMENT_LOG.jsonl)
- **Thay PSCC-Net bằng Mesorch** cho chỉnh sửa cục bộ (chọn trên dev: ctrl_mei dev AUROC 0,945 vs PSCC ≤ 0,56; Korus dev 0,58 vs 0,49). PSCC-Net giữ lại dạng ONNX, tắt mặc định, chỉ để đối chứng.
- **Thêm bài toán ảnh sinh hoàn toàn bằng AI**: Community Forensics ViT-S/16@384 (so sánh với SPAI: kém hơn và chậm ~100× trên CPU).
- **Thêm bài toán deepfake/hoán đổi khuôn mặt**: YuNet + CLIP ViT-L/14 LN-tuning (yermandy).
- Ngưỡng `low`/`high` của từng detector chọn trên dev theo quy tắc FPR ≤ 5 % / ≤ 20 % trên **từng** nguồn âm; trạng thái ghi trong `config.json`.
- Kết luận ba mức + “Không áp dụng” (không có mặt người) + liệt kê bài toán không kiểm tra.

### Sửa lỗi audit
- F01/F02: tiền xử lý cố định theo pipeline chính thức; điểm ảnh = max của chính bản đồ (không còn hai head mâu thuẫn).
- F05: hash pixel + kiểm tra gần trùng/dòng dõi ảnh nền trong đánh giá.
- F06: `runs/<run_id>/`, ghi nguyên tử, CSV có cột từng detector.
- F07: traceback được log; có kiểm thử tiêm lỗi.
- F09: ZIP kiểm CRC + giải nén mới + so SHA-256 từng file + tự kiểm tra trên bản giải nén.

### Ứng dụng
- Bỏ Streamlit/PyTorch khỏi runtime: ONNX Runtime + NumPy + Pillow + OpenCV-headless.
- Web UI tiếng Việt chạy bằng thư viện chuẩn Python, chỉ 127.0.0.1, token phiên, kiểm Host, CSP; nạp mô hình nền, trang hiện tiến độ.
- Gói Windows tự chứa (`runtime\` = CPython 3.13.16 embeddable + wheels), `RUN_APP.bat`, `SELF_TEST.bat`, `PHAN_TICH_THU_MUC.bat`; không cần cài đặt, không cần quyền admin, không cần Internet khi chạy.
- Mọi mô hình được kiểm tra kích thước + SHA-256 trước khi nạp; hỗ trợ ghép mô hình từ nhiều phần có kiểm tra hash.
- `python -m ifd.selftest`: kiểm thử chấp nhận trên máy người dùng (định dạng, lỗi, CLI, web, cổng bận, khớp số với tham chiếu, đo thời gian/RAM).

### Sau khi khóa cấu hình (P04) — không đổi điểm/ngưỡng/kết luận của ảnh (EXPERIMENT_LOG P04b–P08)
- Thông báo lỗi tiếng Việt cho file không phải ảnh (không lộ địa chỉ bộ nhớ Python).
- Self-test: CLI đầy đủ chạy ở tiến trình riêng (không nạp 2 bộ mô hình trong một tiến trình trên máy 8 GB), đo RSS đúng trên Windows, thời gian p50/p95 + end-to-end; thêm mục kiểm tra bộ công cụ kiểm định.
- Nạp mô hình thất bại do thiếu RAM → thông báo hướng dẫn đóng cửa sổ RUN_APP/PHAN_TICH_THU_MUC khác.
- **Bộ công cụ kiểm định trên ảnh có nhãn của công ty** (`KIEM_DINH_CONG_TY.bat`, `python -m ifd.company_eval`): chạy tại máy, báo TP/TN/FP/FN, precision, recall, FPR, FNR, AUROC (CI 95 %), AP theo từng bộ phát hiện, cấp ứng dụng và theo loại; gộp ảnh trùng; không tự đổi ngưỡng.
- **Cảnh báo cấu hình đã đổi**: nếu ngưỡng/bộ phát hiện khác cấu hình đã được đánh giá (SHA-256 trong `reports/validation_summary_vi.json`), mọi kết quả và trang web hiện cảnh báo số liệu chất lượng không còn áp dụng.
- Web: bản đồ nhiệt/ảnh chồng lớp chỉ dựng một lần cho cả file tải về và ảnh xem trước (trước đây 4 lần với ảnh 12 MP); kết quả giữ trong RAM tối đa 8 lần và ~600 MB.
- Sau final test: bộ chỉnh sửa cục bộ (Mesorch) mang nhãn **THỬ NGHIỆM** trên giao diện vì không đạt quy tắc bật mặc định trên Korus (áp dụng cơ học, `research/make_validation_summary.py`); thẻ kết quả hiện dòng tóm tắt ngắn, phần “Phạm vi kiểm tra” hiện số liệu đầy đủ (tỷ lệ bắt / báo nhầm, AUROC).
- Hai điểm vận hành đã đánh giá (chuẩn, nghiêm): giao diện ghi đang dùng điểm nào; `RUN_APP.bat --config research\configs\config_strict.json` để dùng profile nghiêm.
- Tăng tốc dựng bản đồ màu (cùng kết quả từng bit) + PNG nén nhanh (pixel không đổi): ảnh 12 MP đầu-cuối 19,9 s → 9,4 s (Linux 2 vCPU); CLI `--no-images` cho sàng lọc hàng loạt.

## 1.x — trước audit
PSCC-Net + Streamlit (xem gói audit gốc).
