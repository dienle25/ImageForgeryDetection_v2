# STATUS — Image Forgery Detection v2

_Cập nhật: 2026-10-12 ~21:20 UTC (13/10 04:20 giờ VN) — trạng thái lúc đóng gói bàn giao._

## Đã xong (có bằng chứng)
- **Nguyên nhân gốc audit** (F01–F09) và cách xử lý: `docs/ROOT_CAUSE_REPORT_VI.md`; trước/sau trên cùng final test + 64 probe audit: `docs/BEFORE_AFTER_REPORT_VI.md`.
- **Nghiên cứu/chọn mô hình trên dev** (≥ 3 phương án mỗi bài toán, chạy đối chứng PSCC-Net / Mesorch / CAT-Net, Community Forensics / SPAI, CLIP-LN): `docs/RESEARCH_REPORT_VI.md`.
- **Final test một lần** với cấu hình khóa (4 940 ảnh công khai): `docs/QUALITY_REPORT_VI.md`. Quy tắc bật mặc định đăng ký trước → Mesorch **THỬ NGHIỆM** (Korus 0,622), Community Forensics và YuNet/CLIP **đạt**. Mục tiêu: 16/19 chất lượng + 2/2 CPU. Tương đương mã cuối ↔ lượt test: 150/150 ảnh trùng điểm từng bit (R01).
- **Độ bền**: nén/thu nhỏ làm giảm phát hiện ảnh AI 6–11 điểm %; chỉnh sáng/tương phản/cắt không tăng báo nhầm; xoay 90° không EXIF làm mất phát hiện khuôn mặt (E13).
- **Ứng dụng**: web UI tiếng Việt, CLI (`--no-images`), self-test, **bộ kiểm định trên ảnh có nhãn của công ty** (`KIEM_DINH_CONG_TY.bat`), cảnh báo khi cấu hình khác bản đã đánh giá, 2 điểm vận hành đã đánh giá (chuẩn/nghiêm).
- **Kiểm thử**: Linux pytest 50/50, self-test 12/12, UI e2e PASS; Windows 11 (bản RC) self-test 10/10 (`docs/TEST_REPORT_VI.md`). CPU: `docs/CPU_BENCHMARK_VI.md`.

## Chưa xong / cần người dùng hoặc công ty
1. **Chạy `SELF_TEST.bat` của gói cuối trên Windows** (mã sau RC chưa chạy trên Windows).
2. **Kiểm định trên ảnh có nhãn của công ty** (`KIEM_DINH_CONG_TY.bat`, README mục 6) trước khi dùng cho quyết định liên quan tiền; chọn điểm vận hành chuẩn/nghiêm theo chi phí lỗi.
3. **Pháp lý** xem giấy phép trọng số/dữ liệu huấn luyện (`LICENSES_VI.md`) trước khi dùng thương mại hoặc phân phối lại.

## Giới hạn đã biết
- Chỉnh sửa tay chuyên nghiệp gần như không phát hiện được; vùng sửa nhỏ, generator AI mới (flux.2-klein), ảnh AI bị nén lại dễ bị bỏ sót.
- Khoảng 9 % ảnh thật (gộp 5 nguồn công khai) bị “Nghi vấn” ở điểm vận hành chuẩn; cao hơn trên ảnh nhỏ/nén nhiều (COCO 13,5 %).
- Ngoài phạm vi: ảnh chụp màn hình, đồ họa, tài liệu, video, metadata.
