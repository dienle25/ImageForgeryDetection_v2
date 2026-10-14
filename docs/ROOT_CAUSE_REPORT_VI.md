# ROOT_CAUSE_REPORT_VI — Nguyên nhân gốc các lỗi audit F01–F09 và cách xử lý

_Phạm vi: gói audit `Image_Forgery_Detection_CPU_Audit_20261012.zip` (v1, PSCC-Net). Mọi kết luận có mã thí nghiệm trong `EXPERIMENT_LOG.jsonl`. Số liệu chất lượng cuối cùng nằm ở `docs/QUALITY_REPORT_VI.md`._

## 0. Tái hiện trước khi sửa
| Bước | Kết quả | Bằng chứng |
|---|---|---|
| Kiểm kê ZIP audit | 269/269 file khớp `MANIFEST.sha256`; pytest baseline 22/22 PASS (Linux) | E00, `reports/evidence/baseline_pytest_linux.xml` |
| Chuyển PSCC-Net sang ONNX không cần torch | |Δscore| ≤ 3,9·10⁻⁷, |Δmask| ≤ 4·10⁻⁶ so với PyTorch | E01–E02 |
| Chạy lại 64 probe audit trên ONNX | |Δscore| ≤ 1,9·10⁻⁶ → F01–F04 **tái hiện y hệt**; lỗi thuộc mô hình + tiền xử lý, không do cài đặt torch | E03, `reports/evidence/repro_audit_probes.json` |

## 1. F01 — điểm nhảy theo kích thước suy luận; F02 — hai head mâu thuẫn
**Nguyên nhân gốc (đã xác minh từ mã chính thức):** loader huấn luyện của PSCC-Net (`utils/load_tdata.py`) resize **mọi** ảnh huấn luyện (gốc lẫn sửa) về **256×256 bicubic**, không giữ tỉ lệ (E04). v1 suy luận ở cạnh dài 512 (bilinear, giữ tỉ lệ, đệm) → ảnh đầu vào nằm ngoài phân phối huấn luyện về tỉ lệ/nội suy; đầu phân loại (DetectionHead) và đầu bản đồ (NLCDetection) phản ứng khác nhau với lệch phân phối → F01 (điểm đổi theo `max_side`) và F02 (điểm toàn ảnh thấp nhưng bản đồ đỏ toàn ảnh).

**Thí nghiệm (E05 chẩn đoán, E06 trên dev độc lập 418 ảnh camera):**
- 256×256 bicubic (đúng phân phối huấn luyện) cải thiện **định vị**: AUC pixel 0,59 → ~0,77–0,80; kích hoạt nhầm trên ảnh gốc 49 % → 19 %.
- Nhưng **phát hiện mức ảnh vẫn ngang may rủi**: AUROC 0,51–0,56 với mọi chính sách (CI chứa 0,5). Trên Korus dev: AUROC 0,494 (E08).
- ⇒ Sửa tiền xử lý **không đủ**: bản thân PSCC-Net không tách được ảnh sửa/ảnh gốc trên ảnh camera thật. Đây là giới hạn mô hình, không phải lỗi cài đặt.

**Xử lý trong v2:** thay PSCC-Net bằng **Mesorch** (chọn trên dev, E07–E08):
- tiền xử lý **cố định** đúng pipeline kiểm thử chính thức (resize 512×512 `cv2.INTER_LINEAR` + chuẩn hóa ImageNet) → không còn núm `max_side` làm đổi kết quả; cùng bytes luôn ra cùng điểm;
- điểm toàn ảnh = **giá trị lớn nhất của chính bản đồ** 512×512 (Mesorch không có head ảnh) → F02 không thể xảy ra theo cấu trúc (bản đồ đỏ ⇒ điểm cao);
- PSCC-Net được giữ dạng ONNX, **tắt mặc định**, chỉ để so sánh trước/sau.

## 2. F03 — vùng sửa nhỏ bị bỏ sót
**Nguyên nhân:** mọi mô hình ở đây nhìn ảnh đã thu về 256–512 px; vùng sửa < ~1 % diện tích còn vài pixel sau thu nhỏ. Mesorch trên dev kiểm soát: AUROC vùng nhỏ 0,889 so với vùng lớn 0,983 (E07). Trên ảnh sửa tay chất lượng cao (Korus, vùng sửa trung vị < 1 %), mọi ứng viên gần may rủi (E08).
**Xử lý:** không có cách sửa “miễn phí” trên CPU; báo cáo theo nhóm kích thước, cảnh báo phạm vi trong UI/README. Hướng tiếp: suy luận theo ô (tile) độ phân giải gốc — cần đánh giá riêng vì lệch phân phối huấn luyện và tốn CPU (chưa làm).

## 3. F04 — báo nhầm trên đồ họa vẽ (procedural)
**Nguyên nhân:** miền dữ liệu khác hẳn ảnh chụp (mảng màu phẳng, cạnh sắc); điểm PSCC-Net bão hòa ~1,0. Đổi ngưỡng không tạo ra khả năng phân biệt.
**Xử lý:** khai báo **ngoài phạm vi** (đồ họa, ảnh chụp màn hình, tài liệu) trong TASK_CONTRACT, UI và README; đo lại trên v2 ở `docs/BEFORE_AFTER_REPORT_VI.md` (chỉ chẩn đoán).

## 4. F05 — lọt trùng khi đổi encoding
Đã sửa ở v1 (hash pixel sau giải mã). v2 giữ `decoded_rgb_sha256` trong mọi báo cáo và mở rộng kiểm tra rò rỉ: trùng bytes/pixel, gần trùng (dHash + xác minh pixel) và dòng dõi ảnh nền (phát hiện thật 4 nhóm DFF dùng chung ảnh nền giữa dev/test → loại trước khi chấm, P03).

## 5. F06 — CLI để lại kết quả cũ
v2 giữ thiết kế `runs/<run_id>/` + `latest_run.json`, ghi nguyên tử (`.tmp` → `os.replace`), ảnh lỗi có `output_dir: null`; CSV có cột từng detector. Kiểm thử: `tests/test_cli.py`, `ifd.selftest` (chạy lại không ghi đè lượt trước, mã thoát 0/1/2).

## 6. F07 — lỗi bất ngờ trên UI không có traceback
v2 bỏ Streamlit; máy chủ web thư viện chuẩn ghi traceback đầy đủ ra cửa sổ chạy (`log.error(... traceback.format_exc())`) và trả HTTP 500 có thông báo tiếng Việt. Có kiểm thử tiêm lỗi (`tests/test_webapp.py::test_unexpected_error_is_logged`).

## 7. F08 — thiếu quality gate
v2: `TASK_CONTRACT.md` + `QUALITY_TARGETS.yaml` ghi **trước** final test (P01, P02, P03), ngưỡng chọn trên dev theo quy tắc FPR, final test chạy một lần bằng ứng dụng đóng gói; báo đạt/không đạt theo từng mục tiêu.

## 8. F09 — toàn vẹn ZIP
v2: ZIP được kiểm tra CRC toàn bộ, giải nén ra thư mục mới, so SHA-256 từng file với `PACKAGE_MANIFEST.sha256`, chạy tự kiểm tra trên bản giải nén; SHA-256 của ZIP ghi trong báo cáo bàn giao.
