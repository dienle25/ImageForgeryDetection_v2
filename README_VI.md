# Kiểm tra ảnh: chỉnh sửa cục bộ · ảnh sinh AI · deepfake khuôn mặt (v2, CPU, chạy tại máy)

Ứng dụng phân tích **một ảnh JPEG/PNG** (hoặc cả thư mục) **ngay trên máy của bạn**, không cần GPU, không gửi ảnh ra Internet.
Kết quả là **gợi ý cho người kiểm tra**, không phải kết luận pháp lý và không phải “máy phát hiện mọi ảnh giả”.

## 1. Bắt đầu nhanh (Windows 10/11, 64-bit)
1. Giải nén toàn bộ file ZIP vào một thư mục (ví dụ `D:\KiemTraAnh`). Không chạy trực tiếp bên trong file ZIP.
2. Bấm đúp **`RUN_APP.bat`**. Cửa sổ đen hiện địa chỉ dạng `http://127.0.0.1:xxxxx/?t=...` và trình duyệt tự mở.
   Lần đầu nạp mô hình mất khoảng 30–120 giây; trang web hiện “Đang nạp…” rồi tự sẵn sàng.
3. Kéo thả ảnh (hoặc bấm chọn) → **Phân tích ảnh** → xem kết luận, bản đồ nhiệt, khung khuôn mặt → **Tải toàn bộ (ZIP)** nếu cần lưu.
4. Đóng cửa sổ đen để tắt ứng dụng.

Không cần cài Python: gói có sẵn thư mục `runtime\` (Python 3.13 bản embeddable + thư viện). Nếu thư mục này bị xóa, `RUN_APP.bat` sẽ tạo `.venv` từ Python 3.10–3.13 đã cài trên máy.

**Phân tích cả thư mục:** kéo thả thư mục ảnh vào **`PHAN_TICH_THU_MUC.bat`**. Kết quả nằm ở `outputs\runs\<mã lượt chạy>\` (mỗi lần chạy một thư mục riêng, không ghi đè lần trước) gồm `results.csv` (mở bằng Excel), `results.json` và thư mục kết quả của từng ảnh.

**Tự kiểm tra máy:** bấm đúp **`SELF_TEST.bat`** (vài phút). Báo cáo PASS/FAIL nằm ở `reports\selftest\...\selftest.txt`.

**Kiểm định trên ảnh có nhãn của công ty:** xem mục 6 — kéo thả thư mục dữ liệu vào **`KIEM_DINH_CONG_TY.bat`**.

Yêu cầu máy: Windows 10/11 64-bit, CPU x64 (khuyến nghị đời 2015 trở lên, có AVX2), **RAM ≥ 8 GB** (ứng dụng dùng khoảng 2,5–3,5 GB khi đã nạp mô hình), ~2 GB đĩa trống. Nếu báo thiếu DLL, cài *Microsoft Visual C++ Redistributable 2015–2022 (x64)* từ trang của Microsoft. Trên máy 8 GB **không chạy đồng thời** `RUN_APP.bat` và `PHAN_TICH_THU_MUC.bat` (mỗi cái nạp một bộ mô hình riêng ~2,5–3,5 GB).

## 2. Ứng dụng kiểm tra gì
| Bài toán | Mô hình (giấy phép mã) | Đầu ra |
|---|---|---|
| **Chỉnh sửa cục bộ**: ghép ảnh, sao chép vùng, xóa/thay thế (kể cả inpainting AI) | Mesorch, AAAI 2025 (MIT) | điểm toàn ảnh + **bản đồ nhiệt** vùng nghi vấn |
| **Ảnh sinh hoàn toàn bằng AI** | Community Forensics ViT-S/16, CVPR 2025 (MIT) | chỉ điểm toàn ảnh (không có bản đồ vùng) |
| **Deepfake / hoán đổi khuôn mặt** | YuNet (phát hiện mặt) + CLIP ViT-L/14 tinh chỉnh LN trên FF++ (MIT) | điểm từng khuôn mặt (tối đa 3 mặt lớn nhất); không có mặt → “không áp dụng” |

Mỗi bài toán cho một trong ba mức: **Chưa thấy dấu hiệu rõ** · **Cần kiểm tra thêm** · **Nghi vấn**. Kết luận chung lấy mức cao nhất và luôn ghi rõ bài toán nào **không áp dụng / không kiểm tra**.

## 3. Đọc kết quả cho đúng
- Điểm là **điểm thô của mô hình**, không phải xác suất, không phải “% ảnh bị sửa”.
- **Chưa thấy dấu hiệu rõ ≠ ảnh thật.** Ảnh nén mạnh/thu nhỏ, vùng sửa rất nhỏ, chỉnh sửa tay tinh vi, hoặc generator AI mới có thể không bị phát hiện.
- Bản đồ nhiệt dùng **thang màu cố định 0→1** cho mọi ảnh (không chuẩn hóa theo từng ảnh): đỏ = mô hình nghi vấn, **chưa phải vùng sửa đã xác minh**.
- Ngưỡng được chọn trên dữ liệu phát triển công khai/kiểm soát (xem `docs/QUALITY_REPORT_VI.md`); **chưa được kiểm chứng trên dữ liệu của công ty**.
- Trên final test công khai (một lần, cấu hình khóa): bộ chỉnh sửa cục bộ gắn **THỬ NGHIỆM** — bắt 78 % ảnh ghép/copy-move kiểu script nhưng chỉ ~10 % ảnh sửa tay chuyên nghiệp; bộ ảnh AI bắt ~48 % ảnh của 10 generator mới (báo nhầm 0,7 % ảnh thật); bộ khuôn mặt bắt 81 % ảnh hoán đổi mặt (báo nhầm 4,2 %). Gộp mọi nguồn, khoảng **9 % ảnh thật bị “Nghi vấn”** và 35 % bị “Cần kiểm tra thêm” trở lên — hãy coi kết quả là danh sách ưu tiên cho người thẩm định.

**Ngoài phạm vi (dễ báo nhầm hoặc bỏ sót):** ảnh chụp màn hình, đồ họa/ảnh vẽ, tài liệu scan, ảnh chủ yếu là chữ; video; sửa metadata; ảnh thật gắn sai chú thích; ảnh rất nhỏ (< 64 px) hoặc nén rất nặng.

## 4. File kết quả (trong ZIP tải về hoặc thư mục CLI)
| File | Nội dung |
|---|---|
| `result.json` | báo cáo đầy đủ: điểm/ngưỡng/kết luận từng bài toán, mô hình + SHA-256, cấu hình + SHA-256, EXIF tham khảo, thời gian |
| `overlay.png`, `heatmap.png`, `mask.png` | chồng lớp, bản đồ nhiệt màu, mask nhị phân (ngưỡng pixel 0,5) — đúng kích thước ảnh hiển thị (đã xoay theo EXIF) |
| `heatmap.npy`, `heatmap_gray16.png` | bản đồ số thực / 16-bit để phân tích tiếp |
| `faces.png` | khung khuôn mặt (đỏ = nghi vấn, cam = cần kiểm tra, xanh = chưa thấy dấu hiệu) |
| `original.png` | ảnh đúng như mô hình đã đọc |

## 5. Dòng lệnh (nâng cao)
```
runtime\python.exe -m ifd "D:\anh" --output "D:\ket_qua" [--recursive] [--threads 4] [--no-images]
runtime\python.exe -m ifd.webapp.server --port 0 [--no-browser]
runtime\python.exe -m ifd.selftest
runtime\python.exe -m ifd.company_eval "D:\du_lieu_co_nhan" [--labels nhan.csv] --output "D:\ket_qua"
```
Mã thoát CLI: 0 = mọi ảnh OK · 1 = có ảnh lỗi (vẫn xử lý ảnh khác) · 2 = lỗi đầu vào/cấu hình/mô hình.
`--no-images`: chỉ ghi điểm/kết luận (`result.json`, `results.csv`), không ghi ảnh bản đồ nhiệt — nhanh hơn nhiều khi sàng lọc hàng loạt ảnh lớn (ví dụ `PHAN_TICH_THU_MUC.bat "D:\anh" --no-images`).

## 6. Kiểm định trên dữ liệu có nhãn của công ty (khuyến nghị trước khi dùng thật)
Số liệu trong `docs/QUALITY_REPORT_VI.md` đo trên dữ liệu **công khai**; chúng **không** cho biết độ chính xác trên ảnh của công ty. Để đo trên ảnh của chính công ty (chạy tại máy, ảnh không rời khỏi máy):
1. Tạo một thư mục, bên trong có `that\` (ảnh thật, chưa chỉnh sửa) và `sua\` (ảnh đã chỉnh sửa/tạo giả, đã được xác minh). Có thể chia thư mục con theo loại, ví dụ `sua\ghep_anh`, `sua\xoa_vat_the`, `sua\anh_ai`, `sua\deepfake_mat`, `that\dien_thoai` — báo cáo sẽ tách theo loại.
2. Kéo thả thư mục đó vào **`KIEM_DINH_CONG_TY.bat`**. Báo cáo HTML tự mở khi xong; tất cả nằm ở `outputs\kiem_dinh\...\bao_cao_<thời điểm>\` (`bao_cao_kiem_dinh.html/.md`, `metrics.json`, `du_doan.csv`).
3. Có thể dùng file nhãn CSV thay cho thư mục: `KIEM_DINH_CONG_TY.bat "D:\du lieu" --labels "D:\nhan.csv"` với cột `file,label,loai,nhom` (`label` 0 = thật, 1 = sửa/giả; `nhom` = mã cảnh/ảnh gốc để các ảnh cùng nguồn không bị tính như độc lập).

Báo cáo gồm TP/TN/FP/FN, precision, recall, FPR, FNR, AUROC (CI 95 %), AP cho từng bộ phát hiện và cho kết luận chung, tách theo loại; ảnh trùng (cùng bytes/pixel) tính một lần, ảnh trùng mà khác nhãn bị loại. Chạy lại cùng thư mục sẽ dùng lại điểm đã tính (không chấm lại). Khuyến nghị **≥ 200 ảnh thật và ≥ 200 ảnh sửa/giả cho mỗi loại quan tâm**, lấy từ đúng nguồn ảnh công ty thường nhận. Công cụ **không** tự đổi ngưỡng; bảng “ngưỡng tham khảo” chọn trên một nửa ảnh thật và kiểm tra trên nửa còn lại, chỉ để thảo luận — muốn áp dụng phải kiểm tra lại trên tập ảnh mới.

## 7. Cấu hình (`config.json`)
- `limits`: dung lượng tối đa (25 MiB), số pixel tối đa (40 MP), cạnh nhỏ nhất (16 px).
- `detectors.<tên>.enabled`: bật/tắt từng bộ phát hiện (ví dụ tắt `face_clip` để tiết kiệm ~1,2 GB RAM).
- **Hai điểm vận hành đã được đánh giá trên final test** (số liệu: `docs/QUALITY_REPORT_VI.md` mục 5): **chuẩn** (`config.json`, mặc định) và **nghiêm** (`research\configs\config_strict.json`: ít báo nhầm hơn khoảng 4 lần, đổi lại bỏ sót nhiều hơn). Dùng profile nghiêm: `RUN_APP.bat --config research\configs\config_strict.json` (tương tự cho `PHAN_TICH_THU_MUC.bat "D:\anh" --config ...` và `KIEM_DINH_CONG_TY.bat "D:\du lieu" --config ...`). Trang web ghi rõ đang dùng điểm vận hành nào.
- `detectors.<tên>.thresholds`: `low`/`high` đã chọn trên dev (trạng thái ghi trong `status`). **Đổi ngưỡng hoặc bật/tắt bộ phát hiện làm mất hiệu lực các số liệu đã báo cáo**: khi đó mọi kết quả (web, CLI, `result.json`) có cảnh báo “cấu hình khác bản đã được đánh giá” (so với SHA-256 ghi trong `reports/validation_summary_vi.json`); `result.json` luôn ghi ngưỡng và SHA-256 cấu hình đã dùng.
- Ứng dụng kiểm tra SHA-256 của mọi mô hình với `models/manifest.json` trước khi nạp; file sai/hỏng bị từ chối.

## 8. Bảo mật & riêng tư
Máy chủ web chỉ nghe ở `127.0.0.1`, cần mã phiên ngẫu nhiên trong địa chỉ, kiểm tra Host header, chính sách CSP chặt; kết quả chỉ giữ trong RAM (tối đa 8 lần gần nhất, tối đa ~600 MB) tới khi bạn tải xuống; muốn lưu lâu dài hãy bấm tải ZIP. Không có telemetry, không tải gì từ Internet khi chạy.

## 9. Tài liệu kèm theo
`TASK_CONTRACT.md` (định nghĩa nhãn/phạm vi) · `QUALITY_TARGETS.yaml` · `MODEL_CARD.md` · `LICENSES_VI.md` · `docs/QUALITY_REPORT_VI.md` · `docs/RESEARCH_REPORT_VI.md` · `docs/DATA_PROTOCOL.md` · `docs/TEST_REPORT_VI.md` · `docs/CPU_BENCHMARK_VI.md` · `docs/ROOT_CAUSE_REPORT_VI.md` · `docs/BEFORE_AFTER_REPORT_VI.md` · `STATUS.md` · `RESUME.md` · `CHANGELOG.md` · `EXPERIMENT_LOG.jsonl`.
