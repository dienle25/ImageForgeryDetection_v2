# TEST_REPORT_VI — Kết quả kiểm thử (PASS / FAIL / NOT RUN) và nơi đã chạy

_Kiểm thử chức năng/độ bền của ứng dụng. Chất lượng phát hiện (độ chính xác) nằm ở `docs/QUALITY_REPORT_VI.md`. Mọi kết quả có file bằng chứng trong `reports/evidence/` hoặc `reports/selftest/` (self-test trên máy người dùng ghi tại chỗ)._

## 1. Tóm tắt
| Hạng mục | Linux x86_64 (máy build, cloud 2 vCPU / 8 GB) | Windows 11 (laptop người dùng: Ryzen 7 4800H, 7,6 GB RAM) |
|---|---|---|
| pytest — mã cuối | **50/50 PASS** (`reports/evidence/pytest_final_linux.xml`) | NOT RUN (gói Windows không kèm pytest; thay bằng self-test) |
| Self-test chấp nhận (`SELF_TEST.bat` / `python -m ifd.selftest`) | **12/12 PASS** (`reports/evidence/linux_selftest_final/`) | Bản RC: lượt 1 9/10 (FAIL mục CLI), lượt 2 **10/10 PASS** · Gói cuối: **NOT RUN lúc đóng gói** — chạy `SELF_TEST.bat` sau khi giải nén (mục 5) |
| Giao diện web tự động (Playwright + Chromium) | **PASS** (`reports/evidence/ui_e2e_linux_final/`) | NOT RUN tự động; người dùng đã mở `RUN_APP.bat` và dùng thử bản RC |
| Toàn vẹn gói ZIP (CRC, SHA-256 từng file, giải nén mới, không có file lạ/cấm) | kiểm tra trên **chính ZIP bàn giao** sau khi đóng gói, kèm self-test từ bản giải nén mới — kết quả ở file `ImageForgeryDetection_v2.zip.verify.json` đi cùng ZIP (không thể nằm trong ZIP vì chứa SHA-256 của chính ZIP) | Bản RC: `GHEP_ZIP.bat` kiểm SHA-256 khớp, giải nén OK |
| Khớp số học Windows ↔ Linux (4 mô hình, 5 ảnh tham chiếu) | tham chiếu | PASS: chênh lệch tối đa 6,6·10⁻⁷ (đầu ra mô hình), 2,1·10⁻⁷ (điểm ảnh) |
| Tương đương mã cuối ↔ lượt final test | **PASS** — 150 ảnh chấm lại, chênh lệch điểm 0,0, 0 khác biệt kết luận (`reports/test/recheck_final_code.json`) | — |

## 2. Ma trận yêu cầu → kiểm thử
| Yêu cầu | Kiểm thử tự động | Linux | Windows (RC) |
|---|---|---|---|
| Đọc JPEG/PNG; ảnh không vuông/lớn (4200×700, 12 MP) | `test_image_io.py`, self-test `input_formats_and_rejections`, `cpu_timing` | PASS | PASS |
| EXIF orientation, kênh alpha, ảnh xám/16-bit, CMYK | `test_image_io.py`, self-test | PASS | PASS |
| File hỏng/cắt cụt, file chữ đổi đuôi .jpg, file rỗng, BMP/GIF/WEBP/TIFF, ảnh quá nhỏ, ảnh quá lớn (MiB/pixel, decompression bomb) | `test_image_io.py`, self-test | PASS | PASS |
| Đường dẫn sai / không tồn tại; thư mục kết quả nằm trong thư mục vào | `test_cli.py`, `test_company_eval.py`, self-test | PASS | PASS |
| Checkpoint hỏng: lật 1 byte, cắt cụt, thiếu file, ghép từ nhiều phần | `test_models_and_parity.py`, `test_model_parts.py`, self-test | PASS | PASS |
| Cấu hình sai: low > high, ngoài [0,1], NaN, loại detector lạ, JSON hỏng | `test_cli.py`, self-test `config_validation` | PASS | PASS |
| Cổng đang bận → báo rõ + mã thoát 2 | `test_webapp.py`, self-test | PASS | PASS |
| Tải xuống: mọi file + ZIP, `Content-Disposition`, kết quả tách theo lần tải lên | `test_webapp.py`, self-test, UI e2e | PASS | PASS |
| Bảo mật web: token phiên, Host header, upload quá cỡ (413) | `test_webapp.py`, self-test | PASS | PASS |
| CLI: `run_id` riêng, không ghi đè lần trước, mã thoát 0/1/2, CSV có cột từng detector, JSON | `test_cli.py`, self-test | PASS | PASS (lượt 2) |
| CLI đầy đủ 3 mô hình trong tiến trình riêng (đường đi của `PHAN_TICH_THU_MUC.bat`) | self-test `cli_folder_full_pipeline_subprocess` (mới) | PASS | NOT RUN trên gói cuối (mục 5) |
| Bộ kiểm định dữ liệu công ty (`KIEM_DINH_CONG_TY.bat`) | `test_company_eval.py` (3), self-test `company_validation_kit` (mới) | PASS | NOT RUN trên gói cuối (mục 5) |
| Cảnh báo khi cấu hình khác bản đã đánh giá; nhận diện profile chuẩn/nghiêm | `test_config_guard.py` (3) | PASS | (mã thuần Python) |
| Thông báo thiếu RAM khi nạp mô hình | `test_errors.py` | PASS | — |
| File tải về giống hệt sau khi tối ưu dựng bản đồ nhiệt một lần | `test_render.py` (2) | PASS | — |
| Khớp số với PyTorch / với máy build | `test_models_and_parity.py` (probe 4 mô hình, PSCC tham chiếu torch), self-test `model_numeric_parity_vs_build_machine` | PASS | PASS |
| Thời gian CPU, RAM | self-test `cpu_timing`, `tools/benchmark.py` | `docs/CPU_BENCHMARK_VI.md` | `docs/CPU_BENCHMARK_VI.md` |

## 3. Các lượt chạy
| Thời điểm (UTC) | Nơi chạy | Phiên bản mã | Kết quả | Bằng chứng |
|---|---|---|---|---|
| 12/10 07:40 | Linux | v1 audit (trước khi sửa) | pytest 22/22 PASS | `reports/evidence/baseline_pytest_linux.xml` |
| 12/10 13:48 | Linux | RC (P04b) | pytest 38/38 PASS | `reports/evidence/pytest_v2_linux.xml` |
| 12/10 13:49 | Linux, Chromium headless | RC (P04b) | UI e2e PASS | `reports/evidence/ui_e2e_linux/` |
| 12/10 13:53 | Linux | RC ZIP | verify PASS (1 338 file, CRC, SHA-256, không file lạ) | `ImageForgeryDetection_v2.zip.verify.json` (bản RC) |
| 12/10 14:07 | Windows 11 laptop | RC | self-test 9/10 — FAIL `cli_runs_and_artifacts` (mã thoát 2; bản self-test cũ không lưu stderr; nhiều khả năng thiếu RAM khi đang mở RUN_APP) | `reports/evidence/windows_selftest/` |
| 12/10 14:11 | Windows 11 laptop | RC | self-test **10/10 PASS** | `reports/evidence/windows_selftest/` |
| 12/10 20:49 | Linux | mã cuối (P10) | tương đương với lượt final test: 150/150 ảnh trùng điểm từng bit | `reports/test/recheck_final_code.json` |
| 12/10 21:00 | Linux, Chromium headless | mã cuối (P11) | UI e2e PASS: tải trang có token, chờ nạp mô hình, phân tích PNG, 3 thẻ kết quả, nhãn THỬ NGHIỆM ở phạm vi + thẻ, dòng “Điểm vận hành”, không có cảnh báo cấu hình, tab Khuôn mặt với ảnh có mặt, tải ZIP qua giao diện, lỗi file hỏng tiếng Việt, sai token → 403, không lỗi JavaScript | `reports/evidence/ui_e2e_linux_final/` |
| 12/10 21:11 | Linux | mã cuối (P12) | pytest **50/50 PASS** | `reports/evidence/pytest_final_linux.xml` |
| 12/10 21:13 | Linux | mã cuối (P12) | self-test **12/12 PASS** (gồm CLI đầy đủ ở tiến trình riêng, bộ kiểm định công ty, RAM) | `reports/evidence/linux_selftest_final/` |
| sau khi đóng gói | Linux | ZIP bàn giao | `tools/verify_package.py --selftest` | `ImageForgeryDetection_v2.zip.verify.json` (cạnh ZIP) |

## 4. Lỗi gặp trong kiểm thử và cách xử lý
- **Windows, self-test lượt 1 — CLI thoát mã 2**: bản self-test cũ nạp bộ mô hình thứ hai trong cùng tiến trình trên máy 8 GB. Sửa (P04c): kiểm tra CLI đầy đủ chạy ở **tiến trình riêng** trước khi nạp bộ chính; kiểm tra cơ chế CLI dùng cấu hình nhẹ; lưu stderr; thông báo thiếu RAM rõ ràng (P06). Lượt 2 (không mở ứng dụng khác) PASS.
- **Windows — không đo được RAM**: handle tiến trình bị cắt 32-bit trong ctypes → sửa (`ifd/sysinfo.py`).
- **Chuyển gói sang laptop (D01)**: dịch vụ truyền file trả bản cũ của `GHEP_ZIP.bat` → SHA-256 không khớp, Windows báo ZIP hỏng. Đã gửi lại dưới tên mới và đọc ngược để so từng byte; mỗi lần giao dùng thư mục mới.
- **Máy cloud hết RAM giữa final test (D02)**: chạy tiếp từ ảnh 1 258, không ảnh nào bị chấm hai lần hay bỏ sót; tương đương mã ở mục 1.
- **UI e2e**: thông báo lỗi file hỏng từng lộ địa chỉ đối tượng Python → sửa thông báo tiếng Việt (P04b).

## 5. Chưa kiểm thử (NOT RUN) và lý do
- **Gói cuối trên Windows**: mã sau bản RC đã thay đổi (P06–P12: thông báo lỗi, bộ kiểm định công ty, cảnh báo cấu hình, tối ưu dựng ảnh, `--no-images`, self-test mới) — đã kiểm trên Linux nhưng **chưa chạy trên Windows lúc đóng gói**. Sau khi giải nén gói cuối, bấm đúp `SELF_TEST.bat` (khoảng 3–5 phút, đóng cửa sổ RUN_APP trước); báo cáo PASS/FAIL ghi ở `reports\selftest\<thời điểm>_<tên máy>\selftest.txt`.
- Windows 10, CPU Intel/không AVX2, máy 4 GB RAM, máy có chính sách chặn `.bat`/AppLocker, phần mềm diệt virus chặn file `.pyd/.dll` — không có máy để thử.
- Giao diện web trên Windows: không có công cụ điều khiển trình duyệt trên laptop trong phiên này; chỉ có kiểm thử API web trong self-test + người dùng tự mở thử.
- Đường dẫn Windows rất dài (> 260 ký tự), ổ mạng, thư mục OneDrive chỉ-đám-mây.
- Tải đồng thời nhiều người dùng (ứng dụng thiết kế cho một người trên máy cá nhân).
