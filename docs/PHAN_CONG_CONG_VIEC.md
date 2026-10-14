# PHÂN CÔNG CÔNG VIỆC — ImageForgeryDetection_v2

_Lập ngày 14/10/2026. Hạn là đề xuất, trưởng nhóm có thể chỉnh._

| Mã | Việc | Người làm | Hạn | Nộp ở đâu |
|---|---|---|---|---|
| T1 | Chạy `SELF_TEST.bat` của gói cuối trên ≥ 2 máy Windows | @TranKhoaDang | 16/10 | `reports/evidence/windows_selftest_final/` |
| T2 | Kiểm thử giao diện web theo checklist | @TranKhoaDang | 19/10 | `docs/test/KIEM_THU_GIAO_DIEN.md` |
| T3 | Ma trận tương thích Windows | @TranKhoaDang | 21/10 | `docs/test/MA_TRAN_TUONG_THICH.md` |
| T4 | Bộ ảnh có nhãn tự tạo + chạy `KIEM_DINH_CONG_TY.bat` | @TranKhoaDang | 28/10 | `docs/test/KIEM_DINH_BO_ANH_NHOM.md` + `metrics.json` |
| B1 | Rà hướng dẫn sử dụng như người dùng mới + hướng dẫn nhanh 1 trang | @thanhtrumbacang | 19/10 | `docs/HUONG_DAN_NHANH.md`, sửa `README_VI.md` |
| B2 | Test ảnh gửi qua Zalo / Messenger / Facebook / email | @thanhtrumbacang | 23/10 | `docs/test/TEST_KENH_CHIA_SE.md` |
| B3 | Cập nhật tài liệu theo kết quả kiểm thử | @thanhtrumbacang | liên tục, chốt 30/10 | `docs/TEST_REPORT_VI.md`, `STATUS.md`, `CHANGELOG.md`, `EXPERIMENT_LOG.jsonl` |
| B4 | Báo cáo môn học + slide | @thanhtrumbacang | nháp 26/10, chốt 30/10 | `docs/bao_cao/` |
| B5 | Video demo 3–5 phút | @thanhtrumbacang | 30/10 | link trong `docs/bao_cao/README.md` |
| — | Gửi gói chạy được, duyệt & merge PR, chọn điểm vận hành, chốt nộp | @dienle25 | liên tục | — |

## @TranKhoaDang — Kiểm thử (tester)

**T1 — Self-test gói cuối trên Windows (hạn 16/10)**
- Tải gói `ImageForgeryDetection_v2.zip` ở mục Releases của repo, kiểm SHA-256: `certutil -hashfile ImageForgeryDetection_v2.zip SHA256` phải ra `08bd9cfbcd728592e785654a200bfebcb33397dc7b8dfd30f482be55f1ab71b4`.
- Giải nén vào thư mục ngắn (ví dụ `D:\IFD`), đóng ứng dụng nặng, bấm đúp `SELF_TEST.bat` (3–5 phút).
- Chạy trên ít nhất 2 máy. Nộp `selftest.txt` và `.json` (trong `reports\selftest\...`) sau khi thay tên máy/tên người dùng bằng `MAY_A`, `MAY_B`.
- Xong khi: có kết quả của ≥ 2 máy; mỗi mục FAIL có một issue kèm log.

**T2 — Kiểm thử giao diện web (hạn 19/10)** — với mỗi mục ghi Đạt/Không đạt và kèm ảnh chụp màn hình:
- Ảnh JPEG, PNG; ảnh lớn 12 MP; ảnh chụp điện thoại bị xoay (EXIF); PNG nền trong suốt.
- File hỏng, file chữ đổi đuôi `.jpg`, file rỗng → phải báo lỗi tiếng Việt, không treo.
- Đủ 4 tab: Chồng lớp, Bản đồ nhiệt, Khuôn mặt, Ảnh gốc. Tải được từng file và cả ZIP.
- Phân tích liên tiếp 10 ảnh (ghi RAM trong Task Manager).
- Chế độ nghiêm: `RUN_APP.bat --config research\configs\config_strict.json` → trang ghi “Điểm vận hành: nghiêm”.
- Mở `RUN_APP.bat` lần thứ hai khi lần đầu còn chạy → phải báo cổng bận.

**T3 — Ma trận tương thích (hạn 21/10)** — điền bảng PASS / FAIL / NOT RUN (kèm lý do) cho từng máy có được:
- Windows 10 và 11; CPU Intel và AMD; RAM 8 GB và 16 GB.
- Đường dẫn có dấu tiếng Việt + khoảng trắng; đường dẫn dài (> 200 ký tự); thư mục OneDrive; khi bật phần mềm diệt virus.
- Ở mỗi máy ghi: `RUN_APP.bat` mở được không, `SELF_TEST.bat` PASS bao nhiêu mục, thời gian nạp mô hình, `PHAN_TICH_THU_MUC.bat` chạy được không.

**T4 — Kiểm định trên bộ ảnh nhóm tự tạo (hạn 28/10)**
- Tự chụp ≥ 100 ảnh thật; tạo ≥ 100 ảnh sửa bằng công cụ thật (Photoshop/GIMP/Snapseed, tính năng “xóa vật thể” trên điện thoại, ảnh tạo bằng AI). Mỗi loại ≥ 20 ảnh: ghép, sao chép vùng, xóa vật thể, ảnh AI, ghép mặt. Ghép mặt chỉ dùng mặt người đã đồng ý.
- Xếp vào `that\` và `sua\ghep_anh`, `sua\sao_chep_vung`, `sua\xoa_vat_the`, `sua\anh_ai`, `sua\ghep_mat` (xem README_VI.md mục 6), kéo thả thư mục vào `KIEM_DINH_CONG_TY.bat`.
- Nộp `metrics.json` và bảng tóm tắt theo từng loại. **Không đưa ảnh lên GitHub.**

## @thanhtrumbacang — Báo cáo & tài liệu

**B1 — Rà hướng dẫn (hạn 19/10):** làm theo `README_VI.md` từ đầu trên máy chưa cài gì; ghi lại mọi chỗ khó hiểu hoặc thiếu; sửa `README_VI.md`; viết `docs/HUONG_DAN_NHANH.md` (1 trang: tải → giải nén → RUN_APP → đọc kết quả).

**B2 — Test qua kênh chia sẻ (hạn 23/10):** tự tạo 10–20 cặp ảnh (gốc / đã sửa), gửi từng ảnh qua Zalo, Messenger, Facebook, email rồi tải về. Chạy `PHAN_TICH_THU_MUC.bat` cho bản gốc và bản đã qua kênh. Lập bảng: loại sửa, kênh gửi, mức kết luận, bản đồ nhiệt có đúng chỗ sửa không. Chỉ nộp bảng và `results.csv`, không nộp ảnh.

**B3 — Cập nhật tài liệu (liên tục, chốt 30/10):** sau mỗi PR của T1–T4 và B2 thì cập nhật:
- `docs/TEST_REPORT_VI.md`: bảng PASS / FAIL / NOT RUN và mục 3 “Các lượt chạy”;
- `STATUS.md`, `CHANGELOG.md`;
- `EXPERIMENT_LOG.jsonl` (mỗi sự kiện một dòng JSON).

Chỉ ghi kết quả có file bằng chứng; không sửa số liệu final test cũ.

**B4 — Báo cáo môn học + slide (nháp 26/10, chốt 30/10)** gồm các phần:
- bài toán và phạm vi;
- 3 mô hình;
- cách đánh giá (dev / final test chạy một lần);
- kết quả chính (`docs/QUALITY_REPORT_VI.md` mục 4–5);
- giới hạn;
- kết quả kiểm thử của nhóm (T1–T4, B2);
- hướng phát triển.

Không viết “chính xác 95 %”, “phát hiện mọi ảnh giả”, “production-ready”.

**B5 — Video demo 3–5 phút (hạn 30/10):** mở ứng dụng, phân tích 1 ảnh gốc và 3 ảnh đã sửa, giải thích mức kết luận và bản đồ nhiệt.

## @dienle25 — Trưởng nhóm
Đưa gói chạy được lên Releases, mời 2 bạn vào repo, duyệt và merge PR trong 24–48 giờ, chọn điểm vận hành chuẩn/nghiêm sau T4, chốt bản nộp.

## Cách nộp (giống Music_SEG)
- Không push thẳng `main` (nhánh đã khóa, chỉ @dienle25 cập nhật được).
- Mỗi việc làm trên một nhánh `<tên-github>/<mã-việc>` (hoặc trên web chọn “Create a new branch for this commit and start a pull request”). Sau đó mở pull request tiêu đề `[T1] …`; @dienle25 duyệt rồi mới merge.
- Không commit: ảnh cá nhân, file > 10 MB, mô hình `.onnx/.pth`, file ZIP, mật khẩu/token.
- Trong log phải thay tên máy và tên người dùng bằng `MAY_A` và `<THU_MUC>`.
