# ImageForgeryDetection_v2

Ứng dụng chạy trên CPU (Windows) kiểm tra ảnh bị chỉnh sửa cục bộ (ghép, sao chép vùng, xóa/thay thế), ảnh sinh bằng AI và deepfake khuôn mặt. Hướng dẫn sử dụng: [README_VI.md](README_VI.md). Trạng thái: [STATUS.md](STATUS.md).

- **Gói chạy được** (kèm mô hình và Python, ~1 GB): tải `ImageForgeryDetection_v2.zip` ở mục **Releases**. Kiểm SHA-256 `08bd9cfbcd728592e785654a200bfebcb33397dc7b8dfd30f482be55f1ab71b4`. Repo chỉ chứa mã nguồn, tài liệu và báo cáo.
- **Phân công**: [docs/PHAN_CONG_CONG_VIEC.md](docs/PHAN_CONG_CONG_VIEC.md).

## Nộp bài ở đâu
| Việc | Thư mục nộp |
|---|---|
| T1 self-test Windows | `reports/evidence/windows_selftest_final/` |
| T2, T3, T4, B2 kết quả kiểm thử | `docs/test/` |
| B1 hướng dẫn | `README_VI.md`, `docs/HUONG_DAN_NHANH.md` |
| B3 cập nhật tài liệu | `docs/TEST_REPORT_VI.md`, `STATUS.md`, `CHANGELOG.md` |
| B4 báo cáo, slide; B5 link video | `docs/bao_cao/` |

## Cách nộp (phải được @dienle25 duyệt)
1. Không push thẳng `main`: nhánh này đã khóa, chỉ @dienle25 cập nhật được.
2. Tạo nhánh `<tên-github>/<mã-việc>` hoặc trên web chọn *Create a new branch for this commit and start a pull request*.
3. Mở Pull Request với tiêu đề `[T1] …`. Chỉ khi @dienle25 duyệt và merge thì nội dung mới vào `main`.
4. Không commit ảnh cá nhân, file > 10 MB, mô hình `.onnx/.pth`, ZIP, mật khẩu/token.
