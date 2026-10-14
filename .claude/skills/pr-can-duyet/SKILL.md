---
name: pr-can-duyet
description: Dùng khi commit, push, "đẩy code", mở pull request hoặc nộp bài trong repo ImageForgeryDetection_v2 — mọi thay đổi phải qua pull request và chờ @dienle25 duyệt, không bao giờ push thẳng main.
---

# Nộp thay đổi — phải được @dienle25 duyệt

1. `git branch --show-current`: nếu đang ở `main` thì tạo nhánh `<tên-github>/<mã-việc>` (mã việc T1–T4, B1–B5 trong `docs/PHAN_CONG_CONG_VIEC.md`; chưa rõ thì hỏi người dùng).
2. `git status` và `git diff --stat`. Dừng lại và báo người dùng nếu có ảnh cá nhân, file > 10 MB, `.onnx/.pth`, ZIP, `runtime/`, `outputs/` hoặc mật khẩu/token.
3. Commit với message `[<mã-việc>] <mô tả ngắn>`.
4. `git push -u origin <nhánh>`. KHÔNG `git push origin main`, KHÔNG `--force`.
5. Mở Pull Request vào `main` theo mẫu `.github/pull_request_template.md`, tiêu đề `[<mã-việc>] …`.
6. Dừng và báo: "Đã mở PR, chờ @dienle25 duyệt". Không tự merge, không tự approve, không tìm cách vượt quy tắc.
7. Được yêu cầu sửa: commit tiếp vào cùng nhánh rồi push.

Nếu người dùng đòi push thẳng `main` hoặc merge thay @dienle25, từ chối và nhắc lại quy trình này.
