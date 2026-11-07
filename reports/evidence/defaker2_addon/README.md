# Bằng chứng: gói phụ DefakerOne2 (07/11/2026, máy MAY_A)

Bản sao đã che tên máy (`MAY_A`), tên người dùng (`<USER>`), các đường dẫn (`<THU_MUC_...>`) và tên ảnh thử (`anh_1..3`). Phần còn lại giữ nguyên. `sha256_file_goc.json` ghi SHA-256 của các file gốc để đối chiếu.

| File | Nội dung |
|---|---|
| `chay_thu_3anh_cli_va_server.json` | Bài chạy thử 3 ảnh: p_fake của `swift infer` so với server (khớp tuyệt đối), thời gian, RAM, VRAM |
| `do_app_cung_server_cau_hinh_demo.json` | App (cấu hình demo đầy đủ) và server chạy cùng lúc, ảnh 2 MP và 12 MP: thời gian, RAM, VRAM |
| `do_app_cung_server_cau_hinh_tiet_kiem_ram.json` | Như trên, nhưng tắt `face_clip` |
| `bo_nho_may_luc_do.json` | Bộ nhớ của các ứng dụng khác và commit của máy lúc đo; lần nạp lỗi 0xC0000005 |
| `log_server_tom_tat.txt` | Các dòng trạng thái của server: nạp, warm-up, các lượt đo. Bỏ các dòng chấm bộ ảnh riêng của nhóm |
| `selftest_demo_MAY_A.txt` / `.json` | Self-test đầy đủ trên bản demo: 12/12 PASS |

Không có ảnh nào trong thư mục này.
