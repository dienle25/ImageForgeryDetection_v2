# Bằng chứng — DefakerOne2 + Mesorch, cảnh báo độ tin cậy, script đo lại (08/11/2026)

Tên máy đã thay bằng `MAY_A`, thư mục người dùng và đường dẫn trên máy bằng `<THU_MUC>`. Không có ảnh, không có tên ảnh của data.zip.
**Không có số đo chất lượng nào của DefakerOne2 ở đây**: bước này chỉ viết code và script (xem `docs/DEFAKER2_ADDON_VI.md` mục 10).

| File | Nội dung | Tạo lại bằng |
|---|---|---|
| `kiem_tra_du_lieu_cong_khai.txt` | Dữ liệu công khai tải về có đủ ảnh của manifest: CocoGlide 1024/1024, Korus 440/440 (1 697 070 982 byte), DeepFakeFace dev 1200/1200, OpenFake dev 569/569. Chỉ đọc danh mục ZIP và cột `row_id`, không giải nén. | `python tools/tao_nhan_cong_khai.py --kiem-tra` (Python có pyarrow để kiểm OpenFake) |
| `pytest_toan_bo_20261108.txt` | Toàn bộ pytest bằng runtime của gói, môi trường như các file `.bat`: 129/129 đạt. | xem dòng đầu file |
| `pytest_lan1_thieu_utf8_20261108.txt` | Lần chạy đầu thiếu `PYTHONUTF8=1`: 8 test chạy tiến trình con hỏng vì bảng mã; code gốc (d5b700a) hỏng y như vậy. Giữ lại để minh bạch. | như trên, bỏ biến môi trường |
| `selftest_MAY_A_20261108.txt`, `.json` | `python -m ifd.selftest` (như `SELF_TEST.bat`): PASS 12, FAIL 0, SKIP 0. | `SELF_TEST.bat` |
| `kiem_tra_giao_dien/` | Chạy `app.js` thật trên DOM giả với JSON thật của `/api/analyze` (server DefakerOne2 giả lập, Mesorch thật, ảnh tạo ngẫu nhiên): thứ tự thẻ kết luận → thẻ DefakerOne2 → bản đồ "Vùng nghi sửa (Mesorch), tham khảo" → thẻ Mesorch; dòng "Độ tin cậy thấp" và "Hai mô hình không thống nhất" chỉ hiện khi có; không lỗi JavaScript. Không phải ảnh chụp trình duyệt. | `make_responses.py` rồi `harness.js` (lệnh ở đầu `make_responses.py`) |
