# Self-test trên Windows 11 của người dùng (bản RC)

Bản sao đã **che tên máy, tên người dùng Windows và đường dẫn thư mục** (thay bằng `WINDOWS-LAPTOP`, `<USER>`, `<THU_MUC_GIAI_NEN>`); ngoài ra giữ nguyên.
SHA-256 dưới đây là của file **gốc** trên laptop (trong `reports/selftest/` của bản RC) để đối chiếu.

| File (bản che) | SHA-256 file gốc |
|---|---|
| 20261012T140746Z_WINDOWS-LAPTOP_selftest.json | 1fe7b7a807fdf0b466887eca906aee73aa0cbcc33ab121f3c5433a06d4316844 |
| 20261012T140746Z_WINDOWS-LAPTOP_selftest.txt | 779b7e71afb51262faed7de81f1f400dda92078f545369615942ade8e58c4e76 |
| 20261012T141134Z_WINDOWS-LAPTOP_selftest.json | c2c804c561c340f9221d8a8fd486c49a39b432c4af69f9047e23aef880acd193 |
| 20261012T141134Z_WINDOWS-LAPTOP_selftest.txt | b2df07f5ec292b98cb29e7cfadfc3349bc0dcae476ab507b9ba76fc2e08247fc |

- Lượt 1 (14:07:46Z): 9/10 PASS — `cli_runs_and_artifacts` FAIL (exit 2; bản self-test cũ không lưu stderr).
- Lượt 2 (14:11:34Z): 10/10 PASS — cùng bản self-test cũ (10 mục), chạy khi không mở ứng dụng khác.
- Bản self-test mới (11 mục, CLI chạy ở tiến trình riêng, đo RSS) chạy trên gói cuối cùng; kết quả ghi ở `docs/TEST_REPORT_VI.md`.
