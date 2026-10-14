# BEFORE_AFTER_REPORT_VI — v1 (bản được audit) so với v2 (bàn giao)

_v1 = PSCC-Net + Streamlit, tiền xử lý cạnh dài 512, ngưỡng 0,35/0,65 (tái hiện trên runtime ONNX, |Δ điểm| ≤ 5,7·10⁻⁸ so với bản audit — E12). v2 = Mesorch + Community Forensics + YuNet/CLIP với ngưỡng khóa P04. Cả hai chạy trên **cùng final test**, mỗi bản một lần (v1 chạy trên toàn bộ ảnh của bài toán A và mọi ảnh thật, nhưng chỉ trên mẫu ngẫu nhiên cố định 300 ảnh AI OpenFake và 100 ảnh mỗi loại DeepFakeFace — P09). Số liệu sinh tự động: `reports/test/before_after_compare.json`, `metrics_test_v1_baseline.json`; bảng đầy đủ ở cuối `docs/_tables_test.md`._

## 1. Tóm tắt
- **Báo nhầm trên ảnh thật giảm mạnh:** gộp 1 784 ảnh thật cùng được hai bản chấm, “Nghi vấn” giảm từ **27,0 % (v1) xuống 9,0 % (v2)**. Trên ảnh camera: ctrl_mei 74,8 % → 4,0 %, Korus 85,1 % → 7,4 %; ảnh thật OpenFake 32,9 % → 7,0 %.
- **Phát hiện chỉnh sửa cục bộ trên ảnh camera tốt hơn rõ:** AUROC ctrl_mei 0,570 → 0,963 (chênh +0,393, CI 95 % [+0,360; +0,423]); Korus 0,528 → 0,622 (+0,094 [+0,055; +0,131]) — v1 gần như chấm “nghi vấn” mọi ảnh camera nên không phân biệt được (FPR 74,8–85,1 %).
- **Hai bài toán mới:** ảnh sinh AI và deepfake khuôn mặt. v1 gắn “Nghi vấn” cho **0 %** ảnh DeepFakeFace (hoán đổi/vẽ lại mặt/chân dung SD); v2: 77 %, 97 %, 100 % trên cùng mẫu.
- **Kết quả âm tính phải biết — CocoGlide (inpainting AI trên ảnh COCO):** PSCC-Net (v1) xếp hạng tốt hơn Mesorch (AUROC 0,778 so với 0,722; chênh −0,055 [−0,084; −0,026]); ở cấp ứng dụng v2 bắt nhiều ảnh inpainting hơn (59,6 % so với 35,7 %, nhờ bộ ảnh AI) nhưng **báo nhầm ảnh COCO thật nhiều hơn** (13,5 % so với 3,7 %). Có thể do PSCC-Net được huấn luyện trên ảnh sửa tổng hợp từ COCO (trùng miền) — chưa kiểm chứng được.
- v1 trông như “bắt” nhiều ảnh sửa Korus (91,9 %) và ảnh AI OpenFake (89,1 %) hơn v2, nhưng đó là do v1 gắn “Nghi vấn” cho cả ảnh thật cùng nguồn (85,1 %, 32,9 %); AUROC của v1 trên Korus ≈ 0,53 (gần may rủi).

## 2. Bài toán A — cùng ảnh, cùng nhãn
| Tập | Sửa / Gốc | AUROC v1 PSCC-Net | AUROC v2 Mesorch | v2 − v1 [CI 95 %, bootstrap ghép cặp theo nhóm] | F1 pixel macro v1 → v2 | Kích hoạt nhầm TB trên ảnh gốc v1 → v2 |
|---|---|---|---|---|---|---|
| ctrl_mei test | 218 / 202 | 0,570 | 0,963 | +0,393 [+0,360; +0,423] | 0,213 → 0,836 | 52,8 % → 0,7 % diện tích |
| Korus test | 148 / 148 | 0,528 | 0,622 | +0,094 [+0,055; +0,131] | 0,153 → 0,205 | 52,8 % → 0,6 % |
| CocoGlide | 512 / 512 | 0,778 | 0,722 | **−0,055 [−0,084; −0,026]** | 0,422 → 0,450 | 5,3 % → 2,9 % |

Ở điểm vận hành của từng bản (“Nghi vấn”): v1 ctrl_mei TPR 86,7 % / FPR 74,8 %; Korus 91,9 % / 85,1 %; CocoGlide 35,7 % / 3,7 % — so với v2 78,0 % / 0,5 %; 10,1 % / 2,0 %; 26,4 % / 5,7 %.

## 3. Kết luận người dùng nhìn thấy — cùng ảnh
| Nguồn | Loại | n | “Nghi vấn” v1 [Wilson 95 %] | “Nghi vấn” v2 | “Cần kiểm tra” trở lên v1 → v2 |
|---|---|---|---|---|---|
| ctrl_mei | thật | 202 | 74,8 % [68,3–80,2] | 4,0 % [2,0–7,6] | 83,7 % → 27,2 % |
| ctrl_mei | sửa | 218 | 86,7 % [81,5–90,6] | 78,4 % [72,5–83,4] | 90,8 % → 91,7 % |
| Korus | thật | 148 | 85,1 % [78,5–90,0] | 7,4 % [4,2–12,8] | 93,2 % → 32,4 % |
| Korus | sửa | 148 | 91,9 % [86,4–95,3] | 16,2 % [11,1–23,0] | 93,2 % → 45,9 % |
| CocoGlide | thật | 512 | 3,7 % [2,4–5,7] | 13,5 % [10,8–16,7] | 7,2 % → 44,9 % |
| CocoGlide | sửa (AI inpainting) | 512 | 35,7 % [31,7–40,0] | 59,6 % [55,3–63,7] | 41,6 % → 84,2 % |
| OpenFake | thật | 426 | 32,9 % [28,6–37,5] | 7,0 % [5,0–9,9] | 37,3 % → 29,6 % |
| OpenFake | AI (mẫu 294) | 294 | 89,1 % [85,0–92,2] | 59,9 % [54,2–65,3] | 94,6 % → 84,0 % |
| DFF | chân dung thật | 496 | 9,1 % [6,8–11,9] | 8,7 % [6,5–11,5] | 16,1 % → 32,5 % |
| DFF | hoán đổi mặt (mẫu) | 100 | 0,0 % [0,0–3,7] | 77,0 % [67,8–84,2] | 0,0 % → 94,0 % |
| DFF | vẽ lại mặt SD (mẫu) | 100 | 0,0 % [0,0–3,7] | 97,0 % [91,5–99,0] | 0,0 % → 100 % |
| DFF | chân dung SD 1.5 (mẫu) | 100 | 0,0 % [0,0–3,7] | 100 % [96,3–100] | 0,0 % → 100 % |

Lưu ý: vùng “Cần kiểm tra thêm” của v2 rộng hơn (ngưỡng `low` chọn ở FPR 20 % trên dev) nên tỷ lệ ảnh thật cần xem lại gộp chung gần như không đổi (32,7 % → 34,8 %), nhưng phần bị kết luận “Nghi vấn” giảm 3 lần.

## 4. 64 probe của audit (chẩn đoán, không phải độ chính xác)
Tái tạo đúng 64 đầu vào của audit (SHA-256 khớp bản ghi audit) rồi chấm bằng v1 và v2 (`research/before_after_probes.py`, `reports/evidence/before_after_probes.json`). Ảnh mẫu là ảnh demo của tác giả PSCC-Net, biến thể tạo bằng script và đồ họa vẽ — chỉ dùng để kiểm tra các lỗi F01–F04, **không** ước lượng độ chính xác.

| Nhóm probe | n | v1: Nghi vấn / Cần kiểm tra / Chưa thấy | v2: Nghi vấn / Cần kiểm tra / Chưa thấy | Nhận xét |
|---|---|---|---|---|
| F01 — cùng ảnh sửa 64 px, suy luận ở 4 kích thước (copy, splice) | 8 | 5 / 0 / 3 | 8 / 0 / 0 | v1: điểm 0,153–1,000 tùy kích thước; v2: không có tham số kích thước, điểm giống hệt ở cả 4 lần (1,000) |
| F01 — ảnh gốc ở 4 kích thước | 4 | 1 / 0 / 3 | 0 / 0 / 4 | v1: 0,000–0,984 tùy kích thước; v2: 0,364 cả 4 lần |
| F02 — điểm dưới `low` nhưng ≥ 50 % bản đồ ≥ 0,5 | 64 | 5 trường hợp mâu thuẫn | 0 | v2: điểm = max của chính bản đồ |
| F03 — sửa vùng 16 px (0,4 % diện tích) | 9 | 7 / 1 / 1 | 6 / 3 / 0 | |
| Sửa vùng 64 px | 9 | 9 / 0 / 0 | 9 / 0 / 0 | |
| Biến thể lành tính của ảnh gốc (JPEG 90/50, phóng ×2, sáng ×1,1) | 12 | 0 / 0 / 12 | 0 / 2 / 10 | 2 ca “Cần kiểm tra” của v2 do bộ ảnh AI |
| F04 — đồ họa vẽ (ngoài phạm vi) | 10 | 10 / 0 / 0 | 2 / 7 / 1 | v2 vẫn không phù hợp cho đồ họa: phần lớn “Cần kiểm tra thêm” |
| Ảnh demo của tác giả PSCC-Net — gốc | 6 | 0 / 0 / 6 | 0 / 3 / 3 | v1 đúng 12/12 trên ảnh demo của chính nó |
| Ảnh demo của tác giả PSCC-Net — đã sửa | 6 | 6 / 0 / 0 | 3 / 1 / 2 | v2 bỏ sót 2 ảnh xóa vật thể (removal1, removal2) |

Đọc kết quả: F01/F02 hết theo cấu trúc; F03 tương đương; F04 giảm từ “Nghi vấn” xuống chủ yếu “Cần kiểm tra thêm” nhưng đồ họa vẫn ngoài phạm vi. Trên chính ảnh demo của PSCC-Net, v1 tốt hơn v2 — đúng như dự kiến với ảnh do tác giả chọn; đánh giá chất lượng dựa trên final test (mục 1–3), không dựa trên các ảnh này.

## 5. Tình trạng các phát hiện audit F01–F09
| Mã | Vấn đề ở v1 | v2 | Bằng chứng |
|---|---|---|---|
| F01 | Điểm thay đổi theo kích thước suy luận (`max_side`) | **Đã sửa theo cấu trúc**: tiền xử lý cố định 512×512 theo pipeline chính thức của Mesorch, không còn tham số kích thước; cùng bytes → cùng điểm (kiểm chứng số trên Windows ≤ 2,1·10⁻⁷) | ROOT_CAUSE §1, probe mục 4, self-test Windows |
| F02 | Điểm toàn ảnh thấp nhưng bản đồ đỏ toàn ảnh | **Đã sửa theo cấu trúc**: điểm ảnh = max của chính bản đồ | ROOT_CAUSE §1, `tests/test_models_and_parity.py` |
| F03 | Bỏ sót vùng sửa nhỏ | **Giảm nhưng còn**: vùng nhỏ 0,3–1,5 % diện tích chỉ 54,1 % bị “Nghi vấn” (v1: chấm cao mọi ảnh nên không có ý nghĩa) | QUALITY §4.1 |
| F04 | Báo nhầm đồ họa vẽ | **Ngoài phạm vi, khai báo trên UI/README**; số đo probe ở mục 4 | TASK_CONTRACT §2 |
| F05 | Lọt trùng khi đổi encoding | Đã sửa (hash pixel + gần trùng + dòng dõi ảnh nền) | DATA_PROTOCOL §7 |
| F06 | CLI để lại kết quả cũ | Đã sửa (`runs/<run_id>/`, ghi nguyên tử) | `tests/test_cli.py`, self-test |
| F07 | Lỗi UI không có traceback | Đã sửa | `tests/test_webapp.py` |
| F08 | Thiếu quality gate | Đã sửa: hợp đồng + mục tiêu ghi trước, final test một lần, quy tắc bật mặc định áp dụng cơ học | TASK_CONTRACT, QUALITY_TARGETS, EXPERIMENT_LOG P01–P10 |
| F09 | ZIP hỏng/thiếu toàn vẹn | Đã sửa: CRC + SHA-256 từng file + giải nén mới + self-test | `tools/verify_package.py`, TEST_REPORT |

## 6. Vận hành
| | v1 | v2 |
|---|---|---|
| Phụ thuộc | PyTorch + Streamlit | ONNX Runtime + NumPy + Pillow + OpenCV (không cần PyTorch) |
| Cài đặt Windows | cần cài Python/thư viện | gói tự chứa `runtime\` (không cần cài, không cần Internet khi chạy) |
| Bài toán | chỉnh sửa cục bộ | chỉnh sửa cục bộ + ảnh AI + deepfake khuôn mặt, nêu rõ phần không kiểm tra/không áp dụng |
| Giới hạn ảnh | 20 MiB / 20 MP (6 ảnh test bị từ chối) | 25 MiB / 40 MP (2 ảnh test bị từ chối) |
| Thời gian phân tích (Linux 2 vCPU, ảnh 2 MP / 12 MP, p50) | 2,72 s / 3,67 s (PSCC-Net ONNX) | 1,90 s / 2,49 s (cả 3 bộ phát hiện) — `docs/CPU_BENCHMARK_VI.md` |
