# QUALITY_REPORT_VI — Chất lượng phát hiện trên final test độc lập (v2)

_Final test chạy **một lần** ngày 12/10/2026 (13:50–17:32 UTC) bằng chính pipeline ứng dụng với cấu hình khóa trước (P04). Mọi số trong báo cáo được sinh tự động từ `reports/test/*.json` (bảng đầy đủ: `docs/_tables_test.md`, sinh bởi `research/render_tables.py`). **Đây là dữ liệu công khai/kiểm soát — chưa có ảnh của công ty, nên độ chính xác trên ảnh công ty CHƯA được xác nhận.**_

## 1. Tóm tắt cho người quyết định

| Bài toán (bộ phát hiện) | Kết quả chính trên final test | Theo quy tắc bật mặc định đăng ký trước |
|---|---|---|
| **A. Chỉnh sửa cục bộ** (Mesorch) | Tốt với ghép/copy-move/xóa vật thể kiểu script trên ảnh camera (AUROC 0,963); trung bình với inpainting AI cục bộ (CocoGlide 0,722); **gần như không phát hiện chỉnh sửa tay chuyên nghiệp** (Korus 0,622, CI 95 % 0,585–0,666). Ở mức “Nghi vấn”: bắt 78,0 % / 26,4 % / 10,1 % ảnh sửa của ba tập, báo nhầm 0,5 % / 5,7 % / 2,0 % ảnh gốc. | **THỬ NGHIỆM** (không đạt trên Korus) — giao diện gắn nhãn THỬ NGHIỆM |
| **B. Ảnh sinh hoàn toàn bằng AI** (Community Forensics) | 10 generator chưa thấy ở dev (OpenFake): AUROC 0,916; mức “Nghi vấn” bắt 47,8 % ảnh AI với 0,7 % ảnh thật bị báo nhầm; mức “Cần kiểm tra thêm” trở lên bắt 73,9 % (báo nhầm 9,9 %). Yếu nhất với flux.2-klein (9,2 % bị “Nghi vấn”). Chân dung SD 1.5: 100 %. | Đạt |
| **C. Deepfake khuôn mặt** (YuNet + CLIP) | Hoán đổi mặt: AUROC 0,941, “Nghi vấn” bắt 81,2 %, báo nhầm 4,2 % mặt thật; vẽ lại mặt bằng SD: AUROC 0,889, bắt 58,3 %. | Đạt |

**Cấp ứng dụng (kết luận người dùng thấy, điểm vận hành chuẩn):** gộp 1 784 ảnh thật của mọi nguồn, **9,0 %** bị gắn “Nghi vấn” và **34,8 %** bị gắn “Cần kiểm tra thêm” trở lên (dao động theo nguồn: 4,0–13,5 % và 27,2–44,9 %). Với điểm vận hành “nghiêm” (khóa trước test): 2,1 % và 9,0 %, đổi lại bỏ sót nhiều hơn (mục 5).

**Mục tiêu đăng ký trước (`QUALITY_TARGETS.yaml`): đạt 16/19 mục tiêu chất lượng và 2/2 mục tiêu CPU.** Không đạt: AUROC ≥ 0,75 của bài toán A trên Korus (0,622) và CocoGlide (0,722); FPR ≤ 5 % của bài toán B trên ảnh thật COCO cỡ nhỏ (8,6 %).

**Ý nghĩa cho công ty:** ứng dụng phù hợp để **sàng lọc và ưu tiên ảnh cho người thẩm định**, không phù hợp làm căn cứ duy nhất cho quyết định tiền bạc. “Chưa thấy dấu hiệu rõ” **không** chứng minh ảnh thật (ví dụ 54,1 % ảnh sửa tay Korus và 15,3 % ảnh AI OpenFake rơi vào mức này). Khi ảnh giả hiếm, phần lớn ảnh bị “Nghi vấn” sẽ là ảnh thật (mục 5.3). Trước khi dùng thật cần chạy `KIEM_DINH_CONG_TY.bat` trên ảnh có nhãn của chính công ty (README mục 6).

## 2. Cách đọc số liệu
- **Điểm** của mỗi bộ phát hiện là điểm thô của mô hình (không phải xác suất). Hai ngưỡng cố định chia ba mức: < `low` → “Chưa thấy dấu hiệu rõ”; `low` ≤ điểm < `high` → “Cần kiểm tra thêm”; ≥ `high` → “Nghi vấn”.
- **TPR (recall)** = tỷ lệ ảnh sửa/giả bị gắn mức đó trở lên; **FPR** = tỷ lệ ảnh thật bị gắn mức đó trở lên; **precision** phụ thuộc tỷ lệ ảnh giả trong tập (ở đây khoảng một nửa) → **không** chuyển thẳng sang thực tế công ty, nơi ảnh giả có thể hiếm hơn nhiều.
- **AUROC** đo khả năng xếp hạng ảnh giả cao hơn ảnh thật, không phụ thuộc ngưỡng (0,5 = may rủi). **CI 95 %**: bootstrap 1000 lần theo nhóm nguồn (ảnh cùng cảnh/cùng ảnh gốc được lấy cùng nhau).
- **Vùng “Cần kiểm tra thêm”** là vùng từ chối kết luận (abstain); **coverage** = tỷ lệ ảnh có kết luận rõ (“Nghi vấn” hoặc “Chưa thấy dấu hiệu rõ”).
- **Cấp ứng dụng** = kết luận chung (mức cao nhất giữa các bộ phát hiện áp dụng được). Ba bộ phát hiện cùng chạy nên tỷ lệ ảnh thật bị “Nghi vấn” ở cấp ứng dụng cao hơn từng bộ riêng lẻ.

## 3. Dữ liệu và quy trình final test
- Tập test dùng **một lần**, sau khi khóa `config.json` (P04: SHA-256 cấu hình `20329c93…`, `models/manifest.json` `77165d62…`), bằng chính pipeline ứng dụng (`research/eval_product.py` gọi `ifd.analysis.Analyzer`, ONNX Runtime CPU). Không đổi ngưỡng/mô hình sau khi xem kết quả.
- Thành phần (nguồn, giấy phép, cách chia, kiểm tra trùng: `docs/DATA_PROTOCOL.md`; manifest + SHA-256: `reports/manifests/`):

| Bài toán | Tập test | Ảnh dương | Ảnh âm | Ghi chú |
|---|---|---|---|---|
| A | `ctrl_mei` test | 218 ảnh ghép / copy-move / xóa (script), 6 kiểu hậu xử lý | 202 ảnh camera gốc cùng hậu xử lý | cùng phân phối với dev |
| A | Korus test | 148 ảnh sửa tay thực tế | 148 ảnh gốc cùng cảnh | cùng phân phối với dev_korus (P02) |
| A | CocoGlide | 512 ảnh inpainting GLIDE | 512 ảnh COCO gốc | chưa dùng ở dev |
| B | OpenFake test | 770 ảnh của 10 generator **không có trong dev** | 426 ảnh thật (ImageNet 285, DOCCI 141) | holdout generator |
| B | DeepFakeFace text2img | 496 chân dung SD 1.5 | 496 ảnh thật IMDB-WIKI | ngoài miền |
| C | DeepFakeFace test | 496 hoán đổi mặt InsightFace; 496 vẽ lại mặt SD | 496 ảnh thật IMDB-WIKI | khác miền huấn luyện (FF++, video) |

- 4 940 ảnh được chấm; **2 lỗi** (2 ảnh PNG OpenFake > 25 MiB bị ứng dụng từ chối theo giới hạn `limits.max_file_mb` — đúng thiết kế, ghi lỗi chứ không loại thầm lặng); 18 ảnh trùng pixel chỉ tính một lần → 4 920 ảnh trong chỉ số.
- Gián đoạn kỹ thuật: tiến trình bị máy cloud dừng do hết RAM sau 1 257 ảnh và chạy tiếp từ ảnh 1 258 (D02). Phần mã chấm điểm không đổi; tương đương số học được kiểm lại ở mục 9.

## 4. Kết quả theo bài toán (điểm vận hành chuẩn)

### 4.1 A — Chỉnh sửa cục bộ (Mesorch)
| Tập | Sửa / Gốc | AUROC [CI 95 %] | AP | “Nghi vấn”: TP / FP / TN / FN | Precision | TPR | FPR | FNR | ≥ “Cần kiểm tra”: TPR / FPR | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| ctrl_mei | 218 / 202 | 0,963 [0,947; 0,978] | 0,973 | 170 / 1 / 201 / 48 | 99,4 % | 78,0 % | 0,5 % | 22,0 % | 90,4 % / 11,9 % | 88,1 % |
| Korus | 148 / 148 | 0,622 [0,585; 0,666] | 0,667 | 15 / 3 / 145 / 133 | 83,3 % | 10,1 % | 2,0 % | 89,9 % | 27,0 % / 7,4 % | 88,9 % |
| CocoGlide | 512 / 512 | 0,722 [0,695; 0,750] | 0,713 | 135 / 29 / 483 / 377 | 82,3 % | 26,4 % | 5,7 % | 73,6 % | 53,9 % / 20,9 % | 78,6 % |

Mức pixel (ngưỡng pixel 0,5 cố định; chỉ ảnh có mask; ảnh gốc báo riêng là “kích hoạt nhầm”):

| Tập | F1 macro | IoU macro | F1 micro | AUC pixel TB | % ảnh F1 ≥ 0,5 | Kích hoạt nhầm TB trên ảnh gốc |
|---|---|---|---|---|---|---|
| ctrl_mei | 0,836 | 0,785 | 0,906 | 0,989 | 89,0 % | 0,74 % diện tích |
| Korus | 0,205 | 0,162 | 0,216 | 0,742 | 23,0 % | 0,63 % |
| CocoGlide | 0,450 | 0,382 | 0,575 | 0,897 | 48,2 % | 2,89 % |

Theo nhóm (ctrl_mei): vùng sửa **nhỏ** (0,3–1,5 % diện tích) chỉ 54,1 % bị “Nghi vấn” (vừa 87,7 %, lớn 93,0 %); copy-move khó nhất (71,2 %). Hậu xử lý (JPEG 75–95, PNG, thu nhỏ) giữ AUROC 0,933–0,990; riêng “thu nhỏ ½ + JPEG 85” làm FPR ảnh gốc lên 3,7 %. Korus theo máy ảnh: AUROC 0,565–0,729, TPR “Nghi vấn” 5,4–13,5 %.

### 4.2 B — Ảnh sinh hoàn toàn bằng AI (Community Forensics)
| Tập | Sinh / Thật | AUROC [CI 95 %] | AP | “Nghi vấn”: TP / FP / TN / FN | Precision | TPR | FPR | FNR | ≥ “Cần kiểm tra”: TPR / FPR |
|---|---|---|---|---|---|---|---|---|---|
| OpenFake (10 generator mới) | 770 / 426 | 0,916 [0,900; 0,930] | 0,954 | 368 / 3 / 423 / 402 | 99,2 % | 47,8 % | 0,7 % | 52,2 % | 73,9 % / 9,9 % |
| DeepFakeFace SD 1.5 | 496 / 496 | 1,000 [1,000; 1,000] | 1,000 | 496 / 0 / 496 / 0 | 100 % | 100 % | 0,0 % | 0,0 % | 100 % / 3,4 % |

Theo generator (TPR mức “Nghi vấn”): illustrious 91,1 %, midjourney-7 77,8 %, veo-3 55,9 %, gpt-image-1.5 45,2 %, z-image-turbo 41,3 %, **flux.2-klein-9b 9,2 %** (AUROC 0,776); các generator < 10 ảnh không kết luận. Ảnh thật bị báo “ảnh AI” ở mức “Nghi vấn”: ctrl_mei 1,0 %, Korus 2,7 %, **COCO (CocoGlide) 8,6 %**, OpenFake 0,7 % (DOCCI độ phân giải cao 2,1 %, ImageNet 0 %), chân dung IMDB-WIKI 0 %. Trên ảnh chỉ bị inpainting một phần, bộ này cũng kích hoạt (CocoGlide 45,7 %, DFF vẽ lại mặt 94,2 %) — không tính đúng/sai cho B nhưng giúp kết luận chung bắt được các ảnh đó.

### 4.3 C — Deepfake khuôn mặt (YuNet + CLIP)
| So sánh (DeepFakeFace) | Giả / Thật | AUROC [CI 95 %] | AP | “Nghi vấn”: TP / FP / TN / FN | Precision | TPR | FPR | ≥ “Cần kiểm tra”: TPR / FPR | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| Hoán đổi mặt InsightFace | 496 / 496 | 0,941 [0,925; 0,958] | 0,952 | 403 / 21 / 475 / 93 | 95,0 % | 81,2 % | 4,2 % | 92,3 % / 15,7 % | 88,7 % |
| Vẽ lại mặt SD inpainting | 496 / 496 | 0,889 [0,867; 0,909] | 0,895 | 289 / 21 / 475 / 207 | 93,2 % | 58,3 % | 4,2 % | 81,5 % / 15,7 % | 82,7 % |
| (chéo) mặt sinh hoàn toàn SD 1.5 | 496 / 496 | 0,775 [0,747; 0,803] | 0,792 | — | 89,3 % | 35,5 % | 4,2 % | 61,1 % / 15,7 % | — |

Không tìm thấy mặt trên 1,8 % ảnh chân dung thật (1,2–1,6 % ảnh giả) → các ảnh này “không áp dụng”. Trên ảnh thật không phải chân dung, YuNet tìm thấy mặt ở 1,4–30,2 % ảnh, báo nhầm “Nghi vấn” 0–2,7 %.

## 5. Cấp ứng dụng — kết luận người dùng nhìn thấy

### 5.1 Điểm vận hành chuẩn (mặc định) và “nghiêm” (tùy chọn, khóa trước test)
| Nguồn | Ảnh thật → “Nghi vấn” (chuẩn / nghiêm) | Ảnh thật → “Cần kiểm tra” trở lên (chuẩn / nghiêm) | Ảnh sửa/giả → “Nghi vấn” (chuẩn / nghiêm) | Ảnh sửa/giả → “Cần kiểm tra” trở lên (chuẩn / nghiêm) |
|---|---|---|---|---|
| ctrl_mei (ảnh camera, sửa bằng script) | 4,0 % / 1,0 % | 27,2 % / 4,0 % | 78,4 % / 62,4 % | 91,7 % / 78,4 % |
| Korus (sửa tay chuyên nghiệp) | 7,4 % / 0,0 % | 32,4 % / 7,4 % | 16,2 % / 5,4 % | 45,9 % / 16,2 % |
| CocoGlide (inpainting AI cục bộ) | 13,5 % / 4,3 % | 44,9 % / 13,5 % | 59,6 % / 38,1 % | 84,2 % / 59,6 % |
| OpenFake (ảnh AI, 10 generator mới) | 7,0 % / 0,9 % | 29,6 % / 7,0 % | 58,2 % / 30,5 % | 84,7 % / 58,2 % |
| DFF hoán đổi mặt | — | — | 82,1 % / 60,3 % | 95,0 % / 82,1 % |
| DFF vẽ lại mặt (SD) | — | — | 98,0 % / 88,7 % | 100,0 % / 98,0 % |
| DFF chân dung SD 1.5 | — | — | 100,0 % / 99,4 % | 100,0 % / 100,0 % |
| DFF ảnh chân dung thật | 8,7 % / 1,8 % | 32,5 % / 8,7 % | — | — |

- Ngưỡng `low` của profile nghiêm trùng ngưỡng `high` của profile chuẩn (cùng mức FPR 5 % trên dev), nên cột “nghiêm, Cần kiểm tra trở lên” bằng cột “chuẩn, Nghi vấn”.
- Profile nghiêm: `research/configs/config_strict.json`, chạy bằng `RUN_APP.bat --config research\configs\config_strict.json` (README mục 7). Cả hai profile được ghi là “đã đánh giá” trong `reports/validation_summary_vi.json`; trang web hiện tên profile đang dùng, còn cấu hình khác hai profile này sẽ bị cảnh báo.
- Chọn profile là quyết định kinh doanh (chi phí một ảnh giả lọt qua so với chi phí người thẩm định xem ảnh thật): nên quyết định sau khi chạy `KIEM_DINH_CONG_TY.bat` trên ảnh của công ty.

### 5.2 Ai gây ra “Nghi vấn” trên ảnh thật
Bộ ảnh AI là nguồn chính trên ảnh COCO cỡ nhỏ (CocoGlide); Mesorch là nguồn chính trên ảnh chân dung và ảnh nén nhỏ (ImageNet, IMDB-WIKI: 4,6–6,4 % ảnh thật bị Mesorch “Nghi vấn”); bộ khuôn mặt gây khoảng 1–5 % trên ảnh có mặt người (`reports/test/failure_analysis.json`).

### 5.3 Khi ảnh giả hiếm (ví dụ minh họa, không phải ước lượng cho công ty)
Giả sử 1 trên 50 ảnh nhận được là giả, ứng dụng gắn “Nghi vấn” cho 60 % ảnh giả và 7 % ảnh thật (gần mức đo được ở trên). Trong 1 000 ảnh: 20 ảnh giả → 12 bị “Nghi vấn”; 980 ảnh thật → ~69 bị “Nghi vấn”. Tức chỉ khoảng **1/7 ảnh “Nghi vấn” là giả thật**. Vì vậy “Nghi vấn” nghĩa là “chuyển người thẩm định xem kỹ”, không phải “đã xác định là giả”. Tỷ lệ thật của công ty chỉ đo được bằng dữ liệu có nhãn của công ty.

## 6. Độ bền với xử lý lành tính (cùng ảnh, ngưỡng đã khóa; `reports/test/robustness_summary.json`)
**Ảnh AI bị nén/thu nhỏ lại (OpenFake test, 770 ảnh AI + 426 ảnh thật):** khả năng phát hiện giảm rõ — kiểu đường đi qua ứng dụng nhắn tin/mạng xã hội.

| Biến đổi | AUROC | Ảnh AI bị “Nghi vấn” | Ảnh thật bị “Nghi vấn” |
|---|---|---|---|
| gốc | 0,916 | 47,8 % | 0,7 % |
| nén lại JPEG 75 | 0,845 | 36,4 % | 1,4 % |
| thu nhỏ ½ + JPEG 85 | 0,874 | 41,9 % | 0,7 % |
| cạnh dài 1024 + JPEG 80 | 0,859 | 39,2 % | 2,1 % |

**Chỉnh sửa toàn ảnh lành tính trên ảnh thật (E13, đăng ký trước khi chạy):** tỷ lệ ảnh thật bị “Nghi vấn” — gốc → sau biến đổi.

| Biến đổi (đều lưu lại JPEG 95) | Mesorch (368 ảnh camera thật) | Bộ ảnh AI (368 ảnh camera thật) | Bộ khuôn mặt (496 chân dung thật) |
|---|---|---|---|
| chỉ nén lại JPEG 95 | 1,1 % → 0,5 % | 1,6 % → 0,5 % | 4,2 % → 4,8 % |
| tăng sáng ×1,15 | 1,1 % → 1,4 % | 1,6 % → 0,5 % | 4,2 % → 4,8 % |
| giảm tương phản ×0,85 | 1,1 % → 0,5 % | 1,6 % → 1,6 % | 4,2 % → 4,2 % |
| cắt giữa 80 % | 1,1 % → 0,5 % | 1,6 % → 1,9 % | 4,2 % → 4,0 % |
| xoay 90° (không qua EXIF) | 1,1 % → **2,7 %** | 1,6 % → 1,4 % | 4,2 % → 1,8 % — nhưng **76,4 % ảnh không còn tìm thấy mặt** |

Kết luận: chỉnh sáng/tương phản/cắt/nén lại không làm tăng báo nhầm đáng kể. Hai điểm yếu: (1) ảnh AI bị nén/thu nhỏ lại thì bị bỏ sót nhiều hơn (−6 đến −11 điểm phần trăm ở mức “Nghi vấn”); (2) ảnh bị xoay nằm ngang mà không có thẻ EXIF (ví dụ ảnh chụp màn hình của ảnh dọc) làm YuNet không tìm thấy khuôn mặt → **không kiểm tra deepfake** cho ảnh đó (giao diện ghi “Không áp dụng”); ảnh có EXIF xoay thì ứng dụng tự xoay đúng trước khi phân tích.

## 7. Đối chiếu mục tiêu và quy tắc đăng ký trước
### 7.1 Mục tiêu (`QUALITY_TARGETS.yaml`, ghi trước final test) — kết quả `reports/test/targets_check.json`
| Mục tiêu | Ở đâu | Giá trị | Ngưỡng | Đạt? |
|---|---|---|---|---|
| A: AUROC mức ảnh ≥ 0,75 | ctrl_mei | 0,963 | 0,75 | ✅ |
| A: AUROC mức ảnh ≥ 0,75 | Korus | 0,622 | 0,75 | ❌ |
| A: AUROC mức ảnh ≥ 0,75 | CocoGlide | 0,722 | 0,75 | ❌ |
| A: FPR tại `high` ≤ 10 % | ctrl_mei / Korus / CocoGlide | 0,5 % / 2,0 % / 5,7 % | 10 % | ✅ |
| A: báo cáo F1 pixel | 3 tập | mục 4.1 | — | ✅ |
| B: AUROC ≥ 0,85 trên generator holdout | OpenFake | 0,916 | 0,85 | ✅ |
| B: FPR tại `high` ≤ 5 % trên ảnh thật | ctrl_mei / Korus / OpenFake / IMDB-WIKI | 1,0 % / 2,7 % / 0,7 % / 0,0 % | 5 % | ✅ |
| B: FPR tại `high` ≤ 5 % trên ảnh thật | COCO (CocoGlide) | 8,6 % | 5 % | ❌ |
| C: AUROC ≥ 0,80 | hoán đổi mặt / vẽ lại mặt | 0,941 / 0,889 | 0,80 | ✅ |
| C: FPR tại `high` ≤ 10 % | mặt thật / ảnh thật khác (4 nguồn) | 4,2 % / 0–2,7 % | 10 % | ✅ |
| CPU: nạp mô hình ≤ 30 s | Linux 2 vCPU / Windows Ryzen 7 4800H | 8,4 s / 17,0–26,5 s | 30 s | ✅ |
| CPU: ảnh 12 MP đầu-cuối p95 ≤ 60 s | Linux 2 vCPU | 10,2 s | 60 s | ✅ |

### 7.2 Quy tắc bật mặc định (cận dưới CI 95 % AUROC > 0,60 trên mọi tập final test chính; áp dụng cơ học bởi `research/make_validation_summary.py`, P05)
| Bộ phát hiện | Cận dưới CI 95 % trên từng tập | Kết quả | Trong ứng dụng |
|---|---|---|---|
| Mesorch | ctrl_mei 0,947 · CocoGlide 0,695 · **Korus 0,585** | không đạt trên Korus | bật, gắn **THỬ NGHIỆM** + giải thích (quy tắc cho phép “tắt hoặc gắn nhãn thử nghiệm”) |
| Community Forensics | OpenFake 0,900 · SD 1.5 1,000 | đạt | bật |
| YuNet + CLIP | hoán đổi mặt 0,925 · vẽ lại mặt 0,867 | đạt | bật |

Trạng thái và lời giải thích hiển thị nằm trong `reports/validation_summary_vi.json` (chỉ để hiển thị; không đổi điểm/ngưỡng). File này cũng ghi SHA-256 cấu hình đã đánh giá: nếu ai đó đổi ngưỡng/bộ phát hiện, mọi kết quả sẽ có cảnh báo.

## 8. Khi nào ứng dụng sai (bằng chứng: `reports/test/failure_analysis.json`)
1. **Chỉnh sửa tay chuyên nghiệp** (Korus): 54,1 % ảnh sửa nhận “Chưa thấy dấu hiệu rõ”, chỉ 16,2 % bị “Nghi vấn”.
2. **Vùng sửa nhỏ**: copy-move nhỏ bị bỏ sót 32 %, xóa/ghép nhỏ 12 % (ctrl_mei); ảnh lớn bị thu về 512 px trước khi phân tích.
3. **Inpainting AI cục bộ** (CocoGlide): 15,8 % bỏ sót ở cấp ứng dụng; bản đồ nhiệt định vị được khoảng một nửa (F1 ≥ 0,5 ở 48,2 % ảnh).
4. **Generator AI mới**: flux.2-klein-9b bị bỏ sót hoàn toàn 37,4 % (chỉ 9,2 % bị “Nghi vấn”); z-image-turbo 13,6 %, veo-3 14,7 %, gpt-image-1.5 11,3 % bỏ sót.
5. **Ảnh thật nhỏ/nén nhiều** (COCO, ImageNet, chân dung IMDB-WIKI): báo nhầm “Nghi vấn” 7–13,5 % ở cấp ứng dụng — cao hơn ảnh camera gốc (4,0 %).
6. **Ảnh rất lớn**: 2 ảnh PNG > 25 MiB bị từ chối (giới hạn cấu hình được, đổi giới hạn cần kiểm tra RAM).
7. Ngoài phạm vi đã khai báo (không đo ở đây): ảnh chụp màn hình, đồ họa, tài liệu, ảnh chủ yếu là chữ — audit v1 cho thấy đồ họa bị chấm ~1,0.

## 9. Tính toàn vẹn của lượt chạy
- Mã chấm điểm: đoạn ảnh 1–1 257 chạy bằng mã P04b (dấu vân tay `e08d26ef…`), đoạn 1 258–4 940 bằng mã P08 (`57d8c4c2…`); các thay đổi sau khóa không chạm vào tiền xử lý/mô hình/ngưỡng (EXPERIMENT_LOG P04b–P08). Kiểm tra tương đương bằng mã cuối (`research/recheck_scores.py`, seed 20261013): 150 ảnh ngẫu nhiên (37 thuộc đoạn đầu, 113 đoạn sau) được chấm lại — chênh lệch điểm lớn nhất **0,0** (trùng từng bit), 0 khác biệt về kết luận (`reports/test/recheck_final_code.json`).
- Dự đoán từng ảnh: `reports/test/test_v2_product_predictions.jsonl`; chỉ số: `reports/test/metrics_test_v2_standard.json`, `..._strict.json`; SHA-256: `reports/test/SHA256SUMS.txt`.

## 10. Tái lập
```
python research/eval_product.py --config config.json --manifest <ctrl_mei> --manifest <korus_png> --manifest <cocoglide> \
    --manifest <openfake> --manifest <dff_faceswap> --manifest <dff_extra> --split test --split test_korus --split test_cocoglide \
    --threads 2 --out test_v2.jsonl
python research/final_report.py --run test_v2.jsonl --manifest ... --config config.json --splits test test_korus test_cocoglide \
    --out reports/test/metrics_test_v2_standard.json
python research/final_report.py ... --config research/configs/config_strict.json --recompute-decisions --out .../metrics_test_v2_strict.json
python research/failure_analysis.py --run test_v2.jsonl --manifest ... --config config.json --splits ... --out .../failure_analysis.json
python research/make_validation_summary.py --metrics .../metrics_test_v2_standard.json --config config.json \
    --out-summary reports/validation_summary_vi.json --out-targets .../targets_check.json
python research/render_tables.py --v2 ... --strict ... --v1 ... --ba ... --robust ... --out docs/_tables_test.md
python research/recheck_scores.py --run test_v2.jsonl --manifest ... --config config.json --n 150 --out .../recheck_final_code.json
```
Đường dẫn dữ liệu là tham số; dữ liệu không nằm trong gói (giấy phép), chỉ có manifest + hash.
