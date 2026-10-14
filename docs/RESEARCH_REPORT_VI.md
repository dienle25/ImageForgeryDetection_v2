# RESEARCH_REPORT_VI — Khảo sát, đối chứng ứng viên và lựa chọn mô hình

_Mọi lựa chọn trong báo cáo này dựa trên **tập development** (dev). Kết quả test độc lập (chạy một lần sau khi khóa cấu hình) ở `docs/QUALITY_REPORT_VI.md`. Mã thí nghiệm (E…, P…) tra trong `EXPERIMENT_LOG.jsonl`; số liệu thô trong `reports/evidence/`._

## 1. Câu hỏi và ràng buộc
- Ba bài toán tách biệt (TASK_CONTRACT): **A** chỉnh sửa cục bộ (ghép, copy-move, xóa/inpainting kể cả AI), **B** ảnh sinh hoàn toàn bằng AI, **C** deepfake/hoán đổi khuôn mặt. Biến đổi thông thường (resize, nén, chỉnh sáng) là **âm**.
- Ràng buộc: chạy CPU tại máy (laptop Windows), không gửi ảnh ra ngoài, giấy phép mã **và** trọng số phải cho phép đánh giá trong công ty, trọng số phải tải được và nạp được.
- Không có dữ liệu công ty; không có GPU → **không fine-tune** (không có dữ liệu đại diện để fine-tune có kiểm chứng; fine-tune trên dữ liệu công khai dễ quá khớp với chính các bộ đánh giá).

## 2. Khảo sát (≥ 3 phương án mỗi bài toán)
| Bài toán | Ứng viên | Giấy phép mã / trọng số | Khả dụng CPU | Trạng thái |
|---|---|---|---|---|
| A | **PSCC-Net** (baseline v1) | MIT / không ghi | nhanh | **chạy đối chứng** (E06, E08) |
| A | **Mesorch** (AAAI 2025) | MIT / checkpoint chính thức | ~2–3 s/ảnh | **chạy đối chứng → CHỌN** |
| A | **CAT-Net v2** (IJCV 2022) | Apache-2.0 / CC-BY-4.0 | 10–12 s/ảnh 1,5 MP, tăng theo độ phân giải | **chạy đối chứng** (E07, E08) |
| A | IML-ViT | MIT | nặng (ViT-B 1024) | khảo sát, chưa chạy (ưu tiên CPU cho 3 ứng viên trên) |
| A | TruFor | chỉ phi lợi nhuận | — | loại vì giấy phép |
| A | MVSS-Net, SAFIRE | không có giấy phép | — | loại |
| B | **Community Forensics ViT-S/16@384** (CVPR 2025) | MIT / MIT | ~0,3 s/ảnh | **chạy đối chứng → CHỌN** |
| B | **SPAI** (CVPR 2025) | Apache-2.0 / Apache-2.0 | 1,3–1,5 s **mỗi ô 224 px** → 28 s trung vị, 100 s p95 | **chạy đối chứng** (tập con phân tầng) |
| B | NPR, Effort, TrueMedia | không giấy phép / CC BY-NC / không công bố trọng số | — | loại |
| C | **CLIP ViT-L/14 LN-tuning (yermandy)** | MIT / MIT (nền CLIP OpenAI MIT; huấn luyện FF++) | ~0,7–1 s/khuôn mặt | **chạy → CHỌN** |
| C | SBI, Effort, các mô hình DeepfakeBench | chỉ nghiên cứu / CC BY-NC | — | loại vì giấy phép |
| C | YuNet (phát hiện mặt, hỗ trợ) | MIT | ~50 ms | dùng |

## 3. Dữ liệu dev dùng để chọn
`ctrl_mei` dev (418 ảnh camera, 110 nhóm nguồn), `dev_korus` (72 cặp sửa tay thực tế), OpenFake dev (455 thật / 114 sinh, 10 generator), DeepFakeFace dev (300 nhóm). Chi tiết, chia tập và kiểm tra rò rỉ: `docs/DATA_PROTOCOL.md`.

## 4. Kết quả dev
### 4.1 Bài toán A (chỉnh sửa cục bộ) — AUROC mức ảnh [CI 95 % bootstrap theo nhóm]
| Ứng viên (thống kê điểm) | ctrl_mei dev | dev_korus | F1 pixel (ctrl / Korus) | Ghi chú |
|---|---|---|---|---|
| PSCC-Net, chính sách tốt nhất | 0,53–0,56 (CI chứa 0,5) | 0,494 [0,468; 0,520] | 0,324* / 0,152 | tiền xử lý 256 bicubic cải thiện định vị nhưng không cải thiện phát hiện (E06) |
| CAT-Net v2 (max bản đồ) | 0,761 [0,725; 0,797] | **không hoàn thành** (6/144 ảnh; ~42 s/ảnh 2 MP trên CPU 2 lõi) | 0,571 / — | sụp khi ảnh bị resize sau sửa (½ + JPEG85: 0,36) và với xóa vùng (0,60) |
| **Mesorch (max bản đồ)** | **0,945 [0,923; 0,966]** | **0,582 [0,519; 0,634]** | **0,842 / 0,168** | ổn định qua hậu xử lý (0,91–0,98); vùng nhỏ 0,89 |

\* F1 pixel PSCC trên ctrl dev ở chính sách 256 bicubic (E06); với tiền xử lý v1 (cạnh dài 512) chỉ 0,228 và kích hoạt nhầm 48 % diện tích ảnh gốc.

Chọn thống kê điểm ảnh của Mesorch (mô hình không có head ảnh): `max` (0,945/0,582) ≈ `p999` (0,943/0,582) > `top1pct_mean` (0,911/0,588) > `mean` (0,767/0,556). Chọn `max`.

**Kết luận A:** Mesorch tốt nhất trên cả hai tập dev và bền với nén/resize; nhưng trên chỉnh sửa tay thực tế chất lượng cao (Korus) **mọi** ứng viên gần may rủi. Số liệu ctrl_mei lạc quan vì thao tác script giống augmentation huấn luyện của Mesorch (lý do sửa đổi P02 trước khi chấm test).

### 4.2 Bài toán B (ảnh sinh AI)
| Ứng viên | OpenFake dev (10 generator) | Tốc độ CPU |
|---|---|---|
| **Community Forensics** | **AUROC 0,963 [0,945; 0,978]**, AP 0,906, TPR@FPR5 % 0,81 | 0,36 s p50 |
| SPAI (tập con phân tầng, 43 ảnh đã chạy xong) | AUROC 0,797; cùng 43 ảnh Community Forensics 0,968; chênh lệch +0,17 [0,05; 0,32] | 28 s p50, 103 s p95 |

Generator khó cho Community Forensics: gpt-image-2 (TPR@FPR5 % 0,33), nano-banana-pro 0,57, sora-2 0,64. Ảnh thật DOCCI (độ phân giải cao) có điểm cao hơn ảnh ImageNet → ngưỡng phải đặt theo nguồn âm khó nhất.

SPAI chỉ chạy được 43/137 ảnh của tập con trong thời gian cho phép vì chi phí CPU; với chênh lệch có CI không chứa 0 và tốc độ chậm ~100 lần, SPAI không phù hợp cho laptop CPU. Đây là kết luận trên dữ liệu này, không phải đánh giá chung về SPAI.

### 4.3 Bài toán C (deepfake khuôn mặt)
CLIP LN-tuning (yermandy) + YuNet, điểm ảnh = max trên ≤ 3 khuôn mặt lớn nhất, ảnh không thấy mặt = không gắn cờ: DFF dev hoán đổi InsightFace vs ảnh thật **AUROC 0,937 [0,915; 0,956]**, AP 0,955, TPR@FPR5 % 0,80 (E10). Mô hình huấn luyện trên FF++ (video), DFF là ảnh tĩnh khác miền → kết quả test là chuyển miền thật.

## 5. Kết quả âm tính quan trọng (được giữ trong báo cáo, không giấu)
1. **Sửa tiền xử lý PSCC-Net không cứu được phát hiện** (E06): nguyên nhân F01/F02 đã tìm ra (độ phân giải huấn luyện 256) nhưng mô hình vẫn ≈ may rủi trên ảnh camera.
2. **Chỉnh sửa tay chuyên nghiệp gần như không bị phát hiện** bởi mọi ứng viên có giấy phép phù hợp (Korus dev ≤ 0,58).
3. **CAT-Net phụ thuộc lưới JPEG**: mất tác dụng khi ảnh bị thu nhỏ sau khi sửa — trường hợp rất phổ biến khi ảnh đi qua mạng xã hội/ứng dụng chat.
4. **SPAI không khả thi trên CPU** ở độ phân giải gốc.
5. **Tập dữ liệu công khai có lỗi dòng dõi** (DFF dùng chung ảnh nền giữa danh tính) → đã phát hiện và loại trước khi chấm test (P03).

## 6. Bộ cấu hình được khóa cho final test
Mesorch (max) + Community Forensics + YuNet/CLIP; ngưỡng từ `research/derive_thresholds.py` trên dev (FPR ≤ 5 % → “Nghi vấn”, FPR ≤ 20 % → “Cần kiểm tra thêm”, trên **từng** nguồn âm). Hash cấu hình và thời điểm khóa: mục P04 trong `EXPERIMENT_LOG.jsonl`.

## 7. Đối chiếu với final test (sau khi khóa cấu hình)
Lựa chọn trên dev được giữ nguyên; final test (một lần, `docs/QUALITY_REPORT_VI.md`) xác nhận thứ tự dự kiến: Mesorch tốt trên thao tác kiểu script (0,963) và trung bình trên inpainting AI chưa thấy (CocoGlide 0,722), nhưng chỉnh sửa tay chuyên nghiệp vẫn gần may rủi (Korus 0,622, cận dưới CI 0,585 → nhãn THỬ NGHIỆM theo quy tắc đăng ký trước); Community Forensics giảm từ 0,963 (dev) xuống 0,916 trên 10 generator khác hoàn toàn; bộ khuôn mặt ổn định (0,937 dev → 0,941 test). CAT-Net trên Korus không chạy hết vì chi phí CPU; vì CAT-Net phụ thuộc lưới JPEG và kém Mesorch rõ trên ctrl_mei dev, kết luận chọn mô hình không phụ thuộc vào số còn thiếu này.

## 8. Hướng cải thiện có căn cứ (chưa làm)
- Thu thập **ảnh công ty có nhãn** (cả ảnh thật bị báo nhầm và ảnh giả bị bỏ sót) → hiệu chỉnh ngưỡng theo chi phí lỗi thực tế; đây là bước bắt buộc trước khi dùng cho quyết định tài chính.
- Suy luận theo ô ở độ phân giải gốc cho Mesorch (vùng sửa nhỏ, ảnh lớn) — cần đánh giá vì lệch phân phối huấn luyện và tốn CPU.
- Tinh chỉnh (fine-tune) chỉ khi có dữ liệu đại diện + GPU + tập test độc lập.
