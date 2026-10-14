# MODEL_CARD — Image Forgery Detection v2 (bộ ba detector, CPU, ONNX)

## Tổng quan
| | Chỉnh sửa cục bộ (A) | Ảnh sinh AI hoàn toàn (B) | Deepfake khuôn mặt (C) |
|---|---|---|---|
| Khóa cấu hình | `mesorch` | `commfor_aigen` | `face_clip` (+ YuNet) |
| Kiến trúc | Mesorch: ConvNeXt + SegFormer (MiT-B3) đa tần số DCT, cổng trộn theo pixel | ViT-S/16 @384, head tuyến tính 1 logit | YuNet → crop vuông 1,3× → CLIP ViT-L/14 (LN-tuning) → L2-norm → Linear(1024,2) |
| Tham số | 85,8 M | ~22 M | 304 M (+ YuNet 0,08 M) |
| File ONNX | `models/mesorch.onnx` 349 MB, fp32 | `models/commfor_vits16_384.onnx` 87 MB, fp32 | `models/face_clip_vitl14.onnx` 607 MB (trọng số bf16 như bản gốc, tính fp32) |
| Đầu vào | RGB → 512×512 `cv2.INTER_LINEAR`, chuẩn ImageNet | cạnh ngắn 440 (bilinear) → cắt giữa 384, chuẩn ImageNet | khuôn mặt (≤ 3 mặt lớn nhất) → cạnh ngắn 224 bicubic → cắt giữa 224, chuẩn CLIP |
| Đầu ra | bản đồ sigmoid 512×512 (thang cố định) → phóng về kích thước ảnh; điểm ảnh = **max** bản đồ | sigmoid(logit) | softmax[fake] mỗi mặt; điểm ảnh = max |
| Vùng nghi vấn | có (heatmap, mask ngưỡng 0,5) | **không** | khung khuôn mặt |
| Không áp dụng | — | — | không thấy mặt (YuNet ≥ 0,7) |
| Nguồn | github.com/scu-zjz/Mesorch | HF OwensLab/commfor-model-384 | HF yermandy/deepfake-detection; OpenCV Zoo YuNet 2023mar |
| Giấy phép | mã MIT; trọng số không ghi riêng | MIT | MIT |
| Dữ liệu huấn luyện (theo tác giả) | CASIA v2, Fantastic Reality, IMD2020, tampCOCO | Community Forensics (ảnh từ hàng nghìn generator, ~2024) | FaceForensics++ |

Chuyển đổi ONNX kiểm chứng số: Mesorch |Δ| 8,7·10⁻⁷; CLIP |Δ| 1,9·10⁻⁶ (mô-đun dựng lại khớp **bit** TorchScript ở bf16); Community Forensics |Δ logit| 5,8·10⁻⁶ (4 cặp đầu vào ngẫu nhiên). SHA-256 từng file: `models/manifest.json`.

## Mục đích sử dụng
- **Sàng lọc hỗ trợ người thẩm định**: xếp ưu tiên ảnh cần xem kỹ, chỉ vùng nghi vấn, ghi bằng chứng (hash, cấu hình, mô hình).
- **Không dùng** làm căn cứ duy nhất cho quyết định tài chính, pháp lý, nhân sự; không dùng để “chứng nhận ảnh thật”.
- Ngoài phạm vi: ảnh chụp màn hình, đồ họa, tài liệu/chứng từ/văn bản, video, metadata, ảnh thật gắn sai ngữ cảnh.

## Ngưỡng và ý nghĩa điểm
- Điểm là **điểm thô**, chưa hiệu chỉnh thành xác suất. Ngưỡng chọn trên **dev**: `high` = FPR ≤ 5 % trên từng nguồn ảnh thật của dev → “Nghi vấn”; `low` = FPR ≤ 20 % → “Cần kiểm tra thêm”. Giá trị và trạng thái trong `config.json`.
- Kết luận chung = mức cao nhất của các detector áp dụng được. Vì có 3 detector, tỷ lệ ảnh thật bị gắn “Nghi vấn” ở cấp ứng dụng có thể **cao hơn 5 %** — số đo thật ở `docs/QUALITY_REPORT_VI.md` (mục “cấp ứng dụng”).

## Hiệu năng (final test độc lập, chạy một lần với cấu hình khóa — chi tiết `docs/QUALITY_REPORT_VI.md`)
| Bộ phát hiện | Tập test | AUROC [CI 95 %] | Tại `high` (“Nghi vấn”): TPR / FPR | Trạng thái theo quy tắc bật mặc định |
|---|---|---|---|---|
| Mesorch | ctrl_mei (sửa bằng script) | 0,963 [0,947; 0,978] | 78,0 % / 0,5 % | **THỬ NGHIỆM** (không đạt trên Korus) |
| Mesorch | CocoGlide (inpainting AI) | 0,722 [0,695; 0,750] | 26,4 % / 5,7 % | |
| Mesorch | Korus (sửa tay chuyên nghiệp) | 0,622 [0,585; 0,666] | 10,1 % / 2,0 % | |
| Community Forensics | OpenFake, 10 generator không có trong dev | 0,916 [0,900; 0,930] | 47,8 % / 0,7 % | đạt |
| Community Forensics | DeepFakeFace SD 1.5 | 1,000 | 100 % / 0,0 % | |
| YuNet + CLIP | DeepFakeFace hoán đổi mặt | 0,941 [0,925; 0,958] | 81,2 % / 4,2 % | đạt |
| YuNet + CLIP | DeepFakeFace vẽ lại mặt (SD) | 0,889 [0,867; 0,909] | 58,3 % / 4,2 % | |

Cấp ứng dụng (cả ba bộ cùng chạy): 9,0 % ảnh thật (gộp 1 784 ảnh, 5 nguồn) bị “Nghi vấn”, 34,8 % bị “Cần kiểm tra thêm” trở lên; profile nghiêm 2,1 % / 9,0 %. Dev (chọn mô hình): `docs/RESEARCH_REPORT_VI.md` §4. CPU: `docs/CPU_BENCHMARK_VI.md`.

## Giới hạn đã biết (có bằng chứng)
1. **Chỉnh sửa tay chuyên nghiệp** trên ảnh chất lượng cao (Korus): test AUROC 0,622; 54 % ảnh sửa nhận “Chưa thấy dấu hiệu rõ”.
2. **Vùng sửa nhỏ** (0,3–1,5 % diện tích): chỉ 54 % bị “Nghi vấn” (vùng vừa/lớn 88–93 %); ảnh rất lớn bị thu về 512 px.
3. **Generator AI mới**: flux.2-klein-9b chỉ 9 % bị “Nghi vấn” (AUROC 0,776); z-image-turbo, gpt-image-1.5, veo-3 41–56 %.
4. **Ảnh thật nhỏ/nén nhiều** (COCO, ImageNet, chân dung IMDB-WIKI): báo nhầm cao hơn ảnh camera gốc (cấp ứng dụng 7–13,5 % so với 4 %); ảnh COCO bị báo “ảnh AI” 8,6 % (không đạt mục tiêu ≤ 5 %).
5. Deepfake khuôn mặt huấn luyện trên video FF++; ảnh tĩnh, mặt nhỏ (< 64 px), góc nghiêng, nhiều người: độ tin cậy thấp hơn; chỉ 3 mặt lớn nhất được kiểm tra.
6. Mesorch tự chuẩn hóa min-max **đặc trưng DCT đầu vào** theo từng ảnh (một phần kiến trúc đã huấn luyện); bản đồ **đầu ra** không bị chuẩn hóa theo ảnh.
7. Chưa đánh giá trên dữ liệu công ty; giấy phép trọng số/dữ liệu huấn luyện cần pháp lý xem xét trước khi dùng thương mại (`LICENSES_VI.md`).

## Công bằng / quyền riêng tư
Không đánh giá được sai lệch theo nhân khẩu học (không có nhãn); bộ deepfake kiểm thử là người nổi tiếng IMDB-WIKI, lệch về chân dung chất lượng tốt. Ảnh chỉ xử lý cục bộ.
