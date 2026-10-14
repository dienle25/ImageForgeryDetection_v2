# DATA_PROTOCOL — Dữ liệu, nguồn gốc, chia tập, chống rò rỉ

_Viết 12/10/2026. Áp dụng cho mọi số liệu trong `docs/QUALITY_REPORT_VI.md`. Gói bàn giao **không chứa ảnh dữ liệu**; chỉ có manifest (đường dẫn tương đối, nhãn, split, hash, nguồn, giấy phép) trong `reports/manifests/`._

## 1. Nguyên tắc
1. **Không có dữ liệu công ty.** Mọi kết quả là trên dữ liệu công khai/kiểm soát; mức phù hợp cho công ty **chưa được xác nhận**.
2. Development (dev) dùng để: so sánh ứng viên, chọn thống kê điểm, chọn ngưỡng. Final test chạy **một lần** bằng đúng ứng dụng đóng gói (CLI/ONNX) với `config.json` đã khóa (ghi hash trong `EXPERIMENT_LOG.jsonl`).
3. Mọi biến thể của cùng ảnh nguồn (ảnh gốc, ảnh sửa, các mức nén/resize) và ảnh donor nằm **cùng một split**.
4. Kiểm tra trùng: bytes (SHA-256 file), điểm ảnh sau giải mã (SHA-256 của mảng RGB sau EXIF/alpha), gần trùng (dHash 64-bit + xác minh PSNR/tương quan), và dòng dõi ảnh nền (face-swap).
5. Loại trừ chỉ làm **trước khi chấm test**, ghi lý do trong manifest (`exclude`) và trong `EXPERIMENT_LOG.jsonl` (P02, P03). Không bỏ ảnh vì mô hình chấm sai.

## 2. Nguồn dữ liệu

| Tập | Nguồn | Nội dung dùng | Nhãn | Giấy phép / điều kiện | Vai trò |
|---|---|---|---|---|---|
| `ctrl_mei` (tự tạo) | ảnh camera từ github.com/drewnoakes/metadata-extractor-images | 220 ảnh JPEG camera (EXIF make/model, không có tag phần mềm chỉnh sửa), 204 model máy | ghép (splice), copy-move, xóa (OpenCV TELEA) có mask; ảnh gốc đi kèm cùng hậu xử lý | repo ghi “free to use these media files however you wish” | A: dev + test |
| Korus Realistic Tampering | P. Korus & J. Huang (2016–2017) | 220 cặp gốc/sửa, 4 máy ảnh, 1920×1080 TIFF không nén | sửa tay thực tế (chèn/xóa đối tượng), mask GT | “educational and research purposes only” | A: dev_korus + test_korus |
| CocoGlide | GRIP-UNINA (bài TruFor) | 512 ảnh COCO 256×256 bị inpaint bằng GLIDE + 512 ảnh thật tương ứng | mask vùng inpaint | ảnh COCO theo giấy phép Flickr (nhiều CC BY-NC); chỉ đánh giá nghiên cứu | A: test (giữ ngoài hoàn toàn) |
| OpenFake | Hugging Face ComplexDataLab/OpenFake | test shards 0 & 7, 12 row-group (1767 ảnh) | ảnh sinh bởi 20 generator 2025–2026 / ảnh thật ImageNet & DOCCI | CC BY-NC 4.0 – chỉ đánh giá nghiên cứu | B: dev (10 generator) / test (10 generator khác) |
| DeepFakeFace (DFF) | Hugging Face OpenRL/DeepFakeFace | wiki (thật, IMDB-WIKI), insight (InsightFace swap), inpainting (SD inpaint mặt), text2img (SD 1.5) | theo thư mục | thẻ dữ liệu ghi openrail/apache-2.0; ảnh thật IMDB-WIKI chỉ cho nghiên cứu | C: dev/test; B: text2img cross-dataset |

Không dùng: CASIA (chỉ lấy được mask, không có ảnh), COVERAGE (chỉ tải được README), dữ liệu cần tài khoản (credentials `.kaggle` của người dùng **không** được dùng).

## 3. Tập kiểm soát `ctrl_mei` (tự sinh, có lineage)
- Script: `research/make_controlled_set.py`, seed **20261012**, cạnh dài tối đa 1536 (ảnh làm việc = ảnh camera thu nhỏ + JPEG q92).
- Thao tác: splice (dán vùng từ ảnh donor **cùng split**, biên làm mềm), copy-move (vùng trong cùng ảnh, xoay/scale nhẹ), removal (OpenCV TELEA bán kính 5 trên mask nở). Kích thước vùng: nhỏ 0,3–1,5 %, vừa 1,5–6 %, lớn 6–20 % diện tích.
- Hậu xử lý (cùng một cấu hình cho ảnh sửa và ảnh gốc đi cặp → nén/resize không thành “lối tắt” nhãn): JPEG 95/85/75, PNG, thu nhỏ ½ + JPEG 85, cạnh dài 1024 + JPEG 90.
- Chia: nhóm gần trùng của ảnh nguồn (aHash) xáo seed rồi luân phiên dev/test → dev 110 nhóm, test 109 nhóm; 440 ảnh mỗi split (220 sửa/220 gốc).
- Hạn chế: thao tác là script, **không** phải chỉnh sửa chuyên nghiệp hay inpainting bằng mạng nơ-ron; giống augmentation lúc huấn luyện của Mesorch → số liệu trên tập này lạc quan cho Mesorch (lý do sửa đổi P02).
- Loại trừ (P03): 3 ảnh “sửa” có điểm ảnh **giống hệt** ảnh gốc (vùng đồng màu, thao tác không đổi pixel) → không phải ảnh chỉnh sửa (1 dev, 2 test).

## 4. Korus — chia theo cặp cảnh (P02, trước khi chấm)
- TIFF được chép sang PNG không mất dữ liệu (`research/prepare_korus.py`, hash pixel trùng khớp 440/440).
- Chia theo cặp (gốc + sửa cùng nhóm), phân tầng theo máy, seed 20261012: **dev_korus** 18 cặp/máy (144 ảnh), **test_korus** 37 cặp/máy (296 ảnh).
- `test_korus` cùng phân phối với `dev_korus` (ngưỡng đã thấy dữ liệu Korus) → báo riêng, không gọi là “ngoài phân phối”. CocoGlide là tập ngoài thật sự cho bài toán A.

## 5. OpenFake — holdout generator
- Dev: ảnh thật 455 (ImageNet 299, DOCCI 156) + ảnh sinh 114 từ 10 generator (wan-video-2.5, ernie-image-turbo, sora-2, gpt-image-2, recraft-v3, aurora, nano-banana-pro, lumina, ideogram-2.0, frames).
- Test: ảnh thật 426 + ảnh sinh 772 từ 10 generator **không xuất hiện trong dev** (z-image-turbo, flux.2-klein-9b, illustrious, gpt-image-1.5, midjourney-7, veo-3, halfmoon, seedream-v5.0, ernie-image, recraft-v2). Lưu ý họ mô hình gần nhau: ernie-image ~ ernie-image-turbo, recraft-v2 ~ recraft-v3.

## 6. DeepFakeFace — nhóm theo danh tính + dòng dõi ảnh nền
- Nhóm theo tên file (danh tính IMDB-WIKI); dev 300 nhóm, test 500 nhóm; mỗi nhóm có wiki/insight/inpainting/text2img.
- Kiểm tra dòng dõi (P03): ảnh **insight không được tạo từ ảnh wiki cùng tên** (tương quan ≈ 0); 14 ảnh insight dùng ảnh wiki của danh tính khác làm nền, 8 trường hợp vượt dev/test → loại **4 nhóm test** (20 dòng). Ảnh **inpainting được tạo từ ảnh wiki cùng tên** (730/800 khớp).
- 2 ảnh wiki dev không phải JPEG/PNG theo nội dung → ứng dụng từ chối (đúng thiết kế), không tính vào chỉ số.

## 7. Kết quả kiểm tra trùng (bằng chứng: `reports/evidence/dedup_*.json`)
- 7 311 ảnh: **0** bộ trùng bytes và **0** bộ trùng pixel vượt split hoặc nhóm (42 bộ trùng trong cùng nhóm = ảnh gốc giống nhau của 2 cặp sửa cùng hậu xử lý; khi tính chỉ số mỗi ảnh pixel-giống-nhau chỉ tính một lần).
- Gần trùng dHash ≤ 4 vượt split: 58 cặp → xác minh bằng PSNR/tương quan 128 px: chỉ các cặp DFF ở mục 6 là trùng thật; còn lại là ảnh ít chi tiết/đồng màu va chạm hash.
- Không kiểm tra được trùng với **dữ liệu huấn luyện của các mô hình đã huấn luyện sẵn** (CASIA v2, IMD2020, COCO, FF++, …): CocoGlide dùng ảnh COCO, có thể trùng với dữ liệu tiền huấn luyện → ghi là giới hạn.

## 8. Định nghĩa chỉ số
- Ảnh: TP/TN/FP/FN, precision, recall (TPR), F1, FPR, FNR tại ngưỡng **high** (kết luận “Nghi vấn”) và ngưỡng **low** (từ “Cần kiểm tra thêm”); AUROC (Mann–Whitney, xử lý ties), AP; CI 95 % bootstrap theo **nhóm nguồn** (1000 lần, seed 0).
- Vùng “Cần kiểm tra thêm” (low ≤ điểm < high) = abstain; báo **coverage** = tỷ lệ ảnh có kết luận rõ (ngoài vùng này) và độ chính xác trên phần được kết luận.
- Pixel (chỉ bài toán A, ảnh có mask): ngưỡng pixel cố định 0,5 (chọn trước, không tối ưu trên test); F1/IoU macro (trung bình theo ảnh có mask khác rỗng), micro (cộng TP/FP/FN toàn tập), AUC pixel trung bình. **Chính sách mask rỗng**: ảnh gốc không đưa vào macro-F1; thay vào đó báo riêng **tỷ lệ kích hoạt nhầm** = % pixel ≥ 0,5 trên ảnh gốc, và % ảnh gốc có > 1 % pixel kích hoạt.
- Không hiệu chỉnh xác suất; điểm là điểm thô của mô hình.

## 9. Tái lập
```
python research/make_controlled_set.py ...           # ctrl_mei (seed 20261012)
python research/prepare_korus.py RAW_KORUS OUT        # PNG + split P02
python research/build_manifests.py ...                # CocoGlide, Korus (gốc)
python research/dedup_check.py --manifest ... --out dedup_report.json
python research/eval_product.py --config config.json --manifest ... --split dev --split dev_korus --out dev.jsonl
python research/derive_thresholds.py --dev dev.jsonl --manifest ... --out thresholds.json
```
Đường dẫn dữ liệu là tham số dòng lệnh; không có đường dẫn máy người dùng nào được ghi cứng.
