# TASK_CONTRACT — Định nghĩa nhãn, phạm vi và ý nghĩa điểm

_Phiên bản 1 — viết ngày 12/10/2026 **trước** khi chạy final test. Chưa được công ty phê duyệt._

## 1. Ba bài toán tách biệt (không gộp thành một "xác suất giả")

| Mã | Bài toán | Nhãn 1 (dương) | Nhãn 0 (âm, đối chứng bắt buộc) | Đầu ra |
|---|---|---|---|---|
| A | `local_manipulation` — chỉnh sửa cục bộ | Nội dung một phần ảnh bị thay đổi so với lúc chụp: ghép từ ảnh khác (splicing), sao chép vùng trong cùng ảnh (copy-move), xóa/thay thế đối tượng (removal/inpainting, kể cả inpainting bằng AI) | Ảnh gốc và ảnh chỉ qua xử lý toàn cục thông thường: resize, nén lại JPEG/PNG, chỉnh sáng/màu/độ tương phản toàn ảnh, crop, xoay theo EXIF | Điểm toàn ảnh + bản đồ nghi vấn (heatmap) theo tọa độ ảnh hiển thị |
| B | `ai_generated` — ảnh sinh hoàn toàn bằng AI | Toàn bộ ảnh do mô hình sinh tạo (text-to-image, image generation) | Ảnh chụp thật (kể cả đã nén/resize) | Chỉ điểm toàn ảnh. **Không có bản đồ vùng** — không bịa "vùng bị ghép" |
| C | `face_deepfake` — deepfake/hoán đổi khuôn mặt | Khuôn mặt bị hoán đổi/tái tạo/inpainting bằng công cụ chỉnh danh tính | Khuôn mặt thật | Điểm theo từng khuôn mặt phát hiện được; ảnh không có mặt người → "không áp dụng" |

Ảnh chỉ được inpainting một phần bằng AI thuộc bài toán A; kết quả của B trên loại ảnh này được báo riêng, không tính là đúng/sai của B.

## 2. Ngoài phạm vi / chưa hỗ trợ (phải nói rõ với người dùng)
- Ảnh chụp màn hình, đồ họa, ảnh vẽ, tài liệu scan, ảnh chứa chủ yếu chữ: khác miền huấn luyện, có thể báo nhầm (bằng chứng: fixture đồ họa của audit bị chấm ~1,0).
- Sửa văn bản/tài liệu, video, âm thanh, chỉnh sửa metadata, ảnh thật bị gán sai chú thích (cheapfake), xác minh chữ ký C2PA.
- Ảnh rất nhỏ (< 64 px cạnh ngắn) hoặc nén rất nặng: dấu vết có thể đã mất.
- Phương pháp giả mạo/generator mới ra sau dữ liệu huấn luyện của mô hình: có thể bỏ sót.

## 3. Ý nghĩa điểm
- Mọi điểm là **điểm thô của mô hình**, chưa hiệu chỉnh thành xác suất, trừ khi report ghi `calibrated: true` kèm tập hiệu chỉnh và Brier/ECE.
- Ngưỡng `low`/`high` được chọn trên tập **development**, không trên final test; ghi trạng thái trong `config.json`.
- Ba mức kết luận: `Chưa thấy dấu hiệu rõ` (< low), `Cần kiểm tra thêm` (low ≤ điểm < high), `Nghi vấn` (≥ high).
- "Chưa thấy dấu hiệu rõ" ≠ "ảnh thật". Heatmap là kích hoạt của mô hình, không phải vùng sửa đã xác minh.
- Kết luận tổng hợp lấy mức cao nhất giữa các bài toán đã kiểm tra, và luôn liệt kê bài toán **không** được kiểm tra.

## 4. Dữ liệu và chia tập (khóa trước final test)
- Mọi biến thể của cùng ảnh nguồn và donor nằm cùng split; kiểm tra trùng bytes, trùng pixel sau giải mã và near-duplicate (aHash).
- Bài toán A — development: `ctrl_mei` split `dev`. Final test độc lập: `ctrl_mei` split `test`, Korus Realistic Tampering (toàn bộ), CocoGlide (toàn bộ). Bộ audit/sample tác giả chỉ là regression.
- Bài toán B — development và test chia theo **generator** (holdout generator) trên OpenFake; thêm tập ngoài (DeepFakeFace text2img, ảnh camera MEI) làm cross-dataset.
- Bài toán C — DeepFakeFace (InsightFace swap / inpainting / ảnh thật) chia theo nhóm ảnh gốc; ghi rõ chênh lệch miền so với dữ liệu huấn luyện FF++ của mô hình.
- Không có dữ liệu công ty: **mức phù hợp cho công ty chưa được xác nhận**.

## 5. Sửa đổi P02 — ghi 12/10/2026, TRƯỚC khi chấm bất kỳ ảnh Korus/CocoGlide nào
- Lý do: tập dev tổng hợp `ctrl_mei` dùng thao tác (ghép có làm mềm biên, copy-move trong cùng ảnh, xóa bằng OpenCV inpaint) gần với augmentation lúc huấn luyện của Mesorch; chọn mô hình chỉ trên tập này có thể thiên vị Mesorch.
- Korus Realistic Tampering được chia theo **cặp cảnh** (ảnh gốc + ảnh sửa cùng nhóm), phân tầng theo máy ảnh, seed 20261012: 18 cặp/máy → `dev_korus` (72 cặp, 144 ảnh); 37 cặp/máy → `test_korus` (148 cặp, 296 ảnh). Ảnh TIFF được chép sang PNG không mất dữ liệu (kiểm tra hash pixel trùng khớp).
- Bài toán A — development = `ctrl_mei` dev + `dev_korus`. Final test = `ctrl_mei` test, `test_korus` (cùng phân phối với dev_korus — báo riêng), CocoGlide **toàn bộ** (giữ nguyên là tập ngoài hoàn toàn chưa thấy).
- Bài toán B — ngưỡng chọn trên âm của dev: ảnh thật OpenFake dev + ảnh gốc camera `ctrl_mei` dev + ảnh thật DeepFakeFace dev; final test: OpenFake test (generator chưa thấy), DeepFakeFace text2img test, ảnh thật camera/COCO của các final test.
- Bài toán C — dev: DeepFakeFace dev; final test: DeepFakeFace test (swap + inpainting) và ảnh không có mặt người của các final test (đo báo nhầm).
- Final test của **sản phẩm** được chạy bằng chính CLI/ONNX đóng gói, một lần, sau khi khóa `config.json`.
