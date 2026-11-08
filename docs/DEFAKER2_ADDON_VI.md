# DEFAKER2_ADDON_VI — Gói phụ thử nghiệm DefakerOne2

_Lập ngày 07/11/2026, cập nhật 08/11/2026 (mục 5.3–5.5, 10). Số đo trên laptop AMD Ryzen 7 4800H, RAM 8 GB (Windows thấy 7,6 GB), GPU GTX 1650 4 GB, Windows 11. File bằng chứng nằm trong `reports/evidence/defaker2_addon/` và `reports/evidence/defaker2_mesorch/` (đã che tên máy, tên người dùng và đường dẫn)._

> **Cập nhật 08/11/2026.** Thêm bản DefakerOne2 + Mesorch (mục 5.3), ngưỡng DefakerOne2 đọc từ config với mức "Không kết luận" (mục 5.4), cảnh báo "Độ tin cậy thấp" (mục 5.5), và script đo lại / chọn ngưỡng để người dùng tự chạy (mục 10). **Chưa có số đo mới nào**: bước này chỉ viết code và script; bảng kết quả ở mục 10.4 để trống ("chưa đo").

## 1. Tóm tắt

- App có thêm **thẻ thứ tư "DefakerOne2 (thử nghiệm)"**. Thẻ hiện p_fake cho cả ảnh, do mô hình [DefakerOne2](https://github.com/inclusionAI/DefakerOne2) (nền Qwen3.5-2B) tính.
- DefakerOne2 cần PyTorch và GPU, nên chạy **ngoài app**:
  - Server riêng `addons/defaker2/defaker2_server.py`, chạy trong môi trường conda `defaker2`.
  - Server chỉ nghe ở `127.0.0.1:8765`.
  - App gọi server qua HTTP bằng thư viện chuẩn Python và bỏ qua mọi proxy, nên ảnh không rời khỏi máy.
- **Gói chính không đổi**: không thêm PyTorch, không thêm trọng số, không thêm thư viện vào runtime của app. `config.json` và ngưỡng của 3 bộ phát hiện cũ giữ nguyên.
- **Điểm DefakerOne2 KHÔNG được tính vào kết luận chung.**
  - Kết luận, danh sách bài toán đã kiểm tra và cảnh báo chung chỉ dựa trên 3 bộ phát hiện đã đánh giá.
  - Hash cấu hình quyết định của bản demo trùng hash cấu hình chuẩn, nên trang vẫn ghi "Điểm vận hành: chuẩn".
- **Server tắt hoặc quá thời gian** (mặc định 120 s): thẻ báo "Chưa khả dụng", 3 bộ phát hiện cũ vẫn chạy bình thường.
- **Chế độ chạy trên máy này**: GPU, **fp16 kèm nén 8-bit** (bitsandbytes). Vì có nén, số liệu tác giả công bố có thể không còn đúng.
- **Thời gian**: mỗi ảnh dưới 60 s. Riêng phần DefakerOne2 mất khoảng 30 s với ảnh 2 MP và 42 s với ảnh 12 MP.
- **RAM: vượt mức an toàn** khi app và server chạy cùng lúc mà các ứng dụng khác vẫn mở, xem mục 6.

## 2. Kiến trúc

```
Trình duyệt ──HTTP──> App (runtime Python 3.13 + ONNX Runtime, CPU)
                        ├─ mesorch, commfor_aigen, face_clip   → kết luận chung (như cũ)
                        └─ defaker2 (type defaker2_http) ──POST PNG──> 127.0.0.1:8765/score
                                                                       Server DefakerOne2 (conda defaker2:
                                                                       Python 3.12, torch 2.9 cu128, ms-swift 4.1.0)
                                                                       → {"p_fake","p_real","device","seconds",...}
```

- **Server.** Nạp mô hình MỘT lần. Mỗi ảnh chạy đúng pipeline README qua ms-swift (giống lệnh `swift infer`):
  - Câu hỏi: `<image>Is this image real or fake? Answer only Real or Fake`.
  - `MIN_PIXELS=262144`, `MAX_PIXELS=2408448`, `--max_new_tokens 16`, `--enable_thinking False`, `--logprobs True --top_logprobs 20`.
  - p_fake/p_real tính bằng chính hàm `logprobs_to_probs()` trong `logprob2score.py` của tác giả.
  - Mỗi câu trả lời được kiểm tra không có NaN/inf.
  - Bảo vệ: kiểm tra Host (chống DNS rebinding); chỉ nhận ảnh JPEG/PNG với Content-Type `image/png`, `image/jpeg` hoặc `application/octet-stream`; giới hạn 256 MiB.
- **Detector `ifd/detectors/defaker2.py`.**
  - Gửi ảnh mà app đã đọc (đã xoay theo EXIF, nền trong suốt ghép lên nền trắng) dưới dạng PNG không mất dữ liệu.
  - Trả `DetectorResult` có `calibrated: false`, `decision: "experimental"`, `in_summary: false`, không có bản đồ nhiệt và không có ngưỡng.
  - Thông tin mô hình trong báo cáo ghi rõ trọng số nằm ngoài gói và giấy phép trọng số chưa rõ.
- **Cấu hình chỉ chấp nhận URL `http://127.0.0.1:<cổng>` hoặc `http://localhost:<cổng>`.**
- **Bộ kiểm định ảnh có nhãn** (`KIEM_DINH_CONG_TY.bat`, `ifd.company_eval`):
  - Chấm DefakerOne2 và báo trong mục riêng "Bộ phát hiện THỬ NGHIỆM": AUROC/AP, tỷ lệ theo từng loại ảnh, và số liệu tại mốc 0,5 (câu trả lời Real/Fake của chính mô hình, không phải ngưỡng).
  - Kết luận cấp ứng dụng và ngưỡng tham khảo chỉ dùng 3 bộ phát hiện đã đánh giá.
  - Ảnh mà server không chấm được thì không tính vào chỉ số, và được chấm bù ở lần chạy sau.

## 3. Cài đặt (một lần)

Chi tiết và kết quả bài chạy thử nằm ở `D:\deep_fake\DefakerOne2\KET_QUA_CHAY_THU.md`.

```bash
conda create -n defaker2 python=3.12.3 -y
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ms-swift==4.1.0 numpy==1.26.0 scipy==1.11.3 scikit-learn==1.3.2 torchao==0.16.0
python -m pip install bitsandbytes pyarrow==19.0.1 "qwen_vl_utils>=0.0.14"
```

- `pyarrow==19.0.1`: pyarrow 26 đòi NumPy ≥ 2, xung đột với `numpy==1.26.0` của README.
- `qwen_vl_utils`: ms-swift cần gói này để nạp Qwen3.5-VL.
- `bitsandbytes`: để nén 8-bit cho vừa 4 GB VRAM.
- Trọng số: thư mục `release/` trên Hugging Face `qcf568/DefakerOne2`. SHA-256 của `model.safetensors` là `9bab60b0…eff539`.
- Đường dẫn mã nguồn, trọng số, conda và cổng nằm ở **đầu file `addons/defaker2/CHAY_DEFAKER2.bat`**. Đổi ở đó nếu cài chỗ khác.

## 4. Bật demo (3 lệnh)

```bat
addons\defaker2\CHAY_DEFAKER2.bat
RUN_APP.bat --config research\configs\config_demo_defaker2.json
RUN_APP.bat --config research\configs\config_demo_defaker2_tiet_kiem_ram.json
```

1. Lệnh 1: chạy trong một cửa sổ riêng, chờ dòng `DA NAP XONG` (khoảng 1–1,5 phút).
2. Lệnh 2: cửa sổ thứ hai. Đây là bản demo đầy đủ: 3 bộ phát hiện cũ cộng DefakerOne2.
3. Lệnh 3 dùng **thay cho lệnh 2** khi RAM không đủ. Bản này tắt `face_clip`; trang ghi rõ không kiểm tra deepfake khuôn mặt và kết luận khác cấu hình chuẩn.

Thêm `--device cpu` vào lệnh 1 nếu không có GPU NVIDIA. Khi đó server chạy bf16 trên CPU, cần khoảng 4,5 GB RAM trống và rất chậm. Không dùng fp32, vì cần khoảng 9 GB.

**Nếu server báo lỗi `0xC0000005` (mã -1073741819) khi nạp:**
- Trên Windows, lỗi này thường do thiếu bộ nhớ ảo (RAM cộng pagefile). Server cần khoảng 7–8 GB bộ nhớ ảo, dù chỉ dùng khoảng 0,4–1,2 GB RAM thật, vì bộ nhớ GPU cũng được tính vào bộ nhớ ảo của Windows.
- Lần đầu gặp lỗi, Windows thường tự nới pagefile. `CHAY_DEFAKER2.bat` tự thử lại một lần sau 10 s.
- Trên máy đo, lần thử lại đã chạy được: pagefile tăng từ 7,6 lên 14,7 GB.
- Nếu vẫn lỗi: đóng bớt ứng dụng nặng, để trống ít nhất 10 GB trên ổ C:, hoặc đặt pagefile lớn hơn.

## 5. Chế độ chạy và độ khớp số

| Kiểm tra | Kết quả |
|---|---|
| Thiết bị / chế độ | GPU GTX 1650 (sm_75, không có bf16 gốc), fp16 + nén 8-bit LLM.int8. Phần thị giác và `lm_head` giữ fp16. |
| VRAM | 2 915 MiB sau khi nạp. Đỉnh khoảng 3,1 GB với ảnh nhỏ. Đỉnh 3 929 MiB / 4 096 MiB khi chạy ảnh 12 MP. |
| NaN / inf | Không có, trên 3 ảnh thử (42 logprob mỗi ảnh) và mọi ảnh đã chấm. Server kiểm tra mỗi lần chấm. |
| Server so với `swift infer` (3 ảnh thử) | p_fake **khớp tuyệt đối**: 0,039639 / 0,004070 / 0,942506. Khớp cả khi gửi bytes gốc lẫn PNG như app gửi. |
| So với bf16 (độ chính xác gốc của tác giả) | Xem mục 5.1 |

### 5.1. So sánh fp16 + 8-bit (GPU) với bf16 (CPU)

**Chưa chạy.** Máy 8 GB không đủ RAM trống để chạy bf16 trên CPU song song với các ứng dụng khác. Vì vậy chưa có số đo độ lệch p_fake do nén 8-bit. Muốn đo, đóng các ứng dụng khác rồi chạy `CHAY_DEFAKER2.bat --device cpu` và chấm lại 3 ảnh thử.

## 5.2. DefakerOne2 làm MÔ HÌNH CHÍNH (bỏ 3 bộ cũ)

```bat
addons\defaker2\CHAY_DEFAKER2.bat
RUN_APP_DEFAKER2.bat
```

`RUN_APP_DEFAKER2.bat` gọi `RUN_APP.bat --config research\configs\config_defaker2_chinh.json`. Cấu hình này:
- Chỉ bật `defaker2` với `role: "primary"`. Mesorch, Community Forensics và face_clip đều tắt.
- **Ngưỡng theo repo tác giả.** Tác giả không công bố ngưỡng số; kết luận là token trả lời `Real`/`Fake` (README: "Answer only Real or Fake"; accuracy tính trên token này). Theo `logprob2score.py`, "Fake" ⇔ p_fake ≥ 0,5.
  - p_fake ≥ 0,5: "Nghi vấn".
  - p_fake < 0,5: "Chưa thấy dấu hiệu rõ".
  - Không có mức "Cần kiểm tra thêm".
- Server tắt hoặc quá thời gian: kết luận "Chưa khả dụng" (không đoán).
- Hash cấu hình quyết định khác bản đã đánh giá, nên trang có cảnh báo. Ngưỡng 0,5 chưa kiểm chứng trên dữ liệu của nhóm.
- `config.json` và `RUN_APP.bat` giữ nguyên bản 3 bộ đã đánh giá.

Code dùng chung cho cả hai chế độ:
- Trong một cấu hình có thể có một detector `role: "primary"`.
- Nếu cấu hình còn các bộ khác và mô hình chính không chấm được, kết luận tạm theo các bộ đó và có cảnh báo rõ.
- Bộ kiểm định lấy kết luận cấp ứng dụng theo mô hình chính. Ảnh mà mô hình chính chưa chấm được thì không tính vào chỉ số, và được chấm lại ở lần sau.

## 5.3. DefakerOne2 + Mesorch (bản đồ vùng nghi sửa, tham khảo)

```bat
addons\defaker2\CHAY_DEFAKER2.bat
RUN_APP_DEFAKER2_MESORCH.bat
```

1. Lệnh 1: cửa sổ riêng, chờ dòng `DA NAP XONG`.
2. Lệnh 2: cửa sổ thứ hai. `RUN_APP_DEFAKER2_MESORCH.bat` gọi `RUN_APP.bat --config research\configs\config_defaker2_mesorch.json`.

Cấu hình `research/configs/config_defaker2_mesorch.json`:
- DefakerOne2 `role: "primary"`: **kết luận chỉ theo DefakerOne2**.
- Mesorch bật thêm để hiện bản đồ vùng nghi sửa, **chỉ để tham khảo**. Ngưỡng Mesorch chép nguyên từ `config.json` đã khóa (low 0,960912 / high 0,997501 / pixel 0,5); test `tests/test_defaker2_mesorch.py` kiểm tra hai bên bằng nhau.
- Community Forensics, face_clip và PSCC-Net tắt (tiết kiệm RAM).
- `config.json`, `RUN_APP.bat`, `config_defaker2_chinh.json` và `RUN_APP_DEFAKER2.bat` không đổi.

Trang kết quả, từ trên xuống:
1. **Thẻ kết luận**: kết luận theo DefakerOne2. Nếu có, thêm dòng "Độ tin cậy thấp: ảnh nhỏ hoặc đã qua nén lại" (mục 5.5) và dòng "Hai mô hình không thống nhất, cần xem kỹ".
2. **Thẻ DefakerOne2** (MÔ HÌNH CHÍNH) đứng đầu: p_fake, ngưỡng đang dùng, nhãn "CHƯA HIỆU CHỈNH" khi ngưỡng chưa được chọn bằng script (mục 5.4).
3. **Ngay dưới: bản đồ nhiệt Mesorch**, chú thích "Vùng nghi sửa (Mesorch), tham khảo".
4. Thẻ Mesorch (nhãn THAM KHẢO: điểm và mức theo ngưỡng của chính Mesorch), cảnh báo, tải kết quả.

**Không gộp điểm hai mô hình.** "Không thống nhất" nghĩa là một mô hình "Nghi vấn", mô hình kia "Chưa thấy dấu hiệu rõ", mỗi mô hình theo ngưỡng của chính nó.
- "Cần kiểm tra thêm" (Mesorch), "Không kết luận" (DefakerOne2) hay "Chưa khả dụng" không tính là trái nhau.
- Khi không thống nhất, kết luận vẫn theo DefakerOne2; dòng này chỉ nhắc người kiểm tra xem kỹ.
- Mesorch chỉ tìm chỉnh sửa cục bộ. Với ảnh sinh hoàn toàn bằng AI, DefakerOne2 có thể "Nghi vấn" trong khi Mesorch "Chưa thấy dấu hiệu rõ": dòng "không thống nhất" hiện là bình thường.
- Báo cáo JSON: `summary.disagreement` (`text_vi`, `detectors`, `detail_vi`). Bộ kiểm định: cột `mo_hinh_khong_thong_nhat` trong `du_doan.csv` và mục "Mô hình không thống nhất" (tỷ lệ trên ảnh thật / ảnh sửa).

**RAM và thời gian của cấu hình này: chưa đo.** `tools\DO_LAI_DEFAKER2.bat` (mục 10) ghi RAM đỉnh của app, của server, của cả máy và thời gian mỗi ảnh vào `tong_hop.md`, và báo khi RAM đỉnh vượt 6,5 GB hoặc một ảnh mất hơn 60 s.

## 5.4. Ngưỡng DefakerOne2 đọc từ config ("Không kết luận")

- App đọc `low` / `high` trong `thresholds` của detector `defaker2` (role primary):
  - p_fake ≥ high: "Nghi vấn".
  - low ≤ p_fake < high: **"Không kết luận"** (mã `inconclusive`, màu vàng).
  - p_fake < low: "Chưa thấy dấu hiệu rõ".
- Ngưỡng chỉ được coi là đã hiệu chỉnh khi `thresholds` có khối `calibration` (do `tools\chon_nguong_defaker2.py --ghi` ghi: ngày, mã thí nghiệm, file đề xuất và SHA-256). Không có khối này, app ghi **"Chưa hiệu chỉnh"**.
- Hiện nay cả `config_defaker2_chinh.json` và `config_defaker2_mesorch.json` đều là low = high = 0,5 (cách tác giả ra kết luận), nên **chưa có mức "Không kết luận"** và app ghi "Chưa hiệu chỉnh: ngưỡng 0,5 theo tác giả".
- Khối `calibration` và chữ `status` không vào hash cấu hình quyết định; low/high thì có.
- Bộ kiểm định tính "Không kết luận" như mức giữa: bảng "“Không kết luận” trở lên".

## 5.5. Cảnh báo "Độ tin cậy thấp"

`ifd/input_quality.py` dùng thông tin `read_image` đã có (kích thước hiển thị, định dạng, EXIF, kênh alpha) và tên file. Cảnh báo bật khi có **ít nhất một** điều kiện:

| Điều kiện | Lý do hiện trên trang (ví dụ) |
|---|---|
| Ảnh nhỏ hơn 0,5 MP | "0,34 MP, nhỏ hơn 0,5 MP" |
| PNG có kênh alpha nhưng không có pixel trong suốt nào | "PNG có kênh alpha nhưng không có pixel trong suốt nào (dấu hiệu ảnh chụp màn hình)" |
| Tên file kiểu "Screenshot…", "Ảnh chụp màn hình…", "Ảnh màn hình…" (không phân biệt hoa/thường, có dấu hay không) | "tên file “Screenshot 2026-10-18 181055.png” giống ảnh chụp màn hình" |
| JPEG không có EXIF và nhỏ hơn 2 MP | "JPEG không có EXIF và 1,20 MP, nhỏ hơn 2 MP (thường là ảnh đã qua ứng dụng nhắn tin, bị nén lại)" |

- **Riêng "không có EXIF" thì không cảnh báo**: ảnh tải từ Zalo hay Facebook đều mất EXIF.
- "Có EXIF" = có ít nhất một trường EXIF ngoài các trường kỹ thuật mà bộ mã hóa nào cũng có thể ghi (hướng xoay, độ phân giải, vị trí YCbCr).
- Số MP làm tròn xuống (0,3456 MP hiện "0,34 MP").
- Hiện ở **thẻ kết luận**: dòng in đậm "Độ tin cậy thấp: ảnh nhỏ hoặc đã qua nén lại", dưới là "Lý do: …".
- **Báo cáo JSON**: trường `input_quality` (`low_confidence`, `headline_vi`, `text_vi`, `reasons[].code` / `text_vi`, `megapixels`, `rule_version`). Phần `image` có thêm `alpha` và `exif_tags`.
- **Không đổi điểm, ngưỡng hay kết luận** (test `test_warning_never_changes_scores_or_decisions`: cùng điểm ảnh, đổi tên thành "Screenshot…", điểm và kết luận giữ nguyên).
- Áp dụng cho mọi cấu hình, kể cả `config.json`: chỉ thêm một trường báo cáo và một dòng hiển thị.
- Bộ kiểm định (`ifd.company_eval`):
  - `du_doan.csv` có thêm cột `input_quality` (`canh_bao` / `khong_canh_bao`), `input_quality_ly_do` và `seconds`.
  - Báo cáo có bảng "Tách theo cảnh báo “Độ tin cậy thấp”": số ảnh, đúng, sai, tỷ lệ đúng, tỷ lệ báo nhầm, tỷ lệ phát hiện cho nhóm có cảnh báo, không cảnh báo, và từng lý do.
  - Điểm cũ trong lưu đệm (chấm trước khi có cảnh báo): cảnh báo được tính lại bằng cách đọc ảnh, không chấm lại mô hình.
- Ngưỡng 0,5 MP và 2 MP là quy tắc kinh nghiệm theo yêu cầu, **chưa được chọn hay kiểm chứng trên dữ liệu**.

## 6. Số đo trên máy này

Điều kiện đo:
- Các ứng dụng của người dùng vẫn mở: Claude (khoảng 2,4 GB), Zalo (khoảng 1 GB), Edge, SQL Server, phần mềm diệt virus.
- RAM khả dụng trước khi đo: 1,6–1,9 GB.
- Ảnh thử tổng hợp, sinh bằng cùng hàm với mục `cpu_timing` của self-test, JPEG chất lượng 92.

| Chỉ số | Cấu hình demo đầy đủ | Cấu hình demo tiết kiệm RAM (tắt face_clip) |
|---|---|---|
| Ảnh 2 MP (1920×1080), gửi ảnh đến lúc nhận kết quả | 33,3–34,3 s (3 lần) | 33,1–33,7 s (2 lần) |
| Trong đó DefakerOne2 | 29,7–30,3 s. Ảnh đưa vào mô hình: 1920×1088 | 29,8–30,0 s |
| Trong đó 3 bộ cũ | 2,6–3,2 s | 2,4–2,8 s (2 bộ) |
| Ảnh 12 MP (4000×3000), gửi ảnh đến lúc nhận kết quả | 50,6–51,1 s | 50,7–53,3 s |
| Trong đó DefakerOne2 | 41,4–42,0 s. Ảnh đưa vào mô hình: 1792×1344 (thu nhỏ theo MAX_PIXELS) | 41,2–41,8 s |
| RAM thật đỉnh của app (working set) | 2,04 GiB (2,43 GiB lúc nạp mô hình) | 1,30 GiB (1,54 GiB lúc nạp) |
| Bộ nhớ ảo đỉnh của app (private) | 4,14 GiB | 2,79 GiB |
| RAM thật của server khi chạy (working set) | ≤ 0,70 GiB (3,78 GiB lúc nạp trọng số) | ≤ 0,38 GiB |
| Bộ nhớ ảo đỉnh của server (private) | 7,73 GiB | 7,73 GiB |
| RAM hệ thống đã dùng, đỉnh | 7,32 / 7,42 GiB | 7,12 / 7,42 GiB |
| RAM khả dụng thấp nhất | **0,1 GiB** | **0,3 GiB** |
| GPU (nvidia-smi) | 3 929 MiB | 3 929 MiB |

So sánh với app khi chạy một mình (self-test `cpu_timing` trên cùng máy, sau thay đổi):
- Ảnh 2 MP: phân tích p50 3,58 s, gửi đến nhận p50 4,22 s.
- Ảnh 12 MP: phân tích p50 4,00 s, gửi đến nhận p50 8,33 s.
- RAM đỉnh 2,95 GB.

DefakerOne2 là phần chậm nhất. Thời gian của nó tăng theo số token ảnh, tức theo kích thước ảnh sau khi thu nhỏ về tối đa khoảng 2,4 MP:

| Ảnh đưa vào mô hình | Thời gian DefakerOne2 (server đã khởi động) |
| --- | --- |
| 576×480 | 4 s |
| 864×864 | 9 s |
| 1184×1440 | 23 s |
| 1920×1088 | 30 s |
| 1792×1344 | 41 s |

**Đánh giá RAM.** Vượt mức an toàn của máy 8 GB.
- Khi app (cấu hình đầy đủ) và server chạy cùng lúc, RAM khả dụng có lúc chỉ còn 0,1 GiB và Windows phải dùng pagefile.
- Tắt `face_clip` chỉ cải thiện được một phần: RAM khả dụng thấp nhất 0,3 GiB. App giảm 0,74 GiB RAM thật và 1,35 GiB bộ nhớ ảo.

**Đề xuất giảm RAM, theo thứ tự nên làm:**
1. Đóng ứng dụng nặng trước khi demo: Zalo, trình duyệt khác, các cửa sổ Claude không dùng. Tạm dừng dịch vụ SQL Server nếu không cần. Bộ nhớ riêng các ứng dụng này chiếm lúc đo: Claude 2,4 GB, Zalo 1,0 GB, SQL Server 0,4 GB, Edge và WebView2 0,4 GB.
2. Dùng `config_demo_defaker2_tiet_kiem_ram.json` (tắt `face_clip`). Trang ghi rõ không kiểm tra deepfake khuôn mặt.
3. Không chạy `PHAN_TICH_THU_MUC.bat` hoặc `KIEM_DINH_CONG_TY.bat` cùng lúc với `RUN_APP.bat`, vì mỗi cái nạp một bộ mô hình riêng.
4. Để Windows tự quản lý pagefile và giữ ít nhất 10 GB trống trên ổ C:, vì server cần khoảng 7,7 GB bộ nhớ ảo.
5. Không khuyến nghị:
   - Giảm `MAX_PIXELS` để tăng tốc: điểm sẽ khác pipeline của tác giả.
   - Chạy server bằng CPU song song với app: cần thêm khoảng 4,5 GB RAM.

## 7. Giới hạn

- **Không có bản đồ vùng.** DefakerOne2 chỉ cho điểm toàn ảnh. Thẻ thử nghiệm không có tab bản đồ nhiệt. Bản DefakerOne2 + Mesorch (mục 5.3) hiện bản đồ của **Mesorch**, chỉ để tham khảo: bản đồ đó không giải thích điểm của DefakerOne2.
- **Báo nhầm trên ảnh thật nhỏ / nén lại.** Đã thấy ảnh selfie thật khoảng 0,34 MP bị chấm p_fake 0,97. Ảnh dưới 0,26 MP bị phóng to trước khi vào mô hình. Cảnh báo "Độ tin cậy thấp" (mục 5.5) chỉ nhắc, không sửa được điểm.
- **Giấy phép trọng số chưa rõ.** README chỉ ghi mã nguồn theo Apache-2.0, còn giấy phép trọng số dẫn sang trang ModelScope mà chưa công bố rõ. Vì vậy **không phát tán kèm trọng số**: gói app và repo không chứa trọng số, người dùng tự tải.
- **Số liệu tác giả chưa được kiểm chứng độc lập.** Ở đây chạy fp16 + nén 8-bit, nên số liệu công bố có thể không còn đúng. p_fake là điểm thô, chưa hiệu chỉnh, chưa có ngưỡng kiểm chứng. Mốc 0,5 chỉ là câu trả lời Real/Fake của mô hình.
- Câu hỏi của mô hình cố định bằng tiếng Anh. Ảnh nhỏ hơn 0,26 MP bị phóng to, ảnh lớn hơn 2,4 MP bị thu nhỏ trước khi vào mô hình.
- Ảnh có nền trong suốt: app ghép lên nền trắng trước khi gửi. Lệnh `swift infer` trên file gốc thì bỏ kênh alpha. Hai cách cho kết quả giống nhau khi ảnh không có vùng trong suốt.
- Server xử lý từng ảnh một. Khoảng 30–42 s mỗi ảnh lớn trên GTX 1650. Thư viện tăng tốc linear-attention (`flash-linear-attention`, `causal-conv1d`) chưa cài trên Windows.
- Bản demo trong `D:\IFD_demo_defaker2` là gói FINAL giải nén rồi chép thêm các file thay đổi. **Bản này chưa đóng gói lại**:
  - `PACKAGE_MANIFEST.sha256` cũ không khớp các file đã sửa.
  - Self-test không kiểm tra manifest này, chỉ kiểm tra SHA-256 các mô hình ONNX, nên vẫn chạy được.
  - File ZIP gốc giữ nguyên, SHA-256 `08bd9cfb…71b4`.

## 8. Kiểm thử

Đã chạy trên MAY_A (Windows 11, runtime Python 3.13 của gói):
- **Toàn bộ pytest trước khi sửa bộ kiểm định: 72/72 đạt.** Gồm 50 test cũ (không sửa test cũ) và 22 test mới.
- Sau khi sửa bộ kiểm định: `test_defaker2.py` và `test_company_eval.py` đạt 26/26. Thêm test cho HTTP của server: đạt.
- Sau khi thêm chế độ mô hình chính: 27 test đạt, 1 test hỏng (`test_primary_model_down_gives_no_conclusion`). Test này hỏng vì câu kiểm tra chọn sai vị trí cảnh báo. Câu kiểm tra đã sửa nhưng **chưa chạy lại**.
- **Self-test đầy đủ trên bản demo** (trước khi sửa bộ kiểm định): 12/12 PASS. Bản che tên nằm trong `reports/evidence/defaker2_addon/`.
- **Lượt kiểm định trên bộ ảnh nhóm** đã dừng giữa chừng ở 64/180 ảnh, nên chưa có số liệu độ chính xác.

Test mới (`tests/test_defaker2.py`) dùng server giả lập, gồm các trường hợp:
- Chạy được: điểm thử nghiệm, PNG gửi đi không mất dữ liệu.
- Server tắt.
- Quá thời gian: trả kết quả trong khoảng 1 s cộng thời gian chờ.
- 503, 500, JSON hỏng, NaN, điểm ngoài [0, 1], thiếu trường.
- `/health`.
- Cấu hình: chỉ nhận URL 127.0.0.1/localhost có cổng, `timeout_seconds` phải hợp lệ.
- Cấu hình demo giữ hồ sơ "chuẩn".
- Kết luận, bài toán và cảnh báo chung không đổi dù server bật hay tắt.
- API web hiện thẻ thứ tư.
- Bộ kiểm định báo DefakerOne2 riêng và chấm bù sau khi server bị gián đoạn.

### 8.1. Kiểm thử lần cập nhật 08/11/2026

Chạy trên MAY_A bằng runtime Python 3.13 của gói; DefakerOne2 dùng server **giả lập**, không chấm ảnh thật nào. Bằng chứng: `reports/evidence/defaker2_mesorch/`.
- **Toàn bộ pytest: 129/129 đạt** (84 test cũ, không sửa test cũ nào; 45 test mới), trong môi trường như các file `.bat` (`PYTHONUTF8=1`).
  - Lần chạy đầu thiếu `PYTHONUTF8=1`: 8 test chạy tiến trình con (`test_cli.py`, `test_company_eval.py`) hỏng vì tiến trình con in tiếng Việt bằng bảng mã của Windows. Code gốc d5b700a hỏng y như vậy (kiểm bằng `git stash`), nên đây là lỗi môi trường chạy test. File: `pytest_lan1_thieu_utf8_20261108.txt`.
  - Test cũ `test_primary_with_other_detectors_falls_back_when_down` hỏng sẵn ở code gốc (cảnh báo "kết luận TẠM" bị đẩy xuống sau cảnh báo "cấu hình khác"); đã sửa code (`ifd/analysis.py`), không sửa test.
- **Self-test (`python -m ifd.selftest`): PASS 12, FAIL 0, SKIP 0.** File: `selftest_MAY_A_20261108.txt`.
- Test mới:
  - `tests/test_input_quality.py`: từng điều kiện của cảnh báo "Độ tin cậy thấp" (ảnh tạo trong test), tên file thường không bị nhận nhầm, cảnh báo không đổi điểm và kết luận, cột và bảng của bộ kiểm định, tính lại cảnh báo cho điểm cũ trong lưu đệm mà không nạp mô hình.
  - `tests/test_defaker2_mesorch.py`: cấu hình mới (ngưỡng Mesorch bằng `config.json`), "Không kết luận" giữa low và high, "Chưa hiệu chỉnh" khi chưa có khối `calibration`, hash quyết định, quy tắc "không thống nhất" (không gộp điểm), chạy thật DefakerOne2 giả lập + Mesorch ONNX, bố cục trang, bộ kiểm định.
  - `tests/test_tools_defaker2.py`: tạo nhãn trên ZIP chia phần giả lập (kiểm SHA-256, TIFF -> PNG không mất điểm ảnh, dừng khi trùng tập đo, `--kiem-tra`), chọn ảnh `--thu` / `--mau`, đo lại đầu-cuối với server giả lập (tổng hợp không có tên ảnh), dừng khi server tắt, quy tắc ngưỡng theo từng nguồn, `--ghi` (SHA-256 file đề xuất khớp, giữ CRLF, Mesorch không đổi), từ chối dữ liệu test và `config.json`.
- Bố cục trang: chạy `app.js` thật trên DOM giả với JSON thật của server web (`kiem_tra_giao_dien/ket_qua.txt`), không lỗi JavaScript. **Chưa kiểm bằng trình duyệt thật** (máy không có Playwright): nên mở `RUN_APP_DEFAKER2_MESORCH.bat` xem trước khi demo.

## 9. File thay đổi

- Mới:
  - `addons/defaker2/defaker2_server.py`, `addons/defaker2/CHAY_DEFAKER2.bat`
  - `ifd/detectors/defaker2.py`
  - `research/configs/config_demo_defaker2.json`, `research/configs/config_demo_defaker2_tiet_kiem_ram.json`
  - `tests/test_defaker2.py`
  - tài liệu này
  - `reports/evidence/defaker2_addon/`
- Sửa:
  - `ifd/detectors/base.py`: quyết định `experimental`, tên bài toán thử nghiệm, cờ `in_summary`.
  - `ifd/config.py`: kiểm tra `defaker2_http` và `display.notice_vi`.
  - `ifd/analysis.py`: nạp detector HTTP, loại detector thử nghiệm khỏi kết luận và hash quyết định.
  - `ifd/webapp/server.py`, `ifd/webapp/static/app.js`, `ifd/webapp/static/style.css`: thẻ thứ tư, trạng thái server, dòng thông báo.
  - `ifd/company_eval.py`: mục báo cáo riêng cho bộ thử nghiệm.
- Không đổi: `config.json`, ngưỡng, mô hình ONNX, `requirements.txt`.

Cập nhật 08/11/2026:
- Mới:
  - `research/configs/config_defaker2_mesorch.json`, `RUN_APP_DEFAKER2_MESORCH.bat` (mục 5.3).
  - `ifd/input_quality.py` (mục 5.5).
  - `tools/tao_nhan_cong_khai.py`, `tools/do_lai_defaker2.py`, `tools/DO_LAI_DEFAKER2.bat`, `tools/chon_nguong_defaker2.py`, `tools/CHON_NGUONG_DEFAKER2.bat` (mục 10).
  - `tests/test_input_quality.py`, `tests/test_defaker2_mesorch.py`, `tests/test_tools_defaker2.py`, `tests/fake_defaker2.py` (server giả lập dùng chung).
  - `reports/evidence/defaker2_mesorch/` (kiểm tra danh mục dữ liệu công khai, kết quả pytest và self-test).
- Sửa:
  - `ifd/detectors/defaker2.py`: đọc low/high, mức "Không kết luận", trạng thái "chưa hiệu chỉnh" / "đã hiệu chỉnh".
  - `ifd/detectors/base.py`: thêm quyết định `inconclusive` ("Không kết luận").
  - `ifd/analysis.py`: trường `input_quality`, dòng "không thống nhất" (`summary.disagreement`), khối `calibration` không vào hash quyết định; cảnh báo "kết luận TẠM" khi DefakerOne2 tắt nay đứng trước cảnh báo "cấu hình khác bản đã đánh giá" (sửa lỗi làm test cũ `test_primary_with_other_detectors_falls_back_when_down` hỏng từ commit trước).
  - `ifd/image_io.py`: thêm `alpha` (có kênh alpha, giá trị nhỏ nhất, tỷ lệ pixel trong suốt) và `exif_tags` vào metadata; cách đọc ảnh không đổi.
  - `ifd/company_eval.py`: cột `input_quality`, `input_quality_ly_do`, `seconds`, `mo_hinh_khong_thong_nhat`; bảng theo cảnh báo; thời gian mỗi ảnh; RAM đỉnh của tiến trình; mức "Không kết luận".
  - `ifd/sysinfo.py`: đo RAM của tiến trình khác và của cả máy (cho script đo).
  - `ifd/webapp/server.py`, `ifd/webapp/static/*`: bố cục bản DefakerOne2 + Mesorch, dòng cảnh báo trong thẻ kết luận, nhãn CHƯA HIỆU CHỈNH / THAM KHẢO.
  - `addons/defaker2/defaker2_server.py`: `/health` có thêm `memory` (RAM tiến trình, VRAM đỉnh); cách chấm không đổi.
  - `EXPERIMENT_LOG.jsonl`: thêm `P14_defaker2_quy_trinh_hieu_chinh_va_do_lai` (khóa cách chọn ngưỡng và cách đo trước khi chấm).
  - `.gitignore`: chặn `reports/defaker2/**/nhat_ky.log` và `reports/defaker2/**/rieng_tu/`.
- Không đổi: `config.json`, `config_defaker2_chinh.json`, ngưỡng và mô hình của các detector cũ, `requirements.txt`, runtime.

## 10. Đo lại và chọn ngưỡng (người dùng tự chạy)

Bước này **chỉ có code và script**; chưa chấm ảnh nào. Mọi lệnh chạy trong thư mục ứng dụng (nơi có `RUN_APP.bat`), sau khi đã bật `addons\defaker2\CHAY_DEFAKER2.bat` và thấy dòng `DA NAP XONG`. Server chưa chạy thì script báo "Hãy chạy addons\defaker2\CHAY_DEFAKER2.bat trước" rồi thoát (mã 3).

### 10.1. Tách tập hiệu chỉnh và tập đo

Ngưỡng **chỉ** chọn trên tập hiệu chỉnh; tập đo không bao giờ dùng để chọn ngưỡng. Dùng nguyên cách tách dev/test **đã khóa trước khi chấm bất kỳ ảnh test nào** của repo (`reports/manifests/*.jsonl`; EXPERIMENT_LOG P02, P03; `research/prepare_korus.py`), không tách lại:

| Tập | Dùng để | Nguồn | Số ảnh (theo manifest) |
|---|---|---|---|
| Hiệu chỉnh | chọn low/high | Korus `dev_korus` | 72 ảnh thật + 72 ảnh sửa (72 cặp cảnh) |
| Hiệu chỉnh | chọn low/high | DeepFakeFace dev | 300 ảnh thật (wiki) + 900 ảnh giả (insight, inpainting, text2img, mỗi loại 300) |
| Hiệu chỉnh | chọn low/high | OpenFake dev (cần pyarrow) | 455 ảnh thật (299 ImageNet, 156 DOCCI) + 114 ảnh AI |
| Đo | báo cáo | A, B, C, D | theo 4 file nhãn trong `data\khac_phuc` |
| Đo | báo cáo | CocoGlide | 512 ảnh thật + 512 ảnh sửa (toàn bộ là test) |
| Đo | báo cáo | Korus `test_korus` | 148 ảnh thật + 148 ảnh sửa |

- Korus tách theo **cặp cảnh** (ảnh gốc và ảnh sửa cùng `group_id` luôn ở cùng phía), phân tầng theo máy ảnh, **seed 20261012**, 18 cặp mỗi máy vào dev. Bước đo chỉ dùng `test_korus`.
- CocoGlide toàn bộ là test: không dùng để hiệu chỉnh.
- DeepFakeFace và OpenFake: chỉ phần dev; phần test của hai bộ này không dùng ở đâu.
- `ctrl_mei` dev (phần dev mà Mesorch đã dùng) không có ảnh trên máy (phải tải ảnh nguồn từ GitHub rồi tạo lại) nên không dùng.
- OpenFake chỉ lấy được bằng Python có `pyarrow` (runtime của app không có). Các file `.bat` tự dùng `venv_research\Scripts\python.exe` cho bước này nếu có; nếu không, OpenFake bị bỏ qua và `tach_tap.json` ghi rõ.
- Kiểm tra tách biệt:
  - `tools/tao_nhan_cong_khai.py` kiểm SHA-256 từng ảnh với manifest (Korus: SHA-256 file TIFF gốc và SHA-256 điểm ảnh sau khi đổi sang PNG không mất dữ liệu), và **dừng** nếu một ảnh hiệu chỉnh trùng tập đo theo mã ảnh, nhóm, SHA-256 file hoặc SHA-256 điểm ảnh.
  - `tools/chon_nguong_defaker2.py` **dừng** nếu `du_doan.csv` có ảnh ngoài tập hiệu chỉnh, ảnh của CocoGlide / `test_korus`, đường dẫn kiểu data.zip (`that/`, `sua/`, `doi_chung/`), hoặc ảnh trùng byte với ảnh trong thư mục data; dừng cả khi file nhãn hiệu chỉnh khác SHA-256 ghi trong `tach_tap.json`.
- Đã kiểm (chỉ đọc danh mục ZIP và cột `row_id` của OpenFake, không giải nén): đủ CocoGlide 1024/1024, Korus 440/440 (tổng 1 697 070 982 byte, đúng bằng file gốc), DeepFakeFace dev 1200/1200, OpenFake dev 569/569. Bằng chứng: `reports/evidence/defaker2_mesorch/kiem_tra_du_lieu_cong_khai.txt`. Kiểm lại: `python tools/tao_nhan_cong_khai.py --kiem-tra`.
- Ảnh giải nén vào `D:\ASUS\2\Image_Forgery_Detection_CPU\du_lieu_cong_khai` (ngoài repo; khoảng 2–3 GB, chủ yếu là PNG của Korus). Không đưa thư mục này lên git.

### 10.2. Quy tắc chọn ngưỡng

Giống các detector khác (`QUALITY_TARGETS.yaml` mục `operating_point_policy`, `research/derive_thresholds.py`):
- **high** = ngưỡng nhỏ nhất để tỷ lệ báo nhầm (FPR) ≤ 5 % trên **từng** nguồn ảnh thật của tập hiệu chỉnh (Korus, DeepFakeFace wiki, OpenFake ImageNet, OpenFake DOCCI).
- **low** = ngưỡng nhỏ nhất để FPR ≤ 20 % trên từng nguồn; low ≤ high.
- Giữa low và high: app hiện "Không kết luận".
- Ngưỡng làm tròn lên 9 chữ số thập phân. Đề xuất có FPR từng nguồn kèm khoảng tin cậy Wilson 95 %, tỷ lệ phát hiện trên ảnh giả của tập hiệu chỉnh và, để so sánh, kết quả tại mốc 0,5 cũ và phương án "gộp mọi ảnh thật" (chỉ tham khảo, không dùng).
- Script chỉ in và lưu đề xuất vào `reports\defaker2\nguong_de_xuat.json` (+ `.md`). Chỉ khi chạy với `--ghi` mới ghi low/high (kèm khối `calibration`) vào `research/configs/config_defaker2_mesorch.json` và thêm một dòng `D2Txx_…` vào `EXPERIMENT_LOG.jsonl`. **`config.json` đã khóa không bao giờ bị ghi** (script từ chối).
- Muốn ghi cả bản chỉ có DefakerOne2: thêm `--config-ghi research\configs\config_defaker2_chinh.json`. Khi đó test cũ `test_main_config_file_uses_only_defaker2_with_author_threshold` (kiểm tra ngưỡng 0,5 của bản này) phải được cập nhật trong cùng PR ngưỡng.

### 10.3. Lệnh theo thứ tự

Thời gian ước tính theo số đo GTX 1650 ở mục 6 (DefakerOne2 khoảng 4 s với ảnh nhỏ, 30 s với ảnh 2 MP, 41 s với ảnh 12 MP; Mesorch thêm khoảng 1,5 s), tính từ kích thước ảnh ghi trong manifest, cho lần chạy đầu (chưa có điểm lưu đệm). Máy khác sẽ khác.

| # | Lệnh | Việc | Ước tính |
|---|---|---|---|
| 0 | `addons\defaker2\CHAY_DEFAKER2.bat` | bật server, chờ `DA NAP XONG` | 1–1,5 phút |
| 1 | `RUN_APP_DEFAKER2_MESORCH.bat` | demo bản mới (cửa sổ thứ hai) | khoảng 5–45 s mỗi ảnh tùy cỡ |
| 2 | `tools\DO_LAI_DEFAKER2.bat A --thu 10` | chạy thử 10 ảnh đầu (5 thật + 5 sửa); lần đầu tự giải nén data.zip vào `Downloads\data` và chép `khac_phuc` (bộ B–D tự tạo `doi_chung` nếu chưa có) | 3–8 phút |
| 3 | `tools\DO_LAI_DEFAKER2.bat A` | bộ A: 112 ảnh (92 thật + 20 xóa vật thể) | 30–75 phút (tùy cỡ ảnh Zalo) |
| 4 | `tools\DO_LAI_DEFAKER2.bat B`, rồi `C`, rồi `D` | 40 / 39 / 40 ảnh khoảng 0,1–0,25 MP | khoảng 5 phút mỗi bộ |
| 5 | `tools\DO_LAI_DEFAKER2.bat cocoglide` | 1024 ảnh 256×256; lần đầu tự giải nén | khoảng 100 phút |
| 6 | `tools\DO_LAI_DEFAKER2.bat korus` | 296 ảnh 1920×1080 (chỉ phần test); lần đầu tự đổi TIFF sang PNG | khoảng 2,6 giờ (+5–10 phút đổi ảnh); hoặc `korus --mau 100` khoảng 55 phút |
| 7 | `tools\CHON_NGUONG_DEFAKER2.bat --chi-anh-that` | chấm ảnh thật của tập hiệu chỉnh rồi đề xuất ngưỡng (không ghi) | khoảng 3,2 giờ có OpenFake (827 ảnh thật); khoảng 1 giờ nếu không có OpenFake (372 ảnh) |
| 7b | `tools\CHON_NGUONG_DEFAKER2.bat` | như 7, chấm cả ảnh giả để có tỷ lệ phát hiện trên tập hiệu chỉnh | khoảng 5,4 giờ (1913 ảnh); 2,7 giờ nếu không có OpenFake |
| 8 | `tools\CHON_NGUONG_DEFAKER2.bat --du-doan reports\defaker2\hieu_chinh_<ngày giờ>\chi_tiet\du_doan.csv --ghi` | ghi ngưỡng đã đề xuất (không chấm lại) | dưới 1 phút |
| 9 | `tools\DO_LAI_DEFAKER2.bat tat_ca` | đo lại một lần A–D, CocoGlide, Korus với ngưỡng mới | vài phút: điểm lấy từ lưu đệm, chỉ tính lại kết luận |

- `--thu N`: chạy thử N ảnh đầu (nửa thật, nửa sửa). `--mau N --seed S`: mẫu ngẫu nhiên khoảng N ảnh, chọn cả nhóm (cặp cảnh), seed mặc định 20261108. Cả hai luôn được ghi trong `tong_hop.md`.
- `--chi-defaker2`: dùng `config_defaker2_chinh.json` (không bật Mesorch). Mặc định là bản DefakerOne2 + Mesorch, để đo cả RAM khi hai mô hình chạy cùng lúc.
- Mỗi lần đo ghi vào thư mục mới `reports\defaker2\<bộ>_<ngày giờ>\`, không ghi đè lần trước; `tong_hop.md` liệt kê các lần đo trước của cùng bộ.
  - `tong_hop.md`, `tong_hop.json`: AUROC kèm khoảng tin cậy 95 % (bootstrap theo nhóm, 1000 lần), tỷ lệ báo nhầm và tỷ lệ phát hiện tại high và low, luật nền (`luat_nen.py`: "ảnh càng nhỏ càng nghi sửa", "là PNG thì là sửa"), Mesorch (tham khảo), tỷ lệ không thống nhất, bảng theo cảnh báo "Độ tin cậy thấp", thời gian mỗi ảnh, RAM đỉnh. Không có tên ảnh nên chia sẻ được.
  - `chi_tiet\` (CocoGlide, Korus): `du_doan.csv`, `metrics.json`, báo cáo của bộ kiểm định.
  - `rieng_tu\` (A–D): như trên nhưng có tên file chứa mã cuộc trò chuyện; **không đưa lên git** (`.gitignore` đã chặn), không gửi ra ngoài.
  - `nhat_ky.log`: toàn bộ thông báo; gửi file này khi có lỗi. Cũng bị `.gitignore` chặn vì có tên ảnh.
- Điểm từng ảnh lưu đệm trong `outputs\defaker2_kiem_dinh` (theo cấu hình, không theo ngưỡng): đổi ngưỡng rồi đo lại không phải chấm lại.

### 10.4. Kết quả (chưa đo — điền sau khi chạy mục 10.3)

Chỉ điền số có file bằng chứng (`reports\defaker2\...\tong_hop.json`). Không chọn lần đo đẹp nhất: ghi mọi lần đo.

**Bảng 1. Trước khi chỉnh ngưỡng (low = high = 0,5, theo tác giả)**

| Bộ | Ảnh thật / sửa | AUROC [CI 95 %] | Báo nhầm tại 0,5 | Phát hiện tại 0,5 | Luật "ảnh nhỏ" | Luật "là PNG" | Trung vị s/ảnh | RAM đỉnh app + server | File bằng chứng |
|---|---|---|---|---|---|---|---|---|---|
| A — xóa vật thể | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| B — ghép ảnh | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| C — ghép mặt | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| D — ảnh AI | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| CocoGlide | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| Korus (test) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

**Bảng 2. Đề xuất ngưỡng trên tập hiệu chỉnh**

| Nguồn ảnh thật | n | Báo nhầm tại high [CI 95 %] | Báo nhầm tại low | Báo nhầm tại 0,5 (cũ) |
|---|---|---|---|---|
| Korus dev | chưa đo | chưa đo | chưa đo | chưa đo |
| DeepFakeFace wiki | chưa đo | chưa đo | chưa đo | chưa đo |
| OpenFake ImageNet | chưa đo | chưa đo | chưa đo | chưa đo |
| OpenFake DOCCI | chưa đo | chưa đo | chưa đo | chưa đo |
| **Ngưỡng đề xuất** | low = chưa đo · high = chưa đo | | | |

**Bảng 3. Sau khi chỉnh ngưỡng (đo lại một lần)**

| Bộ | Báo nhầm tại high | Phát hiện tại high | "Không kết luận" trên ảnh thật / sửa | Báo nhầm tại low | Phát hiện tại low | File bằng chứng |
|---|---|---|---|---|---|---|
| A — xóa vật thể | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| B — ghép ảnh | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| C — ghép mặt | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| D — ảnh AI | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| CocoGlide | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| Korus (test) | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

**Bảng 4. Tách theo cảnh báo "Độ tin cậy thấp" (kết luận "Nghi vấn" là dương)**

| Bộ | Có cảnh báo: n · tỷ lệ đúng · báo nhầm | Không cảnh báo: n · tỷ lệ đúng · báo nhầm |
|---|---|---|
| A — xóa vật thể | chưa đo | chưa đo |
| B — ghép ảnh | chưa đo | chưa đo |
| C — ghép mặt | chưa đo | chưa đo |
| D — ảnh AI | chưa đo | chưa đo |
| CocoGlide | chưa đo | chưa đo |
| Korus (test) | chưa đo | chưa đo |

### 10.5. Giới hạn của phép đo

- **Số liệu tác giả DefakerOne2 chưa được bình duyệt độc lập.** Ở đây server chạy GPU fp16 kèm nén 8-bit, nên số liệu công bố có thể không còn đúng.
- **Cỡ mẫu nhỏ.** B, C, D chỉ khoảng 40 ảnh mỗi bộ, A có 20 ảnh sửa: khoảng tin cậy rất rộng; khác biệt nhỏ hơn độ rộng khoảng tin cậy không có ý nghĩa. Korus dev chỉ có 72 ảnh thật, nên ngưỡng FPR 5 % trên nguồn này chỉ cho phép 3 ảnh báo nhầm.
- **Ảnh đối chứng do máy tạo.** Ảnh `doi_chung` của B–D được cắt và thu nhỏ từ ảnh gốc bằng `tao_doi_chung.py`, không phải ảnh chụp màn hình thật, nên độ nét ở viền hơi khác. Bộ D chỉ khớp cỡ và định dạng, không khớp nội dung.
- **Tập hiệu chỉnh là dữ liệu công khai** (Korus, DeepFakeFace, OpenFake), không phải ảnh của công ty: tỷ lệ báo nhầm trên ảnh thật thực tế (Zalo, Facebook, ảnh chụp màn hình) có thể khác.
- Cảnh báo "Độ tin cậy thấp" và quy tắc "không thống nhất" là quy tắc mô tả, chưa kiểm chứng; Mesorch chỉ để tham khảo.
- Thời gian và RAM phụ thuộc máy và các ứng dụng đang mở lúc đo.
