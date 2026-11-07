# DEFAKER2_ADDON_VI — Gói phụ thử nghiệm DefakerOne2

_Lập ngày 07/11/2026. Số đo trên laptop AMD Ryzen 7 4800H, RAM 8 GB (Windows thấy 7,6 GB), GPU GTX 1650 4 GB, Windows 11. File bằng chứng nằm trong `reports/evidence/defaker2_addon/` (đã che tên máy, tên người dùng và đường dẫn)._

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

- **Không có bản đồ vùng.** DefakerOne2 chỉ cho điểm toàn ảnh. Thẻ thử nghiệm không có tab bản đồ nhiệt.
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
