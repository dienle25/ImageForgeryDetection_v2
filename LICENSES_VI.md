# LICENSES_VI — Giấy phép mã, trọng số, thư viện và dữ liệu

_Thông tin dưới đây là **dữ kiện về nguồn** để bộ phận pháp lý xem xét, **không phải ý kiến pháp lý**. Quyền dùng cho nghiên cứu không đồng nghĩa với quyền dùng thương mại trong công ty._

## 1. Tóm tắt cho người quyết định

| Thành phần | Mã nguồn | Trọng số mô hình | Dữ liệu huấn luyện của trọng số | Cần pháp lý xem trước khi dùng thương mại? |
|---|---|---|---|---|
| Ứng dụng `ifd/` (v2) | MIT (`LICENSE`) | — | — | Không (mã của dự án) |
| Mesorch — chỉnh sửa cục bộ | MIT (© 2024 K-Squad) | checkpoint chính thức `mesorch-98.pth`, **không có giấy phép riêng cho trọng số** | CASIA v2, Fantastic Reality, IMD2020, tampCOCO (điều khoản thiên về nghiên cứu) | **Có** |
| Community Forensics ViT-S/16@384 — ảnh sinh AI | MIT (© 2025 Jeongsoo Park) | MIT theo model card HF `OwensLab/commfor-model-384` | ảnh sinh từ hàng nghìn generator, mỗi nguồn có điều khoản riêng | **Có** |
| CLIP ViT-L/14 LN-tuning (yermandy) — deepfake khuôn mặt | MIT (© 2025 Andy / yermandy) | MIT theo repo/HF `yermandy/deepfake-detection`; nền CLIP của OpenAI (MIT) | FaceForensics++ (điều khoản chỉ cho nghiên cứu, cần đăng ký) | **Có** |
| YuNet — phát hiện khuôn mặt | MIT (© 2020 Shiqi Yu, OpenCV Zoo) | MIT | — | Thấp |
| PSCC-Net — baseline v1 (tắt mặc định) | MIT | không thấy giấy phép riêng | CASIA v2, … | Có (nếu bật lại) |

Bản sao văn bản giấy phép: `third_party_licenses/` (Mesorch, Community Forensics, yermandy, YuNet, PSCC-Net). Nguồn và SHA-256 từng file mô hình: `models/manifest.json`.

## 2. Thư viện chạy kèm (thư mục `runtime/` của gói Windows)

| Thư viện | Phiên bản | Giấy phép | Ghi chú |
|---|---|---|---|
| CPython (gói embeddable của python.org) | 3.13.16 | PSF License + giấy phép các thư viện đi kèm (OpenSSL Apache-2.0, libffi MIT, bzip2, SQLite public domain, …) | `runtime/LICENSE.txt` |
| numpy | 2.5.3 | BSD-3-Clause (+ OpenBLAS BSD, các thành phần trong `numpy-*.dist-info/licenses`) | |
| Pillow | 12.3.0 | MIT-CMU (HPND) | |
| onnxruntime | 1.29.0 | MIT (+ ThirdPartyNotices trong dist-info) | |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (+ LICENSE-3RD-PARTY) | đã **bỏ** `opencv_videoio_ffmpeg*.dll` (FFmpeg, chỉ dùng cho video – ứng dụng không dùng) |
| protobuf, flatbuffers, packaging | 7.36.2 / 25.12.19 / 26.3 | BSD-3 / Apache-2.0 / Apache-2.0 hoặc BSD | phụ thuộc của onnxruntime |

Văn bản giấy phép của từng thư viện nằm trong `runtime/Lib/site-packages/<tên>.dist-info/`.

**Thư viện C/C++ runtime của Microsoft:** gói chứa đúng các DLL mà bản phân phối chính thức của upstream đã kèm sẵn, không thêm/sửa: `runtime/vcruntime140.dll`, `runtime/vcruntime140_1.dll` (có trong gói CPython embeddable của python.org) và `runtime/Lib/site-packages/numpy.libs/msvcp140-*.dll` (numpy đóng kèm trong wheel). Các DLL này do các dự án đó phân phối lại theo điều khoản *Microsoft Distributable Code*; `runtime/LICENSE.txt` (mục “Additional Conditions for this Windows binary build”) nêu điều kiện khi phân phối tiếp: không sửa thông báo bản quyền của Microsoft, không dùng thương hiệu Microsoft trong tên sản phẩm, chỉ chạy trên nền tảng Microsoft, không dùng trong chương trình độc hại/lừa dối — bộ phận pháp lý nên xác nhận khi công ty phân phối lại gói. Gói **không** kèm bộ cài *Microsoft Visual C++ Redistributable*; nếu máy thiếu DLL khác (ví dụ `msvcp140.dll` dùng bởi onnxruntime/OpenCV), ứng dụng báo và chỉ tới bộ cài chính thức của Microsoft.

## 3. Dữ liệu đánh giá (không phân phối lại, không có trong gói)

| Tập | Điều kiện sử dụng | Hệ quả |
|---|---|---|
| drewnoakes/metadata-extractor-images (ảnh camera cho `ctrl_mei`) | “free to use these media files however you wish” | ảnh sinh ra (`ctrl_mei`) không đưa vào gói để giữ gói gọn và tránh nhầm với dữ liệu công ty |
| Korus Realistic Tampering | “educational and research purposes only” | chỉ dùng đánh giá nội bộ |
| CocoGlide | ảnh COCO theo giấy phép Flickr (nhiều CC BY-NC), nhãn CC BY 4.0 | chỉ dùng đánh giá |
| OpenFake | CC BY-NC 4.0 | chỉ dùng đánh giá, phi thương mại |
| DeepFakeFace | thẻ HF ghi openrail/apache-2.0; ảnh thật từ IMDB-WIKI (nghiên cứu) | chỉ dùng đánh giá |

## 4. Ứng viên đã khảo sát nhưng KHÔNG đóng gói vì giấy phép
- **TruFor** (GRIP-UNINA): chỉ cho mục đích thông tin/phi lợi nhuận.
- **SAFIRE**, **MVSS-Net**, **NPR**: không có tệp giấy phép trong repo → không được phép dùng mặc định.
- **Effort** (CC BY-NC), **SBI** (chỉ nghiên cứu), **DeepfakeBench** (CC BY-NC), **TrueMedia** (trọng số không công khai).
- **CAT-Net v2** (mã Apache-2.0, trọng số CC-BY-4.0 từ 05/08/2026) và **SPAI** (Apache-2.0) có giấy phép phù hợp nhưng **không** được chọn vì chất lượng/tốc độ CPU (xem `docs/RESEARCH_REPORT_VI.md`).

## 5. Quyền riêng tư
- Ảnh người dùng chỉ được xử lý trong tiến trình cục bộ (127.0.0.1); không gửi ra Internet, không lưu xuống đĩa trừ khi người dùng bấm tải hoặc chạy CLI với thư mục kết quả.
- Không dùng credentials của người dùng (ví dụ `.kaggle`) và không đưa credentials, venv hay ảnh riêng vào gói.
