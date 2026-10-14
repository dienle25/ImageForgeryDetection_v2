# RESUME — tiếp tục công việc từ đâu, bằng lệnh gì

## Môi trường đã dùng
- Cloud Linux x86_64, 2 vCPU, ~8 GB RAM, Python 3.13.16; venv nghiên cứu có torch 2.6.0+cpu, timm 1.0.15, onnx 1.23.2, onnxruntime 1.29.0, opencv 5.0.0.
- Runtime sản phẩm chỉ cần: numpy 2.5.3, Pillow 12.3.0, onnxruntime 1.29.0, opencv-python-headless 5.0.0.93 (`requirements.txt`; Windows: `requirements-lock-win-cp313.txt` có hash).
- Linux: đặt `MALLOC_MMAP_THRESHOLD_=65536` khi nạp mô hình (giảm RSS từ ~4,2 GB xuống ~2,7 GB; `run_app.sh` đã đặt).

## Bố cục
- `ifd/` ứng dụng (web, CLI, self-test, detectors ONNX) · `models/` ONNX + `manifest.json` · `config.json` cấu hình đã khóa.
- `research/` toàn bộ script đánh giá/xuất số liệu (đường dẫn dữ liệu là tham số; repo upstream qua biến `IFD_UPSTREAM`).
- `tools/` xuất ONNX, dựng runtime Windows, đóng gói, kiểm tra gói, benchmark, tham chiếu probe.
- `reports/` bằng chứng thô (manifests, dự đoán từng ảnh, metrics, dedup, self-test, benchmark) · `docs/` báo cáo.

## Lệnh tái lập chính
```bash
# 1) đánh giá dev bằng chính ứng dụng (resumable)
python research/eval_product.py --config config.json --manifest <...> --split dev --split dev_korus --out dev.jsonl
# 2) ngưỡng từ dev
python research/derive_thresholds.py --dev dev.jsonl --manifest <...> --out thresholds.json
# 3) final test một lần với cấu hình đã khóa
python research/eval_product.py --config config.json --manifest <...> --split test --split test_korus --split test_cocoglide --out test.jsonl
python research/final_report.py --run test.jsonl --manifest <...> --config config.json --splits test test_korus test_cocoglide --out metrics.json
# 4) gói Windows
python tools/build_windows_runtime.py --embed python-3.13.16-embed-amd64.zip --wheels wheels/runtime --out runtime_build
python tools/build_package.py --runtime runtime_build --out dist --volume-mb 18   # 18 MiB/phần: kênh chuyển file giới hạn ~20 MB/file
python tools/verify_package.py dist/ImageForgeryDetection_v2.zip --selftest
# 5) sau final test: chỉ số, bảng, quy tắc bật mặc định (P05), phân tích lỗi, trước/sau
python research/final_report.py ... ; python research/summarize_robustness.py ... ; python research/failure_analysis.py ...
python research/make_validation_summary.py --metrics reports/test/metrics_test_v2_standard.json --config config.json \
    --out-summary reports/validation_summary_vi.json --out-targets reports/test/targets_check.json \
    --also-config "research/configs/config_strict.json=nghiêm (FPR ≤ 1 % / ≤ 5 % trên dev, tùy chọn)"
python research/before_after_compare.py ... ; python research/render_tables.py ... --out docs/_tables_test.md
python research/recheck_scores.py --run test.jsonl --manifest ... --config config.json --n 150 --out reports/test/recheck_final_code.json
```

## Trạng thái lúc bàn giao (13/10/2026)
- Final test đã chạy (một lần) → **không chạy lại final test để chỉnh cấu hình**. Muốn cải thiện: thu thập dữ liệu mới (tốt nhất là ảnh có nhãn của công ty), ghi kế hoạch vào `EXPERIMENT_LOG.jsonl` trước khi chạy.
- Việc còn mở: self-test gói cuối trên Windows; kiểm định trên ảnh công ty (`KIEM_DINH_CONG_TY.bat`); pháp lý giấy phép trọng số.
- Máy 2 vCPU/8 GB: không chạy hai tiến trình nạp mô hình cùng lúc (sự cố D02).

## Nếu dừng giữa chừng
Đọc `STATUS.md` (đang ở bước nào) và dòng cuối `EXPERIMENT_LOG.jsonl`. Mọi file `*.jsonl` kết quả có thể chạy tiếp (bỏ qua ảnh đã có).
**Không** chạy lại final test sau khi đã xem kết quả để “chỉnh” cấu hình; nếu cần đổi cấu hình phải ghi sửa đổi và dùng tập test mới.
