"""Browser end-to-end test of the web UI (real server process + real Chromium via Playwright). Not part of pytest.

    python tools/ui_e2e.py [--face-image PATH] [--out reports/evidence/ui_e2e]

Checks: page loads with the token, model-loading state then ready, upload + analyse a PNG, verdict + 3 detector
cards rendered, tabs (overlay/heatmap/original; faces tab only when faces exist), preview image actually decoded,
ZIP download through the UI link contains result.json, invalid file shows a Vietnamese error, wrong token is
refused, no JavaScript console errors. Screenshots are taken only for the non-face sample (privacy).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-image", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "evidence" / "ui_e2e")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    a.out.mkdir(parents=True, exist_ok=True)
    url_file = a.out / "url.txt"
    url_file.unlink(missing_ok=True)
    srv = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "ifd.webapp.server", "--port", "0", "--no-browser",
                            "--print-url-file", str(url_file)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8")
    checks, console_errors, url = {}, [], ""
    try:
        for _ in range(600):
            if url_file.exists():
                break
            time.sleep(0.2)
        url = url_file.read_text(encoding="utf-8").strip()
        url_file.unlink()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(str(e)))
            t0 = time.perf_counter()
            page.goto(url)
            checks["title"] = page.title()
            checks["loading_message_seen"] = page.locator(".loading").count() > 0
            page.wait_for_selector("#scope-body .scope-item.ok", timeout=600000)
            page.wait_for_function("() => !document.querySelector('#scope-body .loading')", timeout=600000)
            checks["ready_seconds"] = round(time.perf_counter() - t0, 1)
            checks["scope_items"] = page.locator(".scope-item").count()
            scope_text = page.inner_text("#scope-body")
            checks["scope_experimental_badges"] = page.locator("#scope-body .badge-exp").count()
            checks["scope_shows_operating_point"] = "Điểm vận hành" in scope_text
            checks["scope_config_changed_warning"] = "đã khác bản được đánh giá" in scope_text
            sample = ROOT / "tests" / "data" / "splice1.png"
            page.set_input_files("#file", str(sample))
            page.click("#analyze")
            page.wait_for_selector("#result:not([hidden])", timeout=600000)
            checks["verdict_text"] = page.inner_text("#verdict-title")
            checks["detector_cards"] = page.locator(".det").count()
            checks["chips"] = [c.inner_text() for c in page.locator(".det .chip").all()]
            checks["card_experimental_badges"] = page.locator(".det .badge-exp").count()
            checks["tabs_visible"] = [b.get_attribute("data-view") for b in page.locator(".tab").all() if b.is_visible()]
            page.wait_for_function("() => document.getElementById('view-img').complete && document.getElementById('view-img').naturalWidth > 0")
            checks["preview_natural_width"] = page.evaluate("document.getElementById('view-img').naturalWidth")
            page.click(".tab[data-view='heatmap']")
            page.wait_for_function("() => document.getElementById('view-img').src.includes('preview_heatmap')")
            page.screenshot(path=str(a.out / "result_splice1.png"), full_page=True)
            with page.expect_download() as dl:
                page.click("#dl-links a.main")
            zpath = a.out / "download.zip"
            dl.value.save_as(str(zpath))
            names = zipfile.ZipFile(zpath).namelist()
            checks["zip_members"] = names
            checks["zip_has_result_json"] = "result.json" in names
            rj = json.loads(zipfile.ZipFile(zpath).read("result.json"))
            checks["zip_result_source_name"] = rj["source_name"]
            zpath.unlink()
            bad = a.out / "khong_phai_anh.png"
            bad.write_bytes(b"day khong phai anh PNG")
            page.set_input_files("#file", str(bad))
            page.click("#analyze")
            page.wait_for_selector("#req-error:not([hidden])", timeout=120000)
            checks["invalid_file_error"] = page.inner_text("#req-error")
            bad.unlink()
            if a.face_image:
                page.set_input_files("#file", str(a.face_image))
                page.click("#analyze")
                page.wait_for_selector("#result:not([hidden])", timeout=600000)
                checks["face_tabs_visible"] = [b.get_attribute("data-view") for b in page.locator(".tab").all() if b.is_visible()]
                checks["face_card_text"] = [t for t in page.locator(".det").all_inner_texts() if "khuôn mặt" in t.lower()][:1]
            page2 = browser.new_page()
            r = page2.goto(url.split("?")[0] + "api/status?t=wrong")
            checks["wrong_token_status"] = r.status
            browser.close()
        # the browser logs every non-2xx fetch; the 422 of the deliberately invalid upload is expected
        checks["console_errors_ignored_expected_422"] = [c for c in console_errors if "status of 422" in c]
        console_errors = [c for c in console_errors if "status of 422" not in c]
        checks["console_errors"] = console_errors
        ok = (checks["detector_cards"] == 3 and checks["zip_has_result_json"] and checks["preview_natural_width"] > 0
              and "faces" not in checks["tabs_visible"] and checks["wrong_token_status"] == 403 and not console_errors
              and "JPEG" in checks["invalid_file_error"] and not checks["scope_config_changed_warning"])
        summary = ROOT / "reports" / "validation_summary_vi.json"
        if summary.exists():   # after the final test: the experimental badge and the evaluated profile must be shown
            n_exp = sum(1 for v in json.loads(summary.read_text(encoding="utf-8"))["detectors"].values()
                        if v.get("status") == "experimental")
            checks["expected_experimental_detectors"] = n_exp
            ok = ok and checks["scope_experimental_badges"] == n_exp and checks["card_experimental_badges"] == n_exp \
                and checks["scope_shows_operating_point"]
        if a.face_image:
            ok = ok and "faces" in checks.get("face_tabs_visible", [])
        checks["ok"] = bool(ok)
    finally:
        srv.terminate()
        try:
            log = srv.communicate(timeout=20)[0]
        except Exception:
            log = ""
        token = url.split("t=")[-1] if "t=" in url else None
        checks["server_log_tail"] = (log[-1500:].replace(token, "<token>") if token else log[-1500:]) if log else ""
    (a.out / "ui_e2e.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if k != "server_log_tail"}, ensure_ascii=False, indent=1))
    return 0 if checks.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
