"use strict";
(function () {
  const token = new URLSearchParams(location.search).get("t") || "";
  const $ = (id) => document.getElementById(id);
  let maxMb = 25, chosen = null, reqCounter = 0, current = null, ready = false;

  const TASK_HELP = {
    local_manipulation: "Tìm dấu vết ghép ảnh, sao chép vùng, xóa/thay thế nội dung. Có bản đồ nhiệt.",
    ai_generated: "Đánh giá cả ảnh có phải do AI sinh ra hoàn toàn. Chỉ có điểm toàn ảnh, không chỉ ra vùng.",
    face_deepfake: "Tìm khuôn mặt rồi đánh giá từng khuôn mặt có bị hoán đổi/tạo giả.",
  };

  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function show(id, on) { $(id).hidden = !on; }
  function api(path, opts) {
    opts = opts || {}; opts.headers = Object.assign({"X-IFD-Token": token}, opts.headers || {});
    return fetch(path, opts).then(async (r) => {
      let data = null; try { data = await r.json(); } catch (e) { data = {error: "Phản hồi không hợp lệ từ ứng dụng (" + r.status + ")"}; }
      if (!r.ok) { const err = new Error((data && data.error) || ("Lỗi " + r.status)); err.data = data; throw err; }
      return data;
    });
  }
  function fmtBytes(n) { return n > 1048576 ? (n / 1048576).toFixed(2) + " MiB" : (n / 1024).toFixed(0) + " KiB"; }
  function rurl(rid, name, download) { return "/api/result/" + rid + "/" + name + "?t=" + encodeURIComponent(token) + (download ? "&download=1" : ""); }
  function fmtScore(s) { return s >= 0.001 || s === 0 ? s.toFixed(3) : s.toExponential(1); }

  function renderScope(st) {
    const body = $("scope-body"); body.textContent = "";
    $("version").textContent = "phiên bản " + st.app_version;
    maxMb = st.max_file_mb || 25; $("maxmb").textContent = maxMb;
    if (st.error) { $("fatal").textContent = "Ứng dụng chưa dùng được: " + st.error; show("fatal", true); }
    if (!st.ready && !st.error) body.appendChild(el("div", "loading", "Đang nạp và kiểm tra mô hình (lần đầu có thể mất 30–120 giây)…"));
    if (st.config_matches_evaluated === false) body.appendChild(el("div", "warn-text strong",
      "Cấu hình (bộ phát hiện/ngưỡng) đã khác bản được đánh giá trong báo cáo chất lượng — các số liệu chất lượng không còn áp dụng."));
    if (st.evaluated_profile) body.appendChild(el("div", "small muted", "Điểm vận hành: " + st.evaluated_profile +
      " (số liệu trong docs/QUALITY_REPORT_VI.md)."));
    const grid = el("div", "scope-grid");
    (st.detectors || []).forEach((d) => {
      const it = el("div", "scope-item ok" + (d.validation_status === "experimental" ? " exp" : ""));
      const head = el("div", "strong", "Có kiểm tra: " + (d.task_vi || d.task));
      if (d.validation_status === "experimental") head.appendChild(el("span", "badge-exp", "THỬ NGHIỆM"));
      it.appendChild(head);
      it.appendChild(el("div", "small", TASK_HELP[d.task] || ""));
      it.appendChild(el("div", "small muted", "Mô hình: " + d.title));
      if (d.validation) it.appendChild(el("div", "small " + (d.validation_status === "experimental" ? "warn-text" : "muted"), d.validation));
      grid.appendChild(it);
    });
    (st.tasks_not_checked || []).forEach((t) => {
      const it = el("div", "scope-item no");
      it.appendChild(el("div", "strong", "Không kiểm tra: " + t));
      it.appendChild(el("div", "small muted", "Bản này không có bộ phát hiện đã kiểm chứng cho loại này; kết quả không nói gì về nó."));
      grid.appendChild(it);
    });
    const out = el("div", "scope-item no");
    out.appendChild(el("div", "strong", "Ngoài phạm vi"));
    out.appendChild(el("div", "small muted", "Ảnh chụp màn hình, tài liệu/văn bản, đồ họa vẽ, video, sửa metadata, ảnh thật gắn sai chú thích. Ảnh nén/thu nhỏ mạnh có thể đã mất dấu vết."));
    grid.appendChild(out);
    body.appendChild(grid);
    ready = !!st.ready;
    $("analyze").disabled = !(ready && chosen);
  }

  function pollStatus() {
    api("/api/status").then((st) => {
      renderScope(st);
      if (!st.ready && !st.error) setTimeout(pollStatus, 1500);
    }).catch((e) => { $("fatal").textContent = e.message; show("fatal", true); });
  }

  function choose(file) {
    current = null; show("result", false); show("req-error", false);
    if (!file) return;
    const okType = /\.(jpe?g|png)$/i.test(file.name);
    if (!okType) { $("req-error").textContent = "Chỉ hỗ trợ file .jpg, .jpeg, .png."; show("req-error", true); return; }
    if (file.size > maxMb * 1048576) { $("req-error").textContent = "Ảnh vượt giới hạn " + maxMb + " MiB."; show("req-error", true); return; }
    chosen = file;
    $("pick-name").textContent = file.name; $("pick-size").textContent = fmtBytes(file.size);
    const u = URL.createObjectURL(file); $("pick-thumb").src = u; show("picked", true);
    $("analyze").disabled = !ready;
  }

  function scoreBar(score, th) {
    const bar = el("div", "bar");
    [th.low, th.high].forEach((v) => { if (typeof v === "number") { const t = el("div", "tick"); t.style.left = (v * 100) + "%"; bar.appendChild(t); } });
    const f = el("div", "fill"); f.style.left = "calc(" + (Math.max(0, Math.min(1, score)) * 100) + "% - 1px)"; bar.appendChild(f);
    return bar;
  }

  function kv(pairs) {
    const box = el("div", "kv");
    pairs.forEach(([k, v]) => { box.appendChild(el("div", "", k)); box.appendChild(el("div", "", v === null || v === undefined ? "—" : String(v))); });
    return box;
  }

  function detCard(d) {
    const c = el("div", "det");
    const h = el("div", "det-head");
    const title = el("div", "strong", d.task_vi);
    const val = (d.details || {}).validation;
    if (val && val.status === "experimental") title.appendChild(el("span", "badge-exp", "THỬ NGHIỆM"));
    h.appendChild(title);
    h.appendChild(el("span", "chip " + d.decision, d.decision_vi));
    c.appendChild(h);
    const vtext = val && (val.short_vi || val.text_vi);
    if (vtext) c.appendChild(el("div", "small " + (val.status === "experimental" ? "warn-text" : "muted"), vtext));
    c.appendChild(el("div", "small muted", TASK_HELP[d.task] || ""));
    if (typeof d.score === "number") {
      c.appendChild(el("div", "score", fmtScore(d.score)));
      c.appendChild(scoreBar(d.score, d.thresholds || {}));
      c.appendChild(el("div", "small muted", "Vạch: “cần kiểm tra thêm” từ " + d.thresholds.low.toPrecision(3) +
        ", “nghi vấn” từ " + d.thresholds.high.toPrecision(3) + "."));
    }
    if (d.task === "face_deepfake") {
      const det = d.details || {};
      if (!det.faces_found) c.appendChild(el("div", "small", "Không phát hiện khuôn mặt nên không kiểm tra deepfake khuôn mặt."));
      else {
        c.appendChild(el("div", "small", "Phát hiện " + det.faces_found + " khuôn mặt, đã kiểm tra " + det.faces_checked + ". Xem tab “Khuôn mặt”."));
        const ul = el("ul", "faces");
        (det.faces || []).forEach((f, i) => { ul.appendChild(el("li", "small", "Mặt #" + (i + 1) + ": " + fmtScore(f.score) + " — " + ({suspicious: "nghi vấn", needs_review: "cần kiểm tra thêm", no_clear_sign: "chưa thấy dấu hiệu rõ"}[f.decision] || f.decision))); });
        c.appendChild(ul);
      }
    }
    if (d.task === "ai_generated") c.appendChild(el("div", "small muted", "Không có bản đồ vùng cho loại kiểm tra này."));
    (d.warnings || []).forEach((w) => c.appendChild(el("div", "small warn-text", w)));
    c.appendChild(el("div", "small muted fine", d.score_type));
    return c;
  }

  function render(resp) {
    const rep = resp.report, rid = resp.result_id;
    $("res-name").textContent = "— " + rep.source_name;
    const v = $("verdict"); v.className = "verdict " + rep.summary.overall_decision;
    $("verdict-title").textContent = rep.summary.headline_vi;
    const na = (rep.summary.tasks_not_applicable || []).map((t) => t.task_vi + " (" + t.reason_vi + ")");
    const nc = (rep.summary.tasks_not_checked || []).map((t) => t.task_vi);
    $("verdict-scope").textContent = (na.length ? "Không áp dụng: " + na.join("; ") + ". " : "") + (nc.length ? "Không kiểm tra: " + nc.join("; ") + "." : "");
    const dets = $("detectors"); dets.textContent = "";
    rep.detectors.forEach((d) => dets.appendChild(detCard(d)));
    const w = $("warnings"); w.textContent = "";
    const allw = (rep.summary.warnings || []).concat(rep.image.warnings || []);
    Array.from(new Set(allw)).forEach((t) => w.appendChild(el("div", "banner warn", t)));
    const has = (n) => resp.previews.indexOf(n) >= 0;
    document.querySelectorAll(".tab").forEach((b) => { b.hidden = !has("preview_" + b.dataset.view + ".jpg"); });
    setView(has("preview_overlay.jpg") ? "overlay" : (has("preview_faces.jpg") ? "faces" : "original"), rid);
    const dl = $("dl-links"); dl.textContent = "";
    const add = (name, label, main) => { if (name !== "results.zip" && resp.files.indexOf(name) < 0) return; const a = el("a", main ? "main" : "", label); a.href = rurl(rid, name, true); dl.appendChild(a); };
    add("results.zip", "Tải toàn bộ (ZIP)", true); add("result.json", "Báo cáo JSON"); add("overlay.png", "Ảnh chồng lớp PNG");
    add("heatmap.png", "Bản đồ nhiệt PNG"); add("mask.png", "Mask PNG"); add("faces.png", "Khung khuôn mặt PNG");
    add("original.png", "Ảnh đã đọc (PNG)"); add("heatmap.npy", "Bản đồ nhiệt .npy"); add("heatmap_gray16.png", "Bản đồ nhiệt 16-bit");
    const tech = $("tech"); tech.textContent = "";
    const img = rep.image;
    tech.appendChild(kv([["Tên file", rep.source_name], ["SHA-256 file", img.sha256], ["SHA-256 điểm ảnh", rep.decoded_rgb_sha256],
      ["Định dạng", img.format + " · " + img.original_mode], ["Kích thước hiển thị", img.display_size_wh.join(" × ")],
      ["Chất lượng JPEG ước lượng", img.jpeg_quality_estimate], ["EXIF (tham khảo, dễ bị sửa)", JSON.stringify(img.exif)],
      ["Thời gian xử lý", rep.timing_seconds.total_processing.toFixed(2) + " s (nạp mô hình " + rep.timing_seconds.model_load.toFixed(1) + " s, một lần)"],
      ["Phiên bản ứng dụng", rep.app_version], ["Cấu hình (SHA-256)", rep.config_sha256]]));
    rep.detectors.forEach((d) => {
      tech.appendChild(el("div", "strong", "Bộ phát hiện " + d.detector + " — " + d.task_vi));
      const m = (rep.models || {})[d.detector] || {};
      tech.appendChild(kv([["Nguồn mô hình", m.source], ["Giấy phép mã", m.license_code], ["Giấy phép trọng số", m.license_weights],
        ["SHA-256 mô hình", m.sha256], ["Ngưỡng", JSON.stringify(d.thresholds)], ["Tiền xử lý", JSON.stringify(d.details.transform)],
        ["Thời gian", JSON.stringify(d.timing_seconds)]]));
    });
    $("raw-json").textContent = JSON.stringify(rep, null, 2);
    show("result", true);
    $("result").scrollIntoView({behavior: "smooth", block: "start"});
  }

  function setView(view, rid) {
    rid = rid || (current && current.result_id); if (!rid) return;
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
    $("view-img").src = rurl(rid, "preview_" + view + ".jpg", false);
    show("legend", view === "overlay" || view === "heatmap");
    show("faces-legend", view === "faces");
  }

  function analyze() {
    if (!chosen || !ready) return;
    const myId = ++reqCounter; const file = chosen;
    $("analyze").disabled = true; show("busy", true); show("req-error", false); show("result", false);
    api("/api/analyze", {method: "POST", body: file,
      headers: {"X-File-Name": encodeURIComponent(file.name), "X-Request-Id": String(myId), "Content-Type": "application/octet-stream"}})
      .then((resp) => { if (myId !== reqCounter || file !== chosen) return; current = resp; render(resp); })
      .catch((e) => { if (myId !== reqCounter) return; $("req-error").textContent = e.message; show("req-error", true); })
      .finally(() => { if (myId === reqCounter) { show("busy", false); $("analyze").disabled = !(chosen && ready); } });
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!token) { $("fatal").textContent = "Thiếu mã phiên trong địa chỉ. Hãy mở đúng địa chỉ được in trong cửa sổ RUN_APP."; show("fatal", true); return; }
    pollStatus();
    $("file").addEventListener("change", (e) => { reqCounter++; choose(e.target.files[0]); });
    const drop = $("drop");
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { reqCounter++; choose(e.dataTransfer.files[0]); });
    $("analyze").addEventListener("click", analyze);
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  });
})();
