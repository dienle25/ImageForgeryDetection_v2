// Minimal fake DOM to run the real ifd/webapp/static/app.js render path with real API JSON.
const fs = require("fs");
const [appJs, statusFile, analyzeFile, html] = process.argv.slice(2);
class El {
  constructor(tag, id) { this.tagName = tag; this.id = id || ""; this.children = []; this.parent = null; this._text = "";
    this.hidden = false; this.style = {}; this.dataset = {}; this.listeners = {}; this.className = ""; this.disabled = false;
    const self = this; this.classList = { toggle(c, on) { const s = new Set(self.className.split(" ").filter(Boolean)); (on ?? !s.has(c)) ? s.add(c) : s.delete(c); self.className = [...s].join(" "); }, add(c) { this.toggle(c, true); }, remove(c) { this.toggle(c, false); } }; }
  set textContent(v) { this._text = String(v); this.children.forEach((c) => (c.parent = null)); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(" "); }
  appendChild(c) { if (c.parent) c.parent.children = c.parent.children.filter((x) => x !== c); c.parent = this; this.children.push(c); return c; }
  insertBefore(c, ref) { if (c.parent) c.parent.children = c.parent.children.filter((x) => x !== c); c.parent = this; const i = this.children.indexOf(ref); if (i < 0) this.children.push(c); else this.children.splice(i, 0, c); return c; }
  get childElementCount() { return this.children.length; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  scrollIntoView() {}
}
// build the element tree of index.html: every id, in document order, with the real parent/child nesting of #result
const ids = [...fs.readFileSync(html, "utf8").matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
const byId = {}; ids.forEach((id) => (byId[id] = new El("div", id)));
const result = byId["result"];
["verdict", "primary-slot", "detectors", "warnings", "viewer", "downloads"].forEach((id) => result.appendChild(byId[id]));
["verdict-title", "verdict-quality", "verdict-disagree", "verdict-scope", "verdict-note"].forEach((id) => byId["verdict"].appendChild(byId[id]));
byId["viewer"].appendChild(byId["viewer-title"]);
["verdict-quality", "verdict-disagree", "viewer-title", "result", "picked", "busy", "req-error", "fatal", "faces-legend"].forEach((id) => (byId[id].hidden = true));
const tabs = ["overlay", "heatmap", "faces", "original"].map((v) => { const t = new El("button"); t.dataset.view = v; t.className = "tab"; return t; });
const docListeners = {};
global.document = { getElementById: (id) => byId[id] || (byId[id] = new El("div", id)), createElement: (t) => new El(t),
  addEventListener: (t, f) => (docListeners[t] = f), querySelectorAll: (sel) => (sel === ".tab" ? tabs : []) };
global.location = { search: "?t=tok" };
URL.createObjectURL = () => "blob:x";
const status = JSON.parse(fs.readFileSync(statusFile, "utf8")), analyze = JSON.parse(fs.readFileSync(analyzeFile, "utf8"));
global.fetch = async (path) => ({ ok: true, status: 200, json: async () => (path.startsWith("/api/status") ? status : analyze) });
const errors = []; process.on("unhandledRejection", (e) => errors.push(String(e && e.stack || e)));
eval(fs.readFileSync(appJs, "utf8"));
const flush = () => new Promise((r) => setTimeout(r, 20));
(async () => {
  docListeners["DOMContentLoaded"]();
  await flush();
  byId["file"].listeners.change[0]({ target: { files: [{ name: "Screenshot 2026-10-18 181055.png", size: 1000 }] } });
  byId["analyze"].listeners.click[0]();
  await flush(); await flush();
  if (byId["req-error"].hidden === false) errors.push("req-error: " + byId["req-error"].textContent);
  const order = result.children.filter((c) => !c.hidden).map((c) => c.id + (c.id === "primary-slot" ? "[" + c.children.length + "]" : ""));
  const vt = byId["viewer-title"];
  console.log(JSON.stringify({
    errors, result_hidden: result.hidden, order,
    verdict_class: byId["verdict"].className, headline: byId["verdict-title"].textContent,
    quality_shown: !byId["verdict-quality"].hidden, quality: byId["verdict-quality"].textContent.slice(0, 160),
    disagree_shown: !byId["verdict-disagree"].hidden, disagree: byId["verdict-disagree"].textContent.slice(0, 120),
    primary_card: (byId["primary-slot"].children[0] || {className: "", textContent: ""}).className + " | " + (byId["primary-slot"].children[0] || {textContent: ""}).textContent.slice(0, 200),
    viewer_title_shown: !vt.hidden, viewer_title: vt.textContent,
    other_cards: byId["detectors"].children.map((c) => c.className + " | " + c.textContent.slice(0, 90)),
    scope: byId["scope-body"].textContent.slice(0, 400),
  }, null, 1));
})();
