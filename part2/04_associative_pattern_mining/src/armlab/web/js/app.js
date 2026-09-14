/* Router + shell wiring. Views render lazily and are cached after first paint. */
import {A} from "./api.js";
import {h, esc} from "./ui.js";
import * as VD from "./views_data.js";
import * as VM from "./views_model.js";
import {fmt} from "./charts.js";

const ROUTES = {
  overview:     ["Overview", VD.overview],
  business:     ["Business understanding · Phase 1", VD.business],
  data:         ["Data understanding · Phase 2", VD.data],
  prep:         ["Data preparation · Phase 3", VD.prep],
  itemsets:     ["Frequent itemsets · Phase 4", VM.itemsets],
  rules:        ["Rule explorer · Phase 4", VM.rules],
  autoresearch: ["AutoResearch · Phase 4", VM.autoresearch],
  network:      ["Rule network · Phase 4", VM.network],
  evaluation:   ["Evaluation · Phase 5", VM.evaluation],
  measures:     ["Measures & properties · Phase 5", VM.measures],
  deployment:   ["Deployment · Phase 6", VM.deployment],
  recommender:  ["Live recommender · Phase 6", VM.recommender],
};

const painted = new Set();

async function show(name) {
  if (!ROUTES[name]) name = "overview";
  const [title, render] = ROUTES[name];
  document.getElementById("page-title").textContent = title;
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll("#nav a").forEach(a =>
    a.classList.toggle("active", a.getAttribute("href") === "#" + name));
  const el = document.getElementById("view-" + name);
  el.classList.add("active");
  if (painted.has(name)) return;
  el.innerHTML = `<div class="loading">loading ${esc(title)}…</div>`;
  try {
    await render(el);
    painted.add(name);
  } catch (e) {
    el.innerHTML = `<div class="err"><b>Could not render this view.</b><br>${esc(e.message)}
      <br><br>If the artifacts are missing, build them first:
      <span class="mono">./run_pipeline.sh</span></div>`;
    console.error(e);
  }
}

/* theme */
const THEME_KEY = "armlab-theme";
function applyTheme(t) {
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
}
function currentTheme() {
  return document.documentElement.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
try { const s = localStorage.getItem(THEME_KEY); if (s) applyTheme(s); } catch {}
document.getElementById("theme-toggle").onclick = () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch {}
  // Charts read CSS custom properties at draw time, so repaint the open view.
  painted.clear();
  show(location.hash.slice(1) || "overview");
};

addEventListener("hashchange", () => show(location.hash.slice(1) || "overview"));

(async function init() {
  try {
    const m = await A.manifest();
    const run = m.run || {};
    document.getElementById("chip-dataset").textContent =
      `${fmt.compact(run.preparation?.n_transactions ?? 0)} baskets · ${fmt.compact(run.preparation?.n_items ?? 0)} items`;
    document.getElementById("chip-fingerprint").textContent =
      `run ${m.config ? (run.provenance?.run_fingerprint ?? "") : ""}`.trim() || "run";
    const fp = run.provenance?.run_fingerprint;
    if (fp) document.getElementById("chip-fingerprint").textContent = "run " + fp;
  } catch (e) {
    document.getElementById("chip-dataset").textContent = "artifacts not built";
    document.getElementById("chip-fingerprint").className = "chip bad";
    document.getElementById("chip-fingerprint").textContent = "run ./run_pipeline.sh";
  }
  show(location.hash.slice(1) || "overview");
})();
