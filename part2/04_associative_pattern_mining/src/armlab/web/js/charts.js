/* Minimal SVG chart library built to the dataviz mark specs.
   No CDN dependency: the dashboard must run offline behind the FastAPI service.

   Every chart ships (a) a hover tooltip, (b) a legend when there are >=2 series,
   and (c) a table-view twin, which is also the documented relief for the
   light-mode slots that sit below 3:1 contrast on the surface. */

const NS = "http://www.w3.org/2000/svg";
const el = (n, a = {}, kids = []) => {
  const e = document.createElementNS(NS, n);
  for (const [k, v] of Object.entries(a)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  for (const c of [].concat(kids)) if (c) e.appendChild(c);
  return e;
};
const txt = (s, a = {}) => { const t = el("text", a); t.textContent = s; return t; };
const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim();
export const SERIES = () => [1,2,3,4,5].map(i => css(`--series-${i}`));
export const SEQ = () => [100,200,300,400,500,600,700].map(i => css(`--seq-${i}`));

export const fmt = {
  n: (v, d = 0) => v === null || v === undefined || !isFinite(v) ? "–"
      : Number(v).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d}),
  pct: (v, d = 1) => v === null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%",
  f: (v, d = 3) => v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d),
  sci: (v) => v === null || !isFinite(v) ? "–"
      : (v === 0 ? "0" : (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e5 ? v.toExponential(1) : v.toPrecision(4))),
  // An exact 0.0 here is floating-point underflow in the hypergeometric tail,
  // not a probability of zero. Say so rather than printing a misleading "0".
  p: (v) => v === null || v === undefined || !isFinite(v) ? "–"
      : v === 0 ? "<1e-308" : v < 1e-4 ? v.toExponential(1) : v.toFixed(4),
  compact: (v) => !isFinite(v) ? "–"
      : Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(v),
};

/* ---------------- tooltip singleton ---------------- */
let TIP;
function tip() {
  if (!TIP) { TIP = document.createElement("div"); TIP.className = "tip"; document.body.appendChild(TIP); }
  return TIP;
}
function showTip(ev, title, rows) {
  const t = tip();
  t.innerHTML = `<div class="t-title">${title}</div>` +
    rows.map(([k, v]) => `<div class="t-row"><span>${k}</span><b>${v}</b></div>`).join("");
  t.classList.add("show");
  const r = t.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  t.style.left = x + "px"; t.style.top = y + "px";
}
const hideTip = () => tip().classList.remove("show");

/* ---------------- scales & axes ---------------- */
const lin = (d0, d1, r0, r1) => (v) => d1 === d0 ? r0 : r0 + (v - d0) * (r1 - r0) / (d1 - d0);
function ticks(min, max, n = 5) {
  if (!isFinite(min) || !isFinite(max)) return [0];
  if (min === max) return [min];
  const span = max - min, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out.length ? out : [min, max];
}

function frame(w, h, m) {
  const g = el("g");
  return {g, iw: w - m.l - m.r, ih: h - m.t - m.b};
}

function yAxis(g, sy, dom, m, iw, fmtFn) {
  for (const t of ticks(dom[0], dom[1])) {
    const y = sy(t);
    g.appendChild(el("line", {class: "grid-line", x1: m.l, x2: m.l + iw, y1: y, y2: y}));
    g.appendChild(txt(fmtFn(t), {class: "tick", x: m.l - 7, y: y + 3.5, "text-anchor": "end"}));
  }
}

/* ---------------- mount helper: svg + legend + table twin ---------------- */
function mount(host, {svg, legend, table, note}) {
  host.innerHTML = "";
  if (legend && legend.length > 1) {
    const L = document.createElement("div"); L.className = "legend";
    L.innerHTML = legend.map(s =>
      `<span class="k"><i class="sw${s.dot ? " dot" : ""}" style="background:${s.color}"></i>${s.name}</span>`).join("");
    host.appendChild(L);
  }
  const plot = document.createElement("div"); plot.appendChild(svg); host.appendChild(plot);
  const foot = document.createElement("div"); foot.className = "chart-foot";
  if (table) {
    const b = document.createElement("button"); b.className = "btn"; b.textContent = "Table view";
    const tw = document.createElement("div"); tw.className = "tw tall"; tw.style.display = "none";
    tw.innerHTML = `<table><thead><tr>${table.cols.map((c, i) =>
        `<th class="${i ? "num" : ""}">${c}</th>`).join("")}</tr></thead><tbody>${
      table.rows.map(r => `<tr>${r.map((v, i) =>
        `<td class="${i ? "num" : ""}">${v}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    b.onclick = () => {
      const on = tw.style.display === "none";
      tw.style.display = on ? "" : "none"; plot.style.display = on ? "none" : "";
      b.classList.toggle("on", on); b.textContent = on ? "Chart view" : "Table view";
    };
    foot.appendChild(b); host.appendChild(foot); host.appendChild(tw);
  }
  if (note) { const n = document.createElement("span"); n.className = "cite"; n.textContent = note; foot.appendChild(n); }
  if (!foot.parentNode) host.appendChild(foot);
}

/* Charts measure their container at draw time. A container that is hidden or
   not yet laid out reports zero width, which silently renders every chart at the
   320px floor -- so instead of measuring once, each chart registers a redraw and
   an observer repaints it whenever its container's width actually changes. This
   also makes the whole dashboard respond to window resizing for free. */
function responsive(host, draw) {
  host.__armDraw = draw;
  draw();
  // A chart drawn while its container is still detached (built inside a grid
  // before that grid is appended) measures zero and lands on the 320px floor.
  // Re-check once the browser has laid the container out, and thereafter
  // whenever its width actually changes.
  // ResizeObserver delivers an initial callback as soon as the element has a
  // box, which is exactly the detached -> attached transition we need to catch.
  if (!host.__armObserver) {
    host.__armObserver = new ResizeObserver(() => {
      const cw = host.clientWidth;
      if (cw > 0 && Math.abs(cw - (host.__armWidth || 0)) > 8) host.__armDraw && host.__armDraw();
    });
    host.__armObserver.observe(host);
  }
}

function sized(host, height) {
  // Respect a real measurement even when it is small -- a hard floor here makes
  // charts overflow narrow grid cells. The 640 fallback applies only when the
  // container has no layout at all (detached, or the tab is hidden), and the
  // ResizeObserver repaints once it does.
  const measured = host.clientWidth || host.parentElement?.clientWidth || 0;
  const w = measured > 0 ? Math.max(200, measured) : 640;
  host.__armWidth = w;
  const svg = el("svg", {class: "chart", width: w, height, viewBox: `0 0 ${w} ${height}`});
  return {svg, w};
}

/* ================= vertical bar (one series) ================= */
function _bar(host, o) {
  const {data, height = 220, color = SERIES()[0], yfmt = fmt.compact,
         tipTitle = (d) => d.label, tipRows = (d) => [["value", fmt.n(d.value)]],
         direct = false, xEvery = 1, yLabel = ""} = o;
  const {svg, w} = sized(host, height);
  const m = {t: 12, r: 12, b: 34, l: 48};
  const {g, iw, ih} = frame(w, height, m);
  const max = Math.max(...data.map(d => d.value), 0) * 1.08 || 1;
  const sy = lin(0, max, m.t + ih, m.t);
  yAxis(g, sy, [0, max], m, iw, yfmt);
  const bw = iw / data.length;
  data.forEach((d, i) => {
    const x = m.l + i * bw + 1, bh = Math.max(0, m.t + ih - sy(d.value));
    const rect = el("rect", {x, y: sy(d.value), width: Math.max(1, bw - 2), height: bh,
      fill: color, rx: Math.min(4, (bw - 2) / 2)});
    rect.addEventListener("mousemove", (e) => showTip(e, tipTitle(d), tipRows(d)));
    rect.addEventListener("mouseleave", hideTip);
    g.appendChild(rect);
    if (direct && bh > 16 && data.length <= 16)
      g.appendChild(txt(yfmt(d.value), {class: "dlabel", x: x + (bw - 2) / 2,
        y: sy(d.value) - 5, "text-anchor": "middle"}));
    if (i % xEvery === 0)
      g.appendChild(txt(d.label, {class: "tick", x: x + (bw - 2) / 2,
        y: m.t + ih + 15, "text-anchor": "middle"}));
  });
  g.appendChild(el("line", {class: "axis-line", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih}));
  if (yLabel) g.appendChild(txt(yLabel, {class: "axis-title", x: m.l, y: 9}));
  svg.appendChild(g);
  mount(host, {svg, table: {cols: [o.xName || "Category", o.yName || "Value"],
    rows: data.map(d => [d.label, fmt.n(d.value)])}});
}

/* ================= horizontal bar ================= */
function _hbar(host, o) {
  const {data, color = SERIES()[0], rowH = 22, labelW = 200, xfmt = fmt.compact,
         tipRows = (d) => [["value", fmt.n(d.value)]]} = o;
  const h = data.length * rowH + 30;
  const {svg, w} = sized(host, h);
  const m = {t: 6, r: 54, b: 22, l: labelW};
  const iw = w - m.l - m.r;
  const g = el("g");
  const max = Math.max(...data.map(d => d.value), 0) || 1;
  const sx = lin(0, max, 0, iw);
  data.forEach((d, i) => {
    const y = m.t + i * rowH;
    const bw = Math.max(1, sx(d.value));
    const r = el("rect", {x: m.l, y: y + 3, width: bw, height: rowH - 8, fill: color, rx: 4});
    r.addEventListener("mousemove", (e) => showTip(e, d.label, tipRows(d)));
    r.addEventListener("mouseleave", hideTip);
    g.appendChild(r);
    const lab = d.label.length > 30 ? d.label.slice(0, 29) + "…" : d.label;
    g.appendChild(txt(lab, {class: "tick", x: m.l - 8, y: y + rowH / 2 + 2.5, "text-anchor": "end"}));
    g.appendChild(txt(xfmt(d.value), {class: "dlabel", x: m.l + bw + 6, y: y + rowH / 2 + 3}));
  });
  g.appendChild(el("line", {class: "axis-line", x1: m.l, x2: m.l, y1: m.t, y2: m.t + data.length * rowH}));
  svg.appendChild(g);
  mount(host, {svg, table: {cols: [o.xName || "Item", o.yName || "Value"],
    rows: data.map(d => [d.label, fmt.n(d.value)])}});
}

/* ================= line / multi-series with crosshair ================= */
function _line(host, o) {
  const {series, height = 260, xfmt = fmt.compact, yfmt = fmt.f, xLabel = "", yLabel = "",
         markers = true, yDomain = null, tipRows = null, xIsIndex = false} = o;
  const {svg, w} = sized(host, height);
  const m = {t: 14, r: 16, b: 36, l: 52};
  const {g, iw, ih} = frame(w, height, m);
  const xs = series.flatMap(s => s.points.map(p => p.x));
  const ys = series.flatMap(s => s.points.map(p => p.y)).filter(isFinite);
  const xd = [Math.min(...xs), Math.max(...xs)];
  const yd = yDomain || [Math.min(0, Math.min(...ys)), Math.max(...ys) * 1.06];
  const sx = lin(xd[0], xd[1], m.l, m.l + iw), sy = lin(yd[0], yd[1], m.t + ih, m.t);
  yAxis(g, sy, yd, m, iw, yfmt);
  for (const t of ticks(xd[0], xd[1], 6))
    g.appendChild(txt(xfmt(t), {class: "tick", x: sx(t), y: m.t + ih + 16, "text-anchor": "middle"}));
  g.appendChild(el("line", {class: "axis-line", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih}));

  series.forEach((s) => {
    const pts = s.points.filter(p => isFinite(p.y));
    if (!pts.length) return;
    if (s.type !== "dots") {
      const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`).join(" ");
      g.appendChild(el("path", {d, fill: "none", stroke: s.color, "stroke-width": s.width || 2,
        "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-dasharray": s.dash || null, opacity: s.opacity || 1}));
    }
    if (markers || s.type === "dots") {
      const surf = css("--surface-1");
      pts.forEach(p => {
        const c = el("circle", {cx: sx(p.x), cy: sy(p.y), r: s.r || (s.type === "dots" ? 4 : 3.6),
          fill: s.color, stroke: surf, "stroke-width": 2, opacity: p.dim ? .45 : 1});
        const hit = el("circle", {cx: sx(p.x), cy: sy(p.y), r: 12, class: "hit"});
        const rows = tipRows ? tipRows(p, s) : [["x", xfmt(p.x)], ["y", yfmt(p.y)]];
        hit.addEventListener("mousemove", (e) => showTip(e, p.label || s.name, rows));
        hit.addEventListener("mouseleave", hideTip);
        g.appendChild(c); g.appendChild(hit);
      });
    }
    if (s.directLabel && pts.length) {
      const last = pts[pts.length - 1];
      g.appendChild(txt(s.name, {class: "dlabel", x: sx(last.x) - 4, y: sy(last.y) - 9,
        "text-anchor": "end", fill: s.color}));
    }
  });
  if (yLabel) g.appendChild(txt(yLabel, {class: "axis-title", x: m.l, y: 9}));
  if (xLabel) g.appendChild(txt(xLabel, {class: "axis-title", x: m.l + iw, y: height - 4, "text-anchor": "end"}));
  svg.appendChild(g);
  const cols = [xLabel || "x", ...series.map(s => s.name)];
  const xset = [...new Set(xs)].sort((a, b) => a - b);
  mount(host, {svg, legend: series.map(s => ({name: s.name, color: s.color, dot: s.type === "dots"})),
    table: {cols, rows: xset.map(x => [xfmt(x), ...series.map(s => {
      const p = s.points.find(pp => pp.x === x); return p ? yfmt(p.y) : "–";})])}});
}

/* ================= scatter with sequential color ================= */
function _scatter(host, o) {
  const {points, height = 300, xfmt = fmt.f, yfmt = fmt.f, xLabel = "", yLabel = "",
         cLabel = "", tipRows, logX = false} = o;
  const {svg, w} = sized(host, height);
  const m = {t: 14, r: 74, b: 38, l: 54};
  const {g, iw, ih} = frame(w, height, m);
  const tx = (v) => logX ? Math.log10(Math.max(v, 1e-9)) : v;
  const xs = points.map(p => tx(p.x)), ys = points.map(p => p.y), cs = points.map(p => p.c);
  const xd = [Math.min(...xs), Math.max(...xs)], yd = [Math.min(...ys), Math.max(...ys)];
  const pad = (d) => { const s = (d[1] - d[0]) * .07 || .1; return [d[0] - s, d[1] + s]; };
  const XD = pad(xd), YD = pad(yd);
  const sx = lin(XD[0], XD[1], m.l, m.l + iw), sy = lin(YD[0], YD[1], m.t + ih, m.t);
  const seq = SEQ(), cmin = Math.min(...cs), cmax = Math.max(...cs);
  const colr = (v) => seq[Math.min(seq.length - 1, Math.max(0,
    Math.round((v - cmin) / ((cmax - cmin) || 1) * (seq.length - 1))))];
  yAxis(g, sy, YD, m, iw, yfmt);
  for (const t of ticks(XD[0], XD[1], 6))
    g.appendChild(txt(logX ? xfmt(Math.pow(10, t)) : xfmt(t),
      {class: "tick", x: sx(t), y: m.t + ih + 16, "text-anchor": "middle"}));
  g.appendChild(el("line", {class: "axis-line", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih}));
  const surf = css("--surface-1");
  points.forEach(p => {
    const c = el("circle", {cx: sx(tx(p.x)), cy: sy(p.y), r: p.r || 5, fill: colr(p.c),
      stroke: surf, "stroke-width": 2});
    const hit = el("circle", {cx: sx(tx(p.x)), cy: sy(p.y), r: 13, class: "hit"});
    const rows = tipRows ? tipRows(p) : [[xLabel, xfmt(p.x)], [yLabel, yfmt(p.y)], [cLabel, fmt.f(p.c)]];
    hit.addEventListener("mousemove", (e) => showTip(e, p.label || "point", rows));
    hit.addEventListener("mouseleave", hideTip);
    g.appendChild(c); g.appendChild(hit);
  });
  // sequential scale legend
  const lx = m.l + iw + 16;
  seq.forEach((c, i) => g.appendChild(el("rect", {x: lx, y: m.t + i * 15, width: 12, height: 13, fill: c, rx: 2})));
  g.appendChild(txt(fmt.f(cmax, 2), {class: "tick", x: lx + 16, y: m.t + 9}));
  g.appendChild(txt(fmt.f(cmin, 2), {class: "tick", x: lx + 16, y: m.t + seq.length * 15 - 2}));
  if (cLabel) g.appendChild(txt(cLabel, {class: "axis-title", x: lx, y: m.t - 5}));
  if (yLabel) g.appendChild(txt(yLabel, {class: "axis-title", x: m.l, y: 9}));
  if (xLabel) g.appendChild(txt(xLabel, {class: "axis-title", x: m.l + iw, y: height - 4, "text-anchor": "end"}));
  svg.appendChild(g);
  mount(host, {svg, table: {cols: [xLabel || "x", yLabel || "y", cLabel || "c"],
    rows: points.map(p => [xfmt(p.x), yfmt(p.y), fmt.f(p.c)])}});
}

/* ================= stacked bar ================= */
function _stacked(host, o) {
  const {categories, series, height = 240, yfmt = fmt.f, yLabel = "", tipRows} = o;
  const {svg, w} = sized(host, height);
  const m = {t: 14, r: 12, b: 40, l: 52};
  const {g, iw, ih} = frame(w, height, m);
  const totals = categories.map((_, i) => series.reduce((a, s) => a + (s.values[i] || 0), 0));
  const max = Math.max(...totals) * 1.08 || 1;
  const sy = lin(0, max, m.t + ih, m.t);
  yAxis(g, sy, [0, max], m, iw, yfmt);
  const bw = iw / categories.length;
  const surf = css("--surface-1");
  categories.forEach((cat, i) => {
    let acc = 0;
    series.forEach(s => {
      const v = s.values[i] || 0; if (v <= 0) return;
      const y0 = sy(acc + v), y1 = sy(acc);
      const h = Math.max(0, y1 - y0 - 2);            // 2px surface gap between segments
      const r = el("rect", {x: m.l + i * bw + 3, y: y0, width: Math.max(1, bw - 6),
        height: h, fill: s.color, rx: 3, stroke: surf, "stroke-width": 0});
      const rows = tipRows ? tipRows(s, cat, v) : [[s.name, yfmt(v)]];
      r.addEventListener("mousemove", (e) => showTip(e, cat, rows));
      r.addEventListener("mouseleave", hideTip);
      g.appendChild(r); acc += v;
    });
    g.appendChild(txt(yfmt(totals[i]), {class: "dlabel", x: m.l + i * bw + bw / 2,
      y: sy(totals[i]) - 6, "text-anchor": "middle"}));
    const lab = cat.length > 14 ? cat.slice(0, 13) + "…" : cat;
    g.appendChild(txt(lab, {class: "tick", x: m.l + i * bw + bw / 2, y: m.t + ih + 15, "text-anchor": "middle"}));
  });
  g.appendChild(el("line", {class: "axis-line", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih}));
  if (yLabel) g.appendChild(txt(yLabel, {class: "axis-title", x: m.l, y: 9}));
  svg.appendChild(g);
  mount(host, {svg, legend: series.map(s => ({name: s.name, color: s.color})),
    table: {cols: ["Configuration", ...series.map(s => s.name), "Total"],
      rows: categories.map((c, i) => [c, ...series.map(s => yfmt(s.values[i] || 0)), yfmt(totals[i])])}});
}

/* ================= force-directed network ================= */
function _network(host, o) {
  const {nodes, edges, height = 520, maxNodes = 70} = o;
  const keep = nodes.slice(0, maxNodes);
  const ids = new Set(keep.map(n => n.id));
  const E = edges.filter(e => ids.has(e.source) && ids.has(e.target));
  const {svg, w} = sized(host, height);
  const N = keep.map((n, i) => ({...n,
    x: w / 2 + Math.cos(i * 2.4) * (60 + i * 3.2),
    y: height / 2 + Math.sin(i * 2.4) * (50 + i * 2.6), vx: 0, vy: 0}));
  const idx = new Map(N.map((n, i) => [n.id, i]));
  // Deterministic Fruchterman-Reingold style relaxation.
  const k = Math.sqrt((w * height) / Math.max(N.length, 1)) * 0.42;
  for (let it = 0; it < 260; it++) {
    const t = 1 - it / 260;
    for (let i = 0; i < N.length; i++) for (let j = i + 1; j < N.length; j++) {
      let dx = N[i].x - N[j].x, dy = N[i].y - N[j].y;
      let d2 = dx * dx + dy * dy || .01; const d = Math.sqrt(d2);
      const f = (k * k) / d2 * 0.9;
      dx /= d; dy /= d;
      N[i].vx += dx * f; N[i].vy += dy * f; N[j].vx -= dx * f; N[j].vy -= dy * f;
    }
    for (const e of E) {
      const a = N[idx.get(e.source)], b = N[idx.get(e.target)];
      let dx = b.x - a.x, dy = b.y - a.y; const d = Math.hypot(dx, dy) || .01;
      const f = (d * d) / k * 0.012; dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const n of N) {
      const sp = Math.hypot(n.vx, n.vy) || 1, lim = Math.min(sp, 26 * t);
      n.x += n.vx / sp * lim; n.y += n.vy / sp * lim;
      n.vx *= .82; n.vy *= .82;
      n.x = Math.max(26, Math.min(w - 26, n.x)); n.y = Math.max(20, Math.min(height - 20, n.y));
    }
  }
  const g = el("g");
  const maxLift = Math.max(...E.map(e => e.lift), 1);
  const seq = SEQ();
  for (const e of E) {
    const a = N[idx.get(e.source)], b = N[idx.get(e.target)];
    const t = e.lift / maxLift;
    g.appendChild(el("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y,
      stroke: seq[Math.min(seq.length - 1, Math.round(t * (seq.length - 1)))],
      "stroke-width": 1 + t * 2.4, opacity: .55, "stroke-linecap": "round"}));
  }
  const maxW = Math.max(...N.map(n => n.weight), 1);
  const surf = css("--surface-1"), c1 = SERIES()[0];
  for (const n of N) {
    const r = 4 + 9 * Math.sqrt(n.weight / maxW);
    const c = el("circle", {cx: n.x, cy: n.y, r, fill: c1, stroke: surf, "stroke-width": 2});
    const hit = el("circle", {cx: n.x, cy: n.y, r: Math.max(r, 13), class: "hit"});
    hit.addEventListener("mousemove", (ev) => showTip(ev, n.label,
      [["stock code", n.code], ["item support", fmt.pct(n.support)],
       ["graph weight", fmt.f(n.weight, 1)]]));
    hit.addEventListener("mouseleave", hideTip);
    g.appendChild(c); g.appendChild(hit);
    if (r > 8.5) {
      const lab = n.label.length > 20 ? n.label.slice(0, 19) + "…" : n.label;
      g.appendChild(txt(lab, {class: "dlabel", x: n.x, y: n.y - r - 5, "text-anchor": "middle"}));
    }
  }
  svg.appendChild(g);
  mount(host, {svg, table: {cols: ["Item", "Stock code", "Item support", "Graph weight"],
    rows: N.map(n => [n.label, n.code, fmt.pct(n.support), fmt.f(n.weight, 1)])},
    note: `${N.length} items, ${E.length} rule edges. Edge shade = lift.`});
}

export const bar     = (host, o) => responsive(host, () => _bar(host, o));
export const hbar    = (host, o) => responsive(host, () => _hbar(host, o));
export const line    = (host, o) => responsive(host, () => _line(host, o));
export const scatter = (host, o) => responsive(host, () => _scatter(host, o));
export const stacked = (host, o) => responsive(host, () => _stacked(host, o));
export const network = (host, o) => responsive(host, () => _network(host, o));

export {showTip, hideTip};
