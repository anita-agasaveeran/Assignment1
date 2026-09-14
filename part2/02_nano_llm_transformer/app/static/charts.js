/* Minimal SVG chart primitives.
 *
 * Deliberately dependency-free: the dashboard must run offline on a laptop with
 * no CDN. Follows the project's viz rules -- 2px lines, hairline solid grid,
 * recessive axes, legend whenever >= 2 series, selective direct labels (endpoint
 * only), tooltips on every plot, and a hard cap of 3 categorical series so the
 * palette stays inside its validated all-pairs range.
 */
const SVGNS = "http://www.w3.org/2000/svg";

const CH = (() => {
  const registry = new Map();
  let uid = 0;

  const css = (name) => getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
  const palette = () => ({
    series: [css("--series-1"), css("--series-2"), css("--series-3")],
    grid: css("--grid"), axis: css("--axis"), muted: css("--muted"),
    text1: css("--text-1"), text2: css("--text-2"), surface: css("--surface-1"),
    good: css("--good"), critical: css("--critical"), warning: css("--warning"),
    successTx: css("--success-tx"), divMid: css("--div-mid"),
  });

  const el = (tag, attrs = {}, parent = null) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined) continue;
      n.setAttribute(k, v);
    }
    if (parent) parent.appendChild(n);
    return n;
  };

  const fmt = (v, d) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (d !== undefined) return v.toFixed(d);
    if (a === 0) return "0";
    if (a >= 1e9) return (v / 1e9).toFixed(2) + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + "K";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    if (a >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  };

  // ---------------------------------------------------------------- tooltip
  const tip = () => document.getElementById("tooltip");
  function showTip(html, x, y) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const r = t.getBoundingClientRect();
    let left = x + 14, top = y - r.height / 2;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    top = Math.max(8, Math.min(top, window.innerHeight - r.height - 8));
    t.style.left = left + "px";
    t.style.top = top + "px";
  }
  const hideTip = () => { tip().hidden = true; };

  // ----------------------------------------------------------------- scales
  function makeScale(domain, range, type = "linear") {
    let [d0, d1] = domain;
    if (type === "log") {
      d0 = Math.max(d0, 1e-9); d1 = Math.max(d1, d0 * 10);
      const l0 = Math.log10(d0), l1 = Math.log10(d1);
      return (v) => range[0] + (Math.log10(Math.max(v, 1e-9)) - l0) / (l1 - l0) * (range[1] - range[0]);
    }
    if (d1 === d0) d1 = d0 + 1;
    return (v) => range[0] + (v - d0) / (d1 - d0) * (range[1] - range[0]);
  }

  function niceTicks(min, max, n = 5) {
    if (!isFinite(min) || !isFinite(max)) return [0, 1];
    if (min === max) { min -= 0.5; max += 0.5; }
    const span = max - min;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
      out.push(Math.round(v / step) * step);
    }
    return out;
  }

  function logTicks(min, max) {
    const out = [];
    for (let e = Math.floor(Math.log10(Math.max(min, 1e-9)));
         e <= Math.ceil(Math.log10(Math.max(max, 1e-9))); e++) {
      out.push(Math.pow(10, e));
    }
    return out.filter((v) => v >= min * 0.9 && v <= max * 1.1);
  }

  // ---------------------------------------------------------------- shell
  function frame(container, height, pad) {
    container.innerHTML = "";
    const w = Math.max(container.clientWidth || 640, 260);
    const svg = el("svg", {
      class: "chart", viewBox: `0 0 ${w} ${height}`, width: "100%", height,
      preserveAspectRatio: "none", role: "img",
    }, container);
    return { svg, w, h: height, pad };
  }

  function axes(svg, w, h, pad, xs, ys, xTicks, yTicks, p, opts = {}) {
    const g = el("g", {}, svg);
    for (const t of yTicks) {
      const y = ys(t);
      if (y < pad.t - 1 || y > h - pad.b + 1) continue;
      el("line", { x1: pad.l, x2: w - pad.r, y1: y, y2: y, stroke: p.grid, "stroke-width": 1 }, g);
      el("text", {
        x: pad.l - 7, y: y + 3.5, "text-anchor": "end", fill: p.muted,
        "font-size": 10.5, "font-family": "var(--mono)",
      }, g).textContent = opts.yFmt ? opts.yFmt(t) : fmt(t);
    }
    let lastLabel = null;
    for (const t of xTicks) {
      const x = xs(t);
      if (x < pad.l - 1 || x > w - pad.r + 1) continue;
      const label = opts.xFmt ? opts.xFmt(t) : fmt(t);
      if (label === lastLabel) continue;             // e.g. a single-point series
      lastLabel = label;
      el("text", {
        x, y: h - pad.b + 15, "text-anchor": "middle", fill: p.muted,
        "font-size": 10.5, "font-family": "var(--mono)",
      }, g).textContent = label;
    }
    el("line", { x1: pad.l, x2: w - pad.r, y1: h - pad.b, y2: h - pad.b,
                 stroke: p.axis, "stroke-width": 1 }, g);
    if (opts.yLabel) {
      el("text", { x: 11, y: pad.t - 8, fill: p.muted, "font-size": 10.5 }, g)
        .textContent = opts.yLabel;
    }
    if (opts.xLabel) {
      el("text", { x: w - pad.r, y: h - 2, "text-anchor": "end", fill: p.muted,
                   "font-size": 10.5 }, g).textContent = opts.xLabel;
    }
    return g;
  }

  function legend(container, items) {
    if (items.length < 2) return;             // one series: the title names it
    const d = document.createElement("div");
    d.className = "legend";
    d.innerHTML = items.map((s) =>
      `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
    container.parentNode.insertBefore(d, container);
  }

  // ------------------------------------------------------------ line chart
  /* opts: series [{name, points:[[x,y]], color?, dashed?}], xLabel, yLabel,
     yScale 'linear'|'log', xScale, height, yFmt, xFmt, band {y0,y1,label},
     markers [{x, label, color}] */
  function line(container, opts) {
    const render = () => {
      const p = palette();
      const height = opts.height || 210;
      const pad = { l: 52, r: 16, t: 12, b: 26 };
      const { svg, w, h } = frame(container, height, pad);
      const S = (opts.series || []).filter((s) => s.points && s.points.length);
      if (!S.length) {
        el("text", { x: w / 2, y: h / 2, "text-anchor": "middle", fill: p.muted,
                     "font-size": 12 }, svg).textContent = "no data yet";
        return;
      }
      S.forEach((s, i) => { s.color = s.color || p.series[i % 3]; });

      const xsAll = S.flatMap((s) => s.points.map((d) => d[0]));
      const ysAll = S.flatMap((s) => s.points.map((d) => d[1])).filter(Number.isFinite);
      let xmin = opts.xMin ?? Math.min(...xsAll), xmax = opts.xMax ?? Math.max(...xsAll);
      let ymin = opts.yMin ?? Math.min(...ysAll), ymax = opts.yMax ?? Math.max(...ysAll);
      if (opts.band) { ymin = Math.min(ymin, opts.band.y0); ymax = Math.max(ymax, opts.band.y1); }
      if (opts.yScale !== "log") {
        const padY = (ymax - ymin) * 0.10 || Math.abs(ymax || 1) * 0.1;
        ymin -= padY; ymax += padY;
        if (opts.yZero) { ymin = Math.min(0, ymin); ymax = Math.max(0, ymax); }
      }
      const xs = makeScale([xmin, xmax], [pad.l, w - pad.r], opts.xScale || "linear");
      const ys = makeScale([ymin, ymax], [h - pad.b, pad.t], opts.yScale || "linear");
      const xTicks = opts.xScale === "log" ? logTicks(xmin, xmax) : niceTicks(xmin, xmax, 6);
      const yTicks = opts.yScale === "log" ? logTicks(ymin, ymax) : niceTicks(ymin, ymax, 5);
      axes(svg, w, h, pad, xs, ys, xTicks, yTicks, p, opts);

      if (opts.band) {   // e.g. the seed-noise band around a baseline
        const y1 = ys(opts.band.y1), y0 = ys(opts.band.y0);
        el("rect", { x: pad.l, y: Math.min(y0, y1), width: w - pad.l - pad.r,
                     height: Math.abs(y0 - y1), fill: p.muted, opacity: 0.13 }, svg);
        if (opts.band.label) {
          el("text", { x: w - pad.r - 4, y: Math.min(y0, y1) - 3, "text-anchor": "end",
                       fill: p.muted, "font-size": 10 }, svg).textContent = opts.band.label;
        }
      }

      for (const s of S) {
        const pts = s.points.filter((d) => Number.isFinite(d[1]));
        const dstr = pts.map((d, i) => `${i ? "L" : "M"}${xs(d[0]).toFixed(1)},${ys(d[1]).toFixed(1)}`).join("");
        el("path", { d: dstr, fill: "none", stroke: s.color, "stroke-width": 2,
                     "stroke-linejoin": "round", "stroke-linecap": "round",
                     "stroke-dasharray": s.dashed ? "5 4" : null,
                     opacity: s.faint ? 0.45 : 1 }, svg);
        if (pts.length === 1) {
          el("circle", { cx: xs(pts[0][0]), cy: ys(pts[0][1]), r: 4, fill: s.color }, svg);
        }
      }
      // selective direct label: endpoint only, and only for <= 3 series
      if (S.length <= 3 && opts.directLabel !== false) {
        for (const s of S) {
          const last = s.points.filter((d) => Number.isFinite(d[1])).slice(-1)[0];
          if (!last) continue;
          const lx = xs(last[0]), ly = ys(last[1]);
          if (lx > w - pad.r - 3) continue;
          el("circle", { cx: lx, cy: ly, r: 3.2, fill: s.color, stroke: p.surface,
                         "stroke-width": 2 }, svg);
        }
      }
      for (const m of opts.markers || []) {
        const x = xs(m.x);
        el("line", { x1: x, x2: x, y1: pad.t, y2: h - pad.b, stroke: m.color || p.good,
                     "stroke-width": 1.5, opacity: 0.55 }, svg);
        if (m.label) {
          el("text", { x: x + 3, y: pad.t + 9, fill: m.color || p.good, "font-size": 9.5 }, svg)
            .textContent = m.label;
        }
      }

      // crosshair + tooltip
      const cross = el("line", { y1: pad.t, y2: h - pad.b, stroke: p.axis,
                                 "stroke-width": 1, opacity: 0 }, svg);
      const dots = S.map((s) => el("circle", {
        r: 4, fill: s.color, stroke: p.surface, "stroke-width": 2, opacity: 0 }, svg));
      const hit = el("rect", { x: pad.l, y: pad.t, width: w - pad.l - pad.r,
                               height: h - pad.t - pad.b, fill: "transparent" }, svg);
      hit.style.cursor = "crosshair";
      hit.addEventListener("mousemove", (e) => {
        const rect = svg.getBoundingClientRect();
        const sx = (e.clientX - rect.left) * (w / rect.width);
        let bx = null; const rows = [];
        S.forEach((s, i) => {
          let best = null, bd = Infinity;
          for (const d of s.points) {
            if (!Number.isFinite(d[1])) continue;
            const dd = Math.abs(xs(d[0]) - sx);
            if (dd < bd) { bd = dd; best = d; }
          }
          if (best) {
            bx = best[0];
            dots[i].setAttribute("cx", xs(best[0]));
            dots[i].setAttribute("cy", ys(best[1]));
            dots[i].setAttribute("opacity", 1);
            rows.push(`<div class="tt-row"><span><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${s.color};margin-right:5px"></i>${s.name}</span><b>${opts.tipFmt ? opts.tipFmt(best[1]) : fmt(best[1])}</b></div>`);
          } else dots[i].setAttribute("opacity", 0);
        });
        if (bx === null) return;
        cross.setAttribute("x1", xs(bx)); cross.setAttribute("x2", xs(bx));
        cross.setAttribute("opacity", 0.85);
        const label = opts.xLabel ? `${opts.xLabel} ${opts.xFmt ? opts.xFmt(bx) : fmt(bx)}`
                                  : (opts.xFmt ? opts.xFmt(bx) : fmt(bx));
        showTip(`<div class="tt-title">${label}</div>${rows.join("")}`, e.clientX, e.clientY);
      });
      hit.addEventListener("mouseleave", () => {
        cross.setAttribute("opacity", 0);
        dots.forEach((d) => d.setAttribute("opacity", 0));
        hideTip();
      });
      legend(container, S.map((s) => ({ name: s.name, color: s.color })));
    };
    mount(container, render);
  }

  // ------------------------------------------------------------- bar chart
  /* opts: items [{label, value, color?, note?, tip?}], horizontal, diverging,
     band {lo,hi,label}, valueFmt, height, maxLabel */
  function bars(container, opts) {
    const render = () => {
      const p = palette();
      const items = opts.items || [];
      const horiz = opts.horizontal !== false;
      const rowH = opts.rowH || 22;
      // bottom band must hold tick labels AND the axis title without overlap
      const bBand = horiz ? (opts.xLabel ? 34 : 22) : 42;
      const height = opts.height || (horiz ? Math.max(90, items.length * rowH + 14 + bBand) : 220);
      const labelW = opts.labelW || (horiz ? 150 : 0);
      const pad = { l: horiz ? labelW : 52, r: 54, t: 8, b: bBand };
      const { svg, w, h } = frame(container, height, pad);
      if (!items.length) {
        el("text", { x: w / 2, y: h / 2, "text-anchor": "middle", fill: p.muted,
                     "font-size": 12 }, svg).textContent = "no data yet";
        return;
      }
      const vals = items.map((d) => d.value).filter(Number.isFinite);
      let vmin = Math.min(0, ...vals), vmax = Math.max(0, ...vals);
      if (opts.band) { vmin = Math.min(vmin, opts.band.lo); vmax = Math.max(vmax, opts.band.hi); }
      const span = (vmax - vmin) || 1;
      vmin -= span * 0.06; vmax += span * 0.06;

      if (horiz) {
        const xs = makeScale([vmin, vmax], [pad.l, w - pad.r]);
        const zero = xs(0);
        for (const t of niceTicks(vmin, vmax, 5)) {
          el("line", { x1: xs(t), x2: xs(t), y1: pad.t, y2: h - pad.b, stroke: p.grid,
                       "stroke-width": 1 }, svg);
          el("text", { x: xs(t), y: h - pad.b + 14, "text-anchor": "middle", fill: p.muted,
                       "font-size": 10, "font-family": "var(--mono)" }, svg)
            .textContent = opts.valueFmt ? opts.valueFmt(t) : fmt(t);
        }
        if (opts.band) {
          el("rect", { x: xs(opts.band.lo), y: pad.t, width: Math.max(xs(opts.band.hi) - xs(opts.band.lo), 1),
                       height: h - pad.t - pad.b, fill: p.muted, opacity: 0.14 }, svg);
          if (opts.band.label) {
            el("text", { x: (xs(opts.band.lo) + xs(opts.band.hi)) / 2, y: pad.t + 9,
                         "text-anchor": "middle", fill: p.muted, "font-size": 9.5 }, svg)
              .textContent = opts.band.label;
          }
        }
        el("line", { x1: zero, x2: zero, y1: pad.t, y2: h - pad.b, stroke: p.axis,
                     "stroke-width": 1 }, svg);

        items.forEach((d, i) => {
          const y = pad.t + i * rowH + 2;
          const bh = rowH - 6;                     // 2px+ surface gap between bars
          const v = Number.isFinite(d.value) ? d.value : 0;
          const x0 = Math.min(zero, xs(v)), x1 = Math.max(zero, xs(v));
          const color = d.color || (opts.diverging
            ? (v < 0 ? p.series[0] : p.critical) : p.series[0]);
          const g = el("g", {}, svg);
          el("rect", { x: x0, y, width: Math.max(x1 - x0, 1.5), height: bh, rx: 4, ry: 4,
                       fill: color }, g);
          el("text", { x: pad.l - 8, y: y + bh / 2 + 3.5, "text-anchor": "end",
                       fill: p.text2, "font-size": 11.5 }, g).textContent = d.label;
          el("text", { x: (v < 0 ? x0 - 6 : x1 + 6), y: y + bh / 2 + 3.5,
                       "text-anchor": v < 0 ? "end" : "start", fill: p.text2,
                       "font-size": 10.5, "font-family": "var(--mono)" }, g)
            .textContent = opts.valueFmt ? opts.valueFmt(v) : fmt(v);
          const hit = el("rect", { x: pad.l, y, width: w - pad.l - pad.r, height: bh,
                                   fill: "transparent" }, g);
          hit.addEventListener("mousemove", (e) => showTip(
            d.tip || `<div class="tt-title">${d.label}</div><div class="tt-row"><span>value</span><b>${opts.valueFmt ? opts.valueFmt(v) : fmt(v)}</b></div>`,
            e.clientX, e.clientY));
          hit.addEventListener("mouseleave", hideTip);
        });
      } else {
        const ys = makeScale([vmin, vmax], [h - pad.b, pad.t]);
        const bw = (w - pad.l - pad.r) / items.length;
        for (const t of niceTicks(vmin, vmax, 5)) {
          el("line", { x1: pad.l, x2: w - pad.r, y1: ys(t), y2: ys(t), stroke: p.grid,
                       "stroke-width": 1 }, svg);
          el("text", { x: pad.l - 7, y: ys(t) + 3.5, "text-anchor": "end", fill: p.muted,
                       "font-size": 10, "font-family": "var(--mono)" }, svg)
            .textContent = opts.valueFmt ? opts.valueFmt(t) : fmt(t);
        }
        const zero = ys(0);
        items.forEach((d, i) => {
          const x = pad.l + i * bw + 1.5;
          const bwid = Math.max(bw - 3, 1.5);
          const v = Number.isFinite(d.value) ? d.value : 0;
          const y0 = Math.min(zero, ys(v)), y1 = Math.max(zero, ys(v));
          const g = el("g", {}, svg);
          el("rect", { x, y: y0, width: bwid, height: Math.max(y1 - y0, 1.5), rx: 4, ry: 4,
                       fill: d.color || p.series[0] }, g);
          // Histograms pre-thin their labels, so they opt out of the width guard
          // that stops dense categorical bars from colliding.
          if (d.label && (opts.forceLabels || bw > 26)) {
            el("text", { x: x + bwid / 2, y: h - pad.b + 13, "text-anchor": "middle",
                         fill: p.muted, "font-size": 10 }, g).textContent = d.label;
          }
          const hit = el("rect", { x, y: pad.t, width: bwid, height: h - pad.t - pad.b,
                                   fill: "transparent" }, g);
          hit.addEventListener("mousemove", (e) => showTip(
            d.tip || `<div class="tt-title">${d.label}</div><div class="tt-row"><span>value</span><b>${opts.valueFmt ? opts.valueFmt(v) : fmt(v)}</b></div>`,
            e.clientX, e.clientY));
          hit.addEventListener("mouseleave", hideTip);
        });
        el("line", { x1: pad.l, x2: w - pad.r, y1: zero, y2: zero, stroke: p.axis,
                     "stroke-width": 1 }, svg);
      }
      if (opts.xLabel) {
        el("text", { x: w - pad.r, y: h - 2, "text-anchor": "end", fill: p.muted,
                     "font-size": 10.5 }, svg).textContent = opts.xLabel;
      }
    };
    mount(container, render);
  }

  // -------------------------------------------------------------- histogram
  function hist(container, opts) {
    const edges = opts.edges || [], counts = opts.counts || [];
    const every = Math.max(1, Math.ceil(counts.length / 7));
    bars(container, {
      horizontal: false, height: opts.height || 190,
      xLabel: opts.xLabel, forceLabels: true,
      items: counts.map((c, i) => ({
        label: i % every === 0 ? fmt(edges[i]) : "",
        value: c,
        tip: `<div class="tt-title">${fmt(edges[i])} – ${fmt(edges[i + 1])}</div>
              <div class="tt-row"><span>count</span><b>${c.toLocaleString()}</b></div>`,
      })),
      valueFmt: (v) => fmt(v),
    });
  }

  // ------------------------------------------------------- reliability plot
  /* Calibration: bar = observed accuracy, reference line = perfect calibration. */
  function reliability(container, binsIn, height = 210) {
    const render = () => {
      const p = palette();
      const pad = { l: 46, r: 14, t: 22, b: 34 };     // room for the y-axis title above "1.00"
      const { svg, w, h } = frame(container, height, pad);
      const bins = (binsIn || []).filter((b) => b.count > 0);
      if (!bins.length) {
        el("text", { x: w / 2, y: h / 2, "text-anchor": "middle", fill: p.muted,
                     "font-size": 12 }, svg).textContent = "no data yet";
        return;
      }
      const xs = makeScale([0, 1], [pad.l, w - pad.r]);
      const ys = makeScale([0, 1], [h - pad.b, pad.t]);
      for (const t of [0, 0.25, 0.5, 0.75, 1]) {
        el("line", { x1: pad.l, x2: w - pad.r, y1: ys(t), y2: ys(t), stroke: p.grid,
                     "stroke-width": 1 }, svg);
        el("text", { x: pad.l - 7, y: ys(t) + 3.5, "text-anchor": "end", fill: p.muted,
                     "font-size": 10, "font-family": "var(--mono)" }, svg).textContent = t.toFixed(2);
        el("text", { x: xs(t), y: h - pad.b + 14, "text-anchor": "middle", fill: p.muted,
                     "font-size": 10, "font-family": "var(--mono)" }, svg).textContent = t.toFixed(2);
      }
      const bw = (w - pad.l - pad.r) / 10;
      for (const b of bins) {
        const x = xs(b.bin - 0.05) + 1.5;
        const y = ys(b.accuracy);
        el("rect", { x, y, width: Math.max(bw - 3, 2),
                     height: Math.max(ys(0) - y, 1.5), rx: 4, ry: 4,
                     fill: p.series[0], opacity: 0.9 }, svg);
        const hit = el("rect", { x, y: pad.t, width: Math.max(bw - 3, 2),
                                 height: h - pad.t - pad.b, fill: "transparent" }, svg);
        hit.addEventListener("mousemove", (e) => showTip(
          `<div class="tt-title">confidence bin ${(b.bin - 0.05).toFixed(2)}–${(b.bin + 0.05).toFixed(2)}</div>
           <div class="tt-row"><span>mean confidence</span><b>${fmt(b.confidence, 3)}</b></div>
           <div class="tt-row"><span>accuracy</span><b>${fmt(b.accuracy, 3)}</b></div>
           <div class="tt-row"><span>gap</span><b>${fmt(b.accuracy - b.confidence, 3)}</b></div>
           <div class="tt-row"><span>tokens</span><b>${b.count.toLocaleString()}</b></div>`,
          e.clientX, e.clientY));
        hit.addEventListener("mouseleave", hideTip);
      }
      el("line", { x1: xs(0), y1: ys(0), x2: xs(1), y2: ys(1), stroke: p.text2,
                   "stroke-width": 2, opacity: 0.55 }, svg);
      el("text", { x: xs(1) - 4, y: ys(1) + 13, "text-anchor": "end", fill: p.text2,
                   "font-size": 10 }, svg).textContent = "perfect calibration";
      el("line", { x1: pad.l, x2: w - pad.r, y1: h - pad.b, y2: h - pad.b, stroke: p.axis,
                   "stroke-width": 1 }, svg);
      el("text", { x: w - pad.r, y: h - 2, "text-anchor": "end", fill: p.muted,
                   "font-size": 10.5 }, svg).textContent = "predicted confidence";
      el("text", { x: 11, y: pad.t - 9, fill: p.muted, "font-size": 10.5 }, svg)
        .textContent = "observed accuracy";
    };
    mount(container, render);
  }

  // ----------------------------------------------------------------- mount
  function mount(container, render) {
    const id = container.dataset.chartId || (container.dataset.chartId = "c" + (++uid));
    registry.set(id, { container, render });
    render();
  }

  let raf = null;
  const rerenderAll = () => {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      for (const [id, r] of registry) {
        if (!document.body.contains(r.container)) { registry.delete(id); continue; }
        // legends are inserted as siblings; drop stale ones before re-rendering
        const prev = r.container.previousElementSibling;
        if (prev && prev.classList.contains("legend")) prev.remove();
        r.render();
      }
    });
  };
  window.addEventListener("resize", rerenderAll);

  return { line, bars, hist, reliability, fmt, palette, showTip, hideTip,
           rerenderAll, clear: () => registry.clear() };
})();
