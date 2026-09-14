/* Hand-rolled SVG charts.
 *
 * A charting library would be ~90 KB to draw four figures whose shapes we
 * already know exactly, so these are built by hand: crisp at any width, themed
 * from the same CSS variables as everything else, and no third-party runtime. */

const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const COLORS = {
    taxi: "#f7b500", blue: "#4c8dff", green: "#34d399",
    violet: "#a78bfa", grid: "#1f2b45", muted: "#8b9ab8", faint: "#5d6b8a",
  };

  const el = (name, attrs = {}, text) => {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  };

  const svg = (w, h) => el("svg", {
    viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "xMidYMid meet", role: "img",
  });

  const empty = (host, msg) => {
    host.innerHTML = `<div class="empty">${msg}</div>`;
  };

  const label = (x, y, text, opts = {}) => el("text", {
    x, y, fill: opts.fill || COLORS.faint, "font-size": opts.size || 9,
    "text-anchor": opts.anchor || "middle", "font-family": "inherit",
    "font-weight": opts.weight || 400,
  }, text);

  /* Horizontal gridlines plus their value labels — shared by every chart so
   * the panels read as one family. */
  function axis(g, { x0, x1, yScale, ticks, fmt = (v) => v }) {
    ticks.forEach((t) => {
      const y = yScale(t);
      g.append(el("line", {
        x1: x0, x2: x1, y1: y, y2: y,
        stroke: COLORS.grid, "stroke-width": 1,
      }));
      g.append(label(x0 - 6, y + 3, fmt(t), { anchor: "end" }));
    });
  }

  /* ---------------------------------------------------------------- sweep */
  /* Duration by departure hour, with the p10-p90 band behind the line.  The
   * band is the point: it shows that leaving at 3am is not just faster, it is
   * far more *predictable* than leaving at 6pm. */
  function sweep(host, series, opts = {}) {
    if (!series || !series.length) return empty(host, "Estimate a trip to see the hour-by-hour sweep.");
    const W = 360, H = 190, P = { t: 12, r: 10, b: 24, l: 34 };
    const s = svg(W, H);
    const mins = series.map((d) => d.duration_seconds / 60);
    const los = series.map((d) => d.low_seconds / 60);
    const his = series.map((d) => d.high_seconds / 60);

    const yMax = Math.max(...his) * 1.08;
    const yMin = Math.min(...los) * 0.9;
    const X = (h) => P.l + (h / 23) * (W - P.l - P.r);
    const Y = (v) => H - P.b - ((v - yMin) / (yMax - yMin || 1)) * (H - P.t - P.b);

    const nTicks = 4;
    const ticks = Array.from({ length: nTicks }, (_, i) => yMin + (i * (yMax - yMin)) / (nTicks - 1));
    axis(s, { x0: P.l, x1: W - P.r, yScale: Y, ticks, fmt: (v) => Math.round(v) });

    const band = series.map((d, i) => `${X(i)},${Y(his[i])}`).join(" ") + " " +
      series.map((d, i) => `${X(23 - i)},${Y(los[23 - i])}`).join(" ");
    s.append(el("polygon", { points: band, fill: COLORS.taxi, opacity: 0.13 }));

    s.append(el("polyline", {
      points: series.map((d, i) => `${X(i)},${Y(mins[i])}`).join(" "),
      fill: "none", stroke: COLORS.taxi, "stroke-width": 2.2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));

    const best = mins.indexOf(Math.min(...mins));
    const worst = mins.indexOf(Math.max(...mins));
    [[best, COLORS.green], [worst, "#f87171"]].forEach(([i, c]) => {
      s.append(el("circle", { cx: X(i), cy: Y(mins[i]), r: 4, fill: c, stroke: "#0e1524", "stroke-width": 2 }));
    });

    if (opts.selectedHour != null) {
      const h = opts.selectedHour;
      s.append(el("line", {
        x1: X(h), x2: X(h), y1: P.t, y2: H - P.b,
        stroke: COLORS.blue, "stroke-width": 1.4, "stroke-dasharray": "3 3", opacity: 0.85,
      }));
      s.append(el("circle", { cx: X(h), cy: Y(mins[h]), r: 3.5, fill: COLORS.blue }));
    }

    [0, 6, 12, 18, 23].forEach((h) => s.append(label(X(h), H - 7, `${h}:00`)));
    s.append(label(8, P.t + 2, "min", { anchor: "start", size: 8.5 }));
    host.replaceChildren(s);
  }

  /* -------------------------------------------------------------- columns */
  function columns(host, values, { labels = [], color = COLORS.blue, fmt = (v) => v,
                                    highlight = null, height = 150 } = {}) {
    if (!values || !values.length) return empty(host, "No data.");
    const W = 360, H = height, P = { t: 10, r: 8, b: 22, l: 34 };
    const s = svg(W, H);
    const yMax = Math.max(...values) * 1.1 || 1;
    const Y = (v) => H - P.b - (v / yMax) * (H - P.t - P.b);
    const bw = (W - P.l - P.r) / values.length;

    axis(s, {
      x0: P.l, x1: W - P.r, yScale: Y,
      ticks: [0, yMax / 2, yMax], fmt,
    });

    values.forEach((v, i) => {
      const isHi = highlight === i;
      s.append(el("rect", {
        x: P.l + i * bw + bw * 0.14, y: Y(v),
        width: bw * 0.72, height: Math.max(H - P.b - Y(v), 0.5),
        fill: isHi ? COLORS.taxi : color, opacity: isHi ? 1 : 0.8, rx: 1.5,
      }));
    });
    labels.forEach(([i, text]) => s.append(label(P.l + i * bw + bw / 2, H - 6, text)));
    host.replaceChildren(s);
  }

  /* ----------------------------------------------------------------- line */
  function line(host, values, { labels = [], color = COLORS.green, fmt = (v) => v,
                                 unit = "", height = 150 } = {}) {
    if (!values || !values.length) return empty(host, "No data.");
    const W = 360, H = height, P = { t: 12, r: 10, b: 22, l: 34 };
    const s = svg(W, H);
    const yMax = Math.max(...values) * 1.08, yMin = Math.min(...values) * 0.9;
    const X = (i) => P.l + (i / (values.length - 1)) * (W - P.l - P.r);
    const Y = (v) => H - P.b - ((v - yMin) / (yMax - yMin || 1)) * (H - P.t - P.b);

    axis(s, {
      x0: P.l, x1: W - P.r, yScale: Y,
      ticks: [yMin, (yMin + yMax) / 2, yMax], fmt,
    });

    const pts = values.map((v, i) => `${X(i)},${Y(v)}`).join(" ");
    s.append(el("polygon", {
      points: `${P.l},${H - P.b} ${pts} ${W - P.r},${H - P.b}`,
      fill: color, opacity: 0.1,
    }));
    s.append(el("polyline", {
      points: pts, fill: "none", stroke: color, "stroke-width": 2,
      "stroke-linejoin": "round",
    }));
    labels.forEach(([i, text]) => s.append(label(X(i), H - 6, text)));
    if (unit) s.append(label(8, P.t + 2, unit, { anchor: "start", size: 8.5 }));
    host.replaceChildren(s);
  }

  /* --------------------------------------------------------------- ranked */
  /* Bar widths are square-rooted.  Permutation importance here is dominated by
   * one feature by a factor of ~25, and on a linear axis every other bar
   * collapses to a sliver — the ranking becomes unreadable while telling you
   * only what the top row already said.  Sqrt keeps the order and the winner's
   * dominance visible while leaving ranks 2-12 legible.  Printed values stay
   * raw. */
  function ranked(host, items, { color = COLORS.violet, n = 12 } = {}) {
    if (!items || !items.length) return empty(host, "No data.");
    const top = items.slice(0, n);
    const max = Math.sqrt(Math.max(...top.map((d) => d.value)) || 1);
    const rowH = 21, W = 360, H = top.length * rowH + 6;
    const s = svg(W, H);
    const barX = 128;

    top.forEach((d, i) => {
      const y = i * rowH + 4;
      s.append(label(barX - 8, y + 11, d.label, { anchor: "end", size: 10, fill: COLORS.muted }));
      const w = Math.sqrt(d.value) / max;
      s.append(el("rect", {
        x: barX, y: y + 3, width: Math.max(w * (W - barX - 44), 1),
        height: 11, fill: color, opacity: 0.55 + 0.45 * w, rx: 2.5,
      }));
      s.append(label(W - 38, y + 12, d.display ?? d.value.toFixed(3),
        { anchor: "start", size: 9.5 }));
    });
    host.replaceChildren(s);
  }

  /* -------------------------------------------------------------- heatmap */
  function heatmap(host, matrix, { rows = [], cols = [] } = {}) {
    if (!matrix || !matrix.length) return empty(host, "No data.");
    const W = 360, cellW = (W - 34) / 24, cellH = 17, H = matrix.length * cellH + 22;
    const s = svg(W, H);
    const max = Math.max(...matrix.flat()) || 1;

    // Dark navy -> ember -> gold.  A three-stop ramp rather than a channel
    // formula: it stays monotonic in brightness, so denser cells always read
    // as brighter, and it lands on the panel's own taxi yellow at the top.
    const STOPS = [[16, 26, 48], [150, 62, 66], [247, 181, 0]];
    const ramp = (t) => {
      const i = t < 0.5 ? 0 : 1;
      const k = t < 0.5 ? t * 2 : (t - 0.5) * 2;
      const [a, b] = [STOPS[i], STOPS[i + 1]];
      return `rgb(${a.map((v, j) => Math.round(v + (b[j] - v) * k)).join(",")})`;
    };

    matrix.forEach((row, r) => row.forEach((v, c) => {
      // Square-root scaling: pickup counts span two orders of magnitude, and a
      // linear ramp would flatten every off-peak hour into the same black.
      const t = Math.sqrt(v / max);
      const col = ramp(t);
      s.append(el("rect", {
        x: 34 + c * cellW, y: r * cellH + 2, width: cellW - 1, height: cellH - 2,
        fill: col, rx: 1.5,
      }));
    }));
    rows.forEach((name, r) => s.append(label(29, r * cellH + 14, name, { anchor: "end", size: 9 })));
    cols.forEach(([c, text]) => s.append(label(34 + c * cellW + cellW / 2, H - 6, text)));
    host.replaceChildren(s);
  }

  return { sweep, columns, line, ranked, heatmap, COLORS };
})();
