/* Small DOM builders shared by every view. */
export function h(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}
export const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

export function tile({label, value, unit = "", foot = ""}) {
  return h(`<div class="tile">
    <div class="label">${esc(label)}</div>
    <div class="value">${value}${unit ? `<small> ${esc(unit)}</small>` : ""}</div>
    ${foot ? `<div class="foot">${foot}</div>` : ""}
  </div>`);
}

export function card({title, sub = "", right = "", id = ""} = {}) {
  const c = h(`<div class="card" ${id ? `id="${id}"` : ""}>
    <header><h3>${esc(title)}</h3><span class="spacer"></span><span class="hdr-right">${right}</span></header>
    <p class="sub"${sub ? "" : ' hidden'}>${sub}</p>
    <div class="body"></div>
  </div>`);
  c.body = c.querySelector(".body");
  return c;
}

export function table({cols, rows, cls = "", tall = false, raw = false}) {
  const w = h(`<div class="tw ${tall ? "tall" : ""}"><table class="${cls}">
    <thead><tr>${cols.map(c => typeof c === "string"
      ? `<th>${esc(c)}</th>`
      : `<th class="${c.num ? "num" : ""}" ${c.title ? `title="${esc(c.title)}"` : ""}>${esc(c.label)}</th>`
    ).join("")}</tr></thead>
    <tbody>${rows.map(r => `<tr>${r.map((v, i) => {
      const col = cols[i];
      const num = typeof col === "object" && col.num;
      return `<td class="${num ? "num" : ""} ${typeof col === "object" && col.wrap ? "wrap" : ""}">${
        raw ? v : esc(v)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>`);
  return w;
}

export function kv(pairs) {
  return h(`<dl class="kv">${pairs.map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`);
}

export const dot = (s) => `<span class="status-dot s-${s}"></span>`;
export const chip = (text, cls = "") => `<span class="chip ${cls}">${esc(text)}</span>`;
export const note = (html, cls = "") => `<div class="note ${cls}">${html}</div>`;
export const cite = (s) => `<div class="cite">${esc(s)}</div>`;
export const inlineBar = (frac, color = "var(--series-1)") =>
  `<span class="bar-inline"><i style="width:${Math.max(0, Math.min(1, frac)) * 100}%;background:${color}"></i></span>`;

export function section(el, head) {
  el.innerHTML = "";
  if (head) el.appendChild(h(`<div class="view-head"><p>${head}</p></div>`));
  return el;
}
export function grid(cls, kids) {
  const g = h(`<div class="grid ${cls}"></div>`);
  kids.filter(Boolean).forEach(k => g.appendChild(k));
  return g;
}
