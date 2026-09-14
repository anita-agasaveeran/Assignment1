/* Minimal DOM helpers + the icon set (inline SVG so there are no asset fetches). */

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'value' || k === 'checked' || k === 'disabled' || k === 'hidden') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

const P = (d, extra = '') =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" ` +
  `stroke-linecap="round" stroke-linejoin="round" ${extra}>${d}</svg>`;

export const icons = {
  inbox: P('<path d="M3 13h5l1.5 3h5L16 13h5"/><path d="M4.6 5.6L3 13v5a2 2 0 002 2h14a2 2 0 002-2v-5l-1.6-7.4A2 2 0 0017.5 4h-11a2 2 0 00-1.9 1.6z"/>'),
  today: P('<rect x="3" y="4.5" width="18" height="16" rx="2.5"/><path d="M8 3v3M16 3v3M3 10h18"/><circle cx="12" cy="15" r="1.6" fill="currentColor" stroke="none"/>'),
  upcoming: P('<rect x="3" y="4.5" width="18" height="16" rx="2.5"/><path d="M8 3v3M16 3v3M3 10h18M8.5 14h2M13.5 14h2M8.5 17.5h2"/>'),
  all: P('<path d="M4 6.5h16M4 12h16M4 17.5h11"/>'),
  done: P('<circle cx="12" cy="12" r="8.6"/><path d="M8.4 12.2l2.6 2.6 4.6-5"/>'),
  tag: P('<path d="M3.5 11.2V5.4A1.9 1.9 0 015.4 3.5h5.8a2 2 0 011.4.6l7.3 7.3a2 2 0 010 2.8l-5.7 5.7a2 2 0 01-2.8 0L4.1 12.6a2 2 0 01-.6-1.4z"/><circle cx="8" cy="8" r="1.3" fill="currentColor" stroke="none"/>'),
  clock: P('<circle cx="12" cy="12" r="8.6"/><path d="M12 7.6V12l2.8 1.8"/>'),
  calendar: P('<rect x="3" y="4.5" width="18" height="16" rx="2.5"/><path d="M8 3v3M16 3v3M3 10h18"/>'),
  flag: P('<path d="M5 21V4M5 4.6h11.5l-2 3.6 2 3.6H5"/>'),
  notes: P('<path d="M5 4.5h14M5 9.5h14M5 14.5h9M5 19.5h6"/>'),
  trash: P('<path d="M4.5 6.5h15M9.5 6.5V4.8A1.3 1.3 0 0110.8 3.5h2.4a1.3 1.3 0 011.3 1.3v1.7M7 6.5l.8 12.2a1.8 1.8 0 001.8 1.8h4.8a1.8 1.8 0 001.8-1.8L17 6.5"/>'),
  close: P('<path d="M6 6l12 12M18 6L6 18"/>'),
  plus: P('<path d="M12 5v14M5 12h14"/>'),
  check: P('<path d="M5 12.5l4.2 4.2L19 7"/>'),
  arrowRight: P('<path d="M5 12h13M13 6.5l5.5 5.5-5.5 5.5"/>'),
  drag: '<svg viewBox="0 0 24 24"><circle cx="9" cy="6" r="1.6"/><circle cx="15" cy="6" r="1.6"/><circle cx="9" cy="12" r="1.6"/><circle cx="15" cy="12" r="1.6"/><circle cx="9" cy="18" r="1.6"/><circle cx="15" cy="18" r="1.6"/></svg>',
  sun: P('<circle cx="12" cy="12" r="4.2"/><path d="M12 2.6v2.2M12 19.2v2.2M4.2 12H2M22 12h-2.2M6.3 6.3L4.8 4.8M19.2 19.2l-1.5-1.5M17.7 6.3l1.5-1.5M4.8 19.2l1.5-1.5"/>'),
  moon: P('<path d="M20 14.2A8.2 8.2 0 019.8 4 8.6 8.6 0 1020 14.2z"/>'),
  monitor: P('<rect x="3" y="4.5" width="18" height="12" rx="2"/><path d="M9 20.5h6M12 16.5v4"/>'),
  list: P('<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M7.5 9h9M7.5 12.5h9M7.5 16h5"/>'),
  edit: P('<path d="M4 20h4L19 9a2.1 2.1 0 00-3-3L5 17v3z"/>'),
  search: P('<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>'),
  sparkle: P('<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9-1.9 5.1-1.9-5.1L5 10.5l5.1-1.9z"/>'),
};

export const icon = (name, cls = '') => {
  const span = document.createElement('span');
  span.innerHTML = icons[name] || '';
  const svg = span.firstElementChild;
  if (svg && cls) svg.setAttribute('class', cls);
  return svg || document.createTextNode('');
};

export const PROJECT_COLORS = {
  violet: '#6d5efc', blue: '#3e63dd', teal: '#12a594', green: '#30a46c',
  amber: '#f5a524', red: '#e5484d', pink: '#e93d82', slate: '#8b8b98',
};

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** Highlight the matched substrings of a fuzzy/substring search. */
export function highlight(text, query) {
  if (!query) return escapeHtml(text);
  const i = text.toLowerCase().indexOf(query.toLowerCase());
  if (i === -1) return escapeHtml(text);
  return escapeHtml(text.slice(0, i)) + '<mark>' + escapeHtml(text.slice(i, i + query.length)) +
    '</mark>' + escapeHtml(text.slice(i + query.length));
}
