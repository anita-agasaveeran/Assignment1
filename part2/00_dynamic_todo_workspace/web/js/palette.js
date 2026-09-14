/* ⌘K command palette: search tasks, jump to a view or list, run a command. */

import { h, icon, highlight } from './dom.js';
import { formatDue } from './dates.js';
import { VIEWS, searchScore } from './views.js';
import * as S from './store.js';
import { store } from './store.js';

let items = [];
let cursor = 0;
let commands = [];

export function registerCommands(list) { commands = list; }

const backdrop = () => document.getElementById('paletteBackdrop');
const input = () => document.getElementById('paletteInput');
const results = () => document.getElementById('paletteResults');

export const isOpen = () => !backdrop().hidden;

export function open(initial = '') {
  const bd = backdrop();
  bd.hidden = false;
  const box = input();
  box.value = initial;
  box.focus();
  box.select();
  build(initial);
}

export function close() {
  backdrop().hidden = true;
  results().replaceChildren();
}

function build(query) {
  const q = query.trim();
  const lower = q.toLowerCase();
  items = [];

  if (q) {
    const tasks = [...store.tasks.values()]
      .map((t) => ({ t, score: searchScore(t, q) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score || Number(a.t.done) - Number(b.t.done))
      .slice(0, 7);
    for (const { t } of tasks) {
      items.push({
        group: 'Tasks', iconName: t.done ? 'done' : 'all',
        html: highlight(t.title, q),
        sub: [t.dueAt ? formatDue(t.dueAt, t.allDay) : null,
          t.projectId ? S.projectById(t.projectId)?.name : null].filter(Boolean).join(' · '),
        run: () => {
          if (!inCurrentView(t)) S.savePrefs({ view: t.done ? 'completed' : 'all', projectId: null, tag: null });
          store.detailId = t.id;
          S.emit('detail');
          setTimeout(() => document.querySelector(`.task[data-id="${CSS.escape(t.id)}"]`)
            ?.scrollIntoView({ block: 'center', behavior: 'smooth' }), 60);
        },
      });
    }
  }

  const views = VIEWS.filter((v) => !q || v.label.toLowerCase().includes(lower));
  for (const v of views) {
    items.push({ group: 'Go to', iconName: v.icon, html: highlight(v.label, q),
      run: () => S.savePrefs({ view: v.id, projectId: null }) });
  }
  for (const p of store.projects) {
    if (q && !p.name.toLowerCase().includes(lower)) continue;
    items.push({ group: 'Go to', iconName: 'list', html: highlight(p.name, q),
      sub: 'List', run: () => S.savePrefs({ view: 'project', projectId: p.id }) });
  }

  for (const c of commands) {
    if (q && !c.label.toLowerCase().includes(lower)) continue;
    items.push({ group: 'Commands', iconName: c.icon || 'sparkle', html: highlight(c.label, q),
      sub: c.hint, run: c.run });
  }

  if (q) {
    items.push({ group: 'Create', iconName: 'plus',
      html: `Add task <mark>${escapeText(q)}</mark>`,
      sub: 'Parses dates, #tags and !priority',
      run: () => document.dispatchEvent(new CustomEvent('quickadd', { detail: q })) });
  }

  cursor = 0;
  paint(q);
}

const escapeText = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function inCurrentView(task) {
  return document.querySelector(`.task[data-id="${CSS.escape(task.id)}"]`) !== null;
}

function paint(q) {
  const host = results();
  if (!items.length) {
    host.replaceChildren(h('div', { class: 'palette-empty' },
      `No matches for “${q}”. Press Enter to add it as a task.`));
    return;
  }
  const nodes = [];
  let group = null;
  items.forEach((item, i) => {
    if (item.group !== group) {
      group = item.group;
      nodes.push(h('div', { class: 'palette-group' }, group));
    }
    nodes.push(h('button', {
      class: 'presult', type: 'button', role: 'option', dataset: { i: String(i) },
      'aria-selected': String(i === cursor),
      onmousemove: () => { if (cursor !== i) { cursor = i; syncSelection(); } },
      onclick: () => choose(i),
    }, icon(item.iconName),
    h('span', { class: 'p-main', html: item.html }),
    item.sub ? h('span', { class: 'p-sub' }, item.sub) : null));
  });
  host.replaceChildren(...nodes);
}

function syncSelection() {
  results().querySelectorAll('.presult').forEach((el) => {
    const on = Number(el.dataset.i) === cursor;
    el.setAttribute('aria-selected', String(on));
    if (on) el.scrollIntoView({ block: 'nearest' });
  });
}

function choose(i) {
  const item = items[i];
  close();
  item?.run();
}

export function init() {
  const box = input();
  box.addEventListener('input', () => build(box.value));
  box.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || (e.key === 'n' && e.ctrlKey)) {
      e.preventDefault(); cursor = (cursor + 1) % Math.max(items.length, 1); syncSelection();
    } else if (e.key === 'ArrowUp' || (e.key === 'p' && e.ctrlKey)) {
      e.preventDefault(); cursor = (cursor - 1 + items.length) % Math.max(items.length, 1); syncSelection();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (items.length) choose(cursor);
      else { const q = box.value.trim(); close(); if (q) document.dispatchEvent(new CustomEvent('quickadd', { detail: q })); }
    } else if (e.key === 'Escape') {
      e.preventDefault(); close();
    }
    e.stopPropagation();
  });
  backdrop().addEventListener('mousedown', (e) => { if (e.target === backdrop()) close(); });
}
