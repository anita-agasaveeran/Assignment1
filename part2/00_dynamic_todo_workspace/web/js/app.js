/* Bootstrap: wires the store to the DOM, owns global keyboard handling. */

import { h, icon } from './dom.js';
import { parse } from './parse.js';
import { formatDue, startOfDay, toISO } from './dates.js';
import { SORTS } from './views.js';
import * as S from './store.js';
import { store } from './store.js';
import { toast, openModal, modalOpen, announce } from './ui.js';
import * as palette from './palette.js';
import {
  render, renderDetail, openDetail, closeDetail, startInlineEdit,
  newProjectDialog, clearCompletedFlow, visibleIds,
} from './render.js';

const $ = (id) => document.getElementById(id);

/* ── theme ───────────────────────────────────────────────────────────── */
const THEMES = ['system', 'light', 'dark'];
function applyTheme() {
  const t = store.prefs.theme;
  document.documentElement.dataset.theme = t;
  const btn = $('themeBtn');
  btn.replaceChildren(icon({ system: 'monitor', light: 'sun', dark: 'moon' }[t]));
  btn.title = `Theme: ${t} (T)`;
  btn.setAttribute('aria-label', `Theme: ${t}. Click to change.`);
}
function cycleTheme() {
  const next = THEMES[(THEMES.indexOf(store.prefs.theme) + 1) % THEMES.length];
  S.savePrefs({ theme: next });
  applyTheme();
  announce(`Theme set to ${next}`);
}

/* ── composer ────────────────────────────────────────────────────────── */
function composerDefaults() {
  const d = {};
  if (store.prefs.view === 'project') d.projectId = store.prefs.projectId;
  if (store.prefs.view === 'today') { d.dueAt = toISO(startOfDay()); d.allDay = true; }
  if (store.prefs.view === 'upcoming') { d.dueAt = toISO(new Date(Date.now() + 864e5)); d.allDay = true; }
  if (store.prefs.tag) d.tags = [store.prefs.tag];
  return d;
}

function quickAdd(text) {
  const raw = (text || '').trim();
  if (!raw) return null;
  const parsed = parse(raw, { projects: store.projects });
  const defaults = composerDefaults();
  const task = {
    title: parsed.title || raw,
    priority: parsed.priority,
    dueAt: parsed.dueAt ?? defaults.dueAt ?? null,
    allDay: parsed.dueAt ? parsed.allDay : (defaults.allDay ?? true),
    tags: parsed.tags.length ? parsed.tags : (defaults.tags || []),
    projectId: parsed.projectId ?? defaults.projectId ?? null,
  };
  const created = S.createTask(task);
  const bits = [];
  if (task.dueAt) bits.push(formatDue(task.dueAt, task.allDay));
  if (task.projectId) bits.push(S.projectById(task.projectId)?.name);
  announce(`Added ${task.title}${bits.length ? ', ' + bits.join(', ') : ''}`);
  return created;
}

function renderParseChips(value) {
  const host = $('parseChips');
  if (!value.trim()) { host.replaceChildren(); return; }
  const p = parse(value, { projects: store.projects });
  const chips = [];
  if (p.dueAt) {
    chips.push(h('span', { class: 'chip date' }, icon(p.allDay ? 'calendar' : 'clock'),
      formatDue(p.dueAt, p.allDay)));
  }
  if (p.priority < 4) {
    chips.push(h('span', { class: `chip p${p.priority}` }, icon('flag'),
      { 1: 'Urgent', 2: 'High', 3: 'Medium' }[p.priority]));
  }
  for (const t of p.tags) chips.push(h('span', { class: 'chip' }, icon('tag'), t));
  if (p.projectName) chips.push(h('span', { class: 'chip list' }, icon('list'), p.projectName));
  host.replaceChildren(...chips);
}

function wireComposer() {
  const form = $('composer');
  const box = $('composerInput');
  const submit = $('composerSubmit');

  // The hint is worth the width on a laptop, not on a phone.
  const narrow = matchMedia('(max-width: 600px)');
  const setPlaceholder = () => {
    box.placeholder = narrow.matches
      ? 'Add a task…'
      : 'Add a task —  try “report friday 2pm !p1 #work”';
  };
  setPlaceholder();
  narrow.addEventListener('change', setPlaceholder);

  box.addEventListener('input', () => {
    submit.disabled = !box.value.trim();
    renderParseChips(box.value);
  });
  box.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Escape') { box.value = ''; submit.disabled = true; renderParseChips(''); box.blur(); }
    // Submit explicitly rather than relying on implicit form submission.
    if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const created = quickAdd(box.value);
    if (!created) return;
    box.value = '';
    submit.disabled = true;
    renderParseChips('');
    $('listArea').scrollTo({ top: 0, behavior: 'smooth' });
  });

  document.addEventListener('quickadd', (e) => {
    const created = quickAdd(e.detail);
    if (created) toast({ message: `Added “${created.title}”`,
      action: { label: 'Edit', onClick: () => openDetail(created.id) } });
  });
}

/* ── sort / filter menu ──────────────────────────────────────────────── */
function buildSortMenu() {
  const menu = $('sortMenu');
  const row = (label, checked, onClick) => h('button', {
    type: 'button', role: 'menuitemradio', 'aria-checked': String(checked),
    onclick: () => { onClick(); closeMenu(); },
  }, icon('check', 'tick'), label);

  menu.replaceChildren(
    h('div', { class: 'group-label' }, 'Sort by'),
    ...SORTS.map((s) => row(s.label, store.prefs.sort === s.id, () => S.savePrefs({ sort: s.id }))),
    h('hr'),
    h('button', {
      type: 'button', role: 'menuitemcheckbox',
      'aria-checked': String(store.prefs.showCompleted),
      onclick: () => { S.savePrefs({ showCompleted: !store.prefs.showCompleted }); closeMenu(); },
    }, icon('check', 'tick'), 'Show completed'),
    h('hr'),
    h('button', { type: 'button', role: 'menuitem',
      onclick: () => { closeMenu(); clearCompletedFlow(); } },
    icon('trash', 'tick'), 'Clear completed…'),
    h('button', { type: 'button', role: 'menuitem',
      onclick: () => { closeMenu(); newProjectDialog(); } },
    icon('plus', 'tick'), 'New list…'));
}

function openMenu() {
  buildSortMenu();
  $('sortMenu').hidden = false;
  $('sortBtn').setAttribute('aria-expanded', 'true');
  setTimeout(() => document.addEventListener('mousedown', outsideMenu), 0);
  document.addEventListener('keydown', escMenu, true);
}
function closeMenu() {
  $('sortMenu').hidden = true;
  $('sortBtn').setAttribute('aria-expanded', 'false');
  document.removeEventListener('mousedown', outsideMenu);
  document.removeEventListener('keydown', escMenu, true);
}
const outsideMenu = (e) => { if (!e.target.closest('.menu-wrap')) closeMenu(); };
const escMenu = (e) => { if (e.key === 'Escape') { e.stopPropagation(); closeMenu(); } };
const menuIsOpen = () => !$('sortMenu').hidden;

/* ── help ────────────────────────────────────────────────────────────── */
const SHORTCUTS = [
  ['Add a task', ['N']], ['Search / command palette', ['⌘', 'K']], ['Search tasks', ['/']],
  ['Move between tasks', ['J', 'K']], ['Complete / reopen', ['X']], ['Open details', ['Enter']],
  ['Rename in place', ['E']], ['Set priority', ['1', '–', '4']], ['Delete task', ['⌫']],
  ['Undo', ['⌘', 'Z']], ['Switch theme', ['T']], ['Close panel', ['Esc']],
];

function showHelp() {
  openModal({
    title: 'Keyboard shortcuts',
    lede: 'Everything here is reachable without the mouse.',
    body: h('div', { class: 'shortcut-grid' }, SHORTCUTS.map(([label, keys]) =>
      h('div', { class: 'shortcut-row' }, h('span', {}, label),
        h('span', { class: 'keys' }, keys.map((k) => h('kbd', {}, k)))))),
    actions: [{ label: 'Got it', variant: 'primary' }],
  });
}

/* ── export ──────────────────────────────────────────────────────────── */
async function exportJson() {
  try {
    const res = await fetch('/api/export');
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = h('a', { href: url, download: `tasks-${new Date().toISOString().slice(0, 10)}.json` });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast({ message: `Exported ${data.tasks.length} tasks` });
  } catch {
    toast({ message: 'Export failed — is the server still running?', tone: 'error' });
  }
}

/* ── keyboard ────────────────────────────────────────────────────────── */
const isTyping = (el) => el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);

function focusedTask() {
  const el = document.activeElement?.closest?.('.task');
  return el?.dataset.id || null;
}

function moveFocus(delta) {
  const ids = visibleIds;
  if (!ids.length) return;
  const current = focusedTask() || store.focusId;
  let i = ids.indexOf(current);
  i = i === -1 ? (delta > 0 ? 0 : ids.length - 1) : Math.min(ids.length - 1, Math.max(0, i + delta));
  const id = ids[i];
  store.focusId = id;
  const el = document.querySelector(`.task[data-id="${CSS.escape(id)}"]`);
  el?.focus();
  el?.scrollIntoView({ block: 'nearest' });
}

function onKeydown(e) {
  if (palette.isOpen() || modalOpen()) return;
  const mod = e.metaKey || e.ctrlKey;

  if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); palette.open(); return; }
  if (mod && e.key.toLowerCase() === 'z' && !e.shiftKey) {
    if (isTyping(e.target)) return;
    e.preventDefault();
    S.undo().then((label) => label && toast({ message: label }));
    return;
  }
  if (isTyping(e.target)) return;

  const id = focusedTask();
  switch (e.key) {
    case '/':
      e.preventDefault(); palette.open(); return;
    case 'n': case 'c':
      e.preventDefault(); $('composerInput').focus(); return;
    case 'j': case 'ArrowDown':
      e.preventDefault(); moveFocus(1); return;
    case 'k': case 'ArrowUp':
      e.preventDefault(); moveFocus(-1); return;
    case 'x': case ' ':
      if (id) { e.preventDefault(); S.toggleTask(id); } return;
    case 'Enter':
      if (id) { e.preventDefault(); openDetail(id); } return;
    case 'e':
      if (id) { e.preventDefault(); startInlineEdit(id); } return;
    case 'Backspace': case 'Delete':
      if (id) {
        e.preventDefault();
        const task = store.tasks.get(id);
        const next = visibleIds[Math.min(visibleIds.indexOf(id) + 1, visibleIds.length - 1)];
        S.deleteTask(id);
        toast({ message: `Deleted “${task.title}”`, action: { label: 'Undo', onClick: () => S.undo() } });
        setTimeout(() => document.querySelector(`.task[data-id="${next ? CSS.escape(next) : ''}"]`)?.focus(), 50);
      }
      return;
    case '1': case '2': case '3': case '4':
      if (id) { e.preventDefault(); S.updateTask(id, { priority: +e.key }, { undoable: true }); }
      return;
    case 't':
      e.preventDefault(); cycleTheme(); return;
    case '?':
      e.preventDefault(); showHelp(); return;
    case 'Escape':
      if (store.detailId) { closeDetail(); }
      else if (store.query) { store.query = ''; S.emit('query'); }
      else if (document.getElementById('app').classList.contains('nav-open')) toggleNav(false);
      return;
    default:
  }
}

/* ── mobile nav ──────────────────────────────────────────────────────── */
function toggleNav(open) {
  const app = $('app');
  const on = open ?? !app.classList.contains('nav-open');
  app.classList.toggle('nav-open', on);
  $('scrim').hidden = !on;
  $('menuBtn').setAttribute('aria-expanded', String(on));
  if (on) $('sidebar').querySelector('.nav-item')?.focus();
}

/* ── boot ────────────────────────────────────────────────────────────── */
function wireChrome() {
  $('menuBtn').onclick = () => toggleNav(true);
  $('sidebarClose').onclick = () => toggleNav(false);
  $('scrim').onclick = () => toggleNav(false);
  $('searchTrigger').onclick = () => palette.open();
  $('addListBtn').onclick = () => newProjectDialog();
  $('themeBtn').onclick = cycleTheme;
  $('exportBtn').onclick = exportJson;
  $('helpBtn').onclick = showHelp;
  $('sortBtn').onclick = () => (menuIsOpen() ? closeMenu() : openMenu());

  // On small screens, picking a view should close the drawer.
  $('sidebar').addEventListener('click', (e) => {
    if (e.target.closest('.nav-item, .tag-pill') && matchMedia('(max-width: 860px)').matches) {
      toggleNav(false);
    }
  });
}

function failScreen(message) {
  $('boot').replaceChildren(h('div', { style: 'text-align:center;max-width:34ch' },
    h('h2', { style: 'font-size:16px;margin-bottom:6px' }, 'Can’t reach the server'),
    h('p', { style: 'font-size:13.5px;color:var(--text-2);margin-bottom:14px' }, message),
    h('button', { class: 'btn primary', onclick: () => location.reload() }, 'Try again')));
}

async function main() {
  applyTheme();
  palette.init();
  palette.registerCommands([
    { label: 'New task', icon: 'plus', hint: 'N', run: () => $('composerInput').focus() },
    { label: 'New list', icon: 'list', run: () => newProjectDialog() },
    { label: 'Toggle completed tasks', icon: 'done',
      run: () => S.savePrefs({ showCompleted: !store.prefs.showCompleted }) },
    { label: 'Clear completed tasks', icon: 'trash', run: () => clearCompletedFlow() },
    { label: 'Switch theme', icon: 'sun', hint: 'T', run: cycleTheme },
    { label: 'Export as JSON', icon: 'notes', run: exportJson },
    { label: 'Keyboard shortcuts', icon: 'sparkle', hint: '?', run: showHelp },
  ]);
  wireChrome();
  wireComposer();

  S.setErrorHandler(({ message, tone }) =>
    toast({ message, tone: tone === 'error' ? 'error' : 'default' }));
  S.subscribe((reason) => {
    if (reason === 'detail') { renderDetail(); render(); } else render();
  });
  document.addEventListener('keydown', onKeydown);

  try {
    await S.load();
  } catch (err) {
    failScreen(err.message || 'The API did not respond.');
    return;
  }
  render();
  S.startLiveSync();

  $('app').hidden = false;
  $('boot').classList.add('done');
  setTimeout(() => $('boot').remove(), 320);

  // Re-render every minute so relative dates ("Today", "2h ago") stay honest.
  setInterval(() => S.emit('tick'), 60000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) S.emit('visible'); });
}

main();
