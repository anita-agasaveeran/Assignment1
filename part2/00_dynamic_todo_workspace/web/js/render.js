/* All DOM rendering. Reads the store, writes the page. */

import { h, icon, PROJECT_COLORS, escapeHtml } from './dom.js';
import { formatDue, dueState, formatStamp, toInputValue, fromInputValue, addDays, startOfDay, toISO } from './dates.js';
import { VIEWS, SORTS, buildGroups, counts, viewMeta, scopeCounts } from './views.js';
import * as S from './store.js';
import { store } from './store.js';
import { toast, confirmDialog, openModal } from './ui.js';

const $ = (id) => document.getElementById(id);
export let visibleIds = [];

/* ── sidebar ─────────────────────────────────────────────────────────── */
function navItem({ id, label, iconName, color, count, active, danger, onClick, onDrop, actions }) {
  const el = h('button', {
    class: `nav-item ${active ? 'active' : ''} ${danger ? 'overdue' : ''}`,
    type: 'button', 'aria-current': active ? 'page' : null, onclick: onClick,
  },
  color
    ? h('span', { class: 'swatch', style: `color:${PROJECT_COLORS[color] || color}` })
    : icon(iconName),
  h('span', { class: 'label' }, label),
  count ? h('span', { class: 'count' }, String(count)) : null);

  if (onDrop) {
    el.addEventListener('dragover', (e) => {
      if (!e.dataTransfer.types.includes('text/task-id')) return;
      e.preventDefault();
      el.classList.add('drop-target');
    });
    el.addEventListener('dragleave', () => el.classList.remove('drop-target'));
    el.addEventListener('drop', (e) => {
      e.preventDefault();
      el.classList.remove('drop-target');
      onDrop(e.dataTransfer.getData('text/task-id'));
    });
  }
  if (actions) el.addEventListener('contextmenu', (e) => { e.preventDefault(); actions(e); });
  return el;
}

export function renderSidebar() {
  const tasks = [...store.tasks.values()];
  const c = counts(tasks, store.projects);

  $('viewNav').replaceChildren(...VIEWS.map((v) => navItem({
    label: v.label, iconName: v.icon, count: c[v.id],
    active: store.prefs.view === v.id,
    danger: v.id === 'today' && c.overdue > 0,
    onClick: () => S.savePrefs({ view: v.id, projectId: null }),
    onDrop: v.id === 'inbox' ? (id) => moveToProject(id, null) : null,
  })));

  const projectNodes = store.projects.map((p) => navItem({
    label: p.name, color: p.color, count: c.byProject[p.id],
    active: store.prefs.view === 'project' && store.prefs.projectId === p.id,
    onClick: () => S.savePrefs({ view: 'project', projectId: p.id }),
    onDrop: (id) => moveToProject(id, p.id),
    actions: () => editProjectDialog(p),
  }));
  if (!projectNodes.length) {
    projectNodes.push(h('p', {
      style: 'font-size:12.5px;color:var(--text-3);padding:2px 10px 6px',
    }, 'No lists yet — press + to add one.'));
  }
  $('projectNav').replaceChildren(...projectNodes);

  const tagNames = store.tags.filter((t) => c.byTag[t]);
  $('tagSection').hidden = tagNames.length === 0;
  $('tagCloud').replaceChildren(...tagNames.map((t) => h('button', {
    class: `tag-pill ${store.prefs.tag === t ? 'active' : ''}`, type: 'button',
    onclick: () => S.savePrefs({ tag: store.prefs.tag === t ? null : t }),
  }, `#${t}`, ' ', h('span', { style: 'opacity:.6' }, String(c.byTag[t])))));

  // Queued writes outrank the stream's own status: if something is waiting to
  // sync, say so, whatever the event stream thinks.
  const queued = S.queuedWrites();
  const status = queued ? 'offline' : store.connection;
  const label = queued
    ? `${queued} change${queued === 1 ? '' : 's'} waiting to sync`
    : { live: 'Live sync on', connecting: 'Connecting…', reconnecting: 'Reconnecting…',
      offline: 'Offline — changes queued' }[status] || status;
  const conn = $('conn');
  conn.className = `conn ${status === 'live' ? 'live' : ''} ${status === 'offline' ? 'offline' : ''}`;
  conn.title = queued ? 'Changes are saved locally and will sync automatically' : 'Live connection status';
  $('connText').textContent = label;
}

function moveToProject(taskId, projectId) {
  const task = store.tasks.get(taskId);
  if (!task || task.projectId === projectId) return;
  S.updateTask(taskId, { projectId }, { undoable: true });
  const name = projectId ? S.projectById(projectId)?.name : 'Inbox';
  toast({ message: `Moved to ${name}`, action: { label: 'Undo', onClick: () => S.undo() } });
}

/* ── header ──────────────────────────────────────────────────────────── */
export function renderHeader() {
  const meta = viewMeta(store.prefs, store.projects);
  const tasks = [...store.tasks.values()];
  const { total } = buildGroups(tasks, store.prefs, store.query);

  $('viewTitle').textContent = meta.title;

  const { open: openInScope, done } = scopeCounts(tasks, store.prefs);
  const openCount = total;
  const denom = openInScope + done;
  const pct = denom ? Math.round((done / denom) * 100) : 0;

  const subtitle = store.prefs.view === 'completed'
    ? `${total} completed`
    : openCount === 0 ? 'Nothing left here' :
      `${openCount} task${openCount === 1 ? '' : 's'}` +
      (store.query ? ` matching “${store.query}”` : '');
  $('viewSubtitle').textContent = subtitle;

  const bar = $('progressBar');
  const circ = 2 * Math.PI * 15.5;
  bar.style.strokeDasharray = circ;
  bar.style.strokeDashoffset = circ * (1 - pct / 100);
  $('progressText').textContent = `${pct}%`;
  $('progress').classList.toggle('complete', pct === 100 && denom > 0);
  $('progress').setAttribute('aria-label', `${pct}% complete`);

  renderFilterChips();
}

function renderFilterChips() {
  const chips = [];
  if (store.query) {
    chips.push(chip(`Search: ${store.query}`, () => { store.query = ''; S.emit('query'); }));
  }
  if (store.prefs.tag) chips.push(chip(`#${store.prefs.tag}`, () => S.savePrefs({ tag: null })));
  if (store.prefs.sort !== 'manual') {
    chips.push(chip(`Sorted by ${SORTS.find((s) => s.id === store.prefs.sort).label.toLowerCase()}`,
      () => S.savePrefs({ sort: 'manual' })));
  }
  if (store.prefs.showCompleted && store.prefs.view !== 'completed') {
    chips.push(chip('Showing completed', () => S.savePrefs({ showCompleted: false })));
  }
  $('filterChips').replaceChildren(...chips);
}

const chip = (label, onClear) => h('span', { class: 'filter-chip' }, label,
  h('button', { type: 'button', 'aria-label': `Clear ${label}`, onclick: onClear }, icon('close')));

/* ── task rows ───────────────────────────────────────────────────────── */
function taskEl(task) {
  const li = h('li', {
    class: `task p${task.priority} ${task.done ? 'done' : ''} ${store.detailId === task.id ? 'selected' : ''}`,
    dataset: { id: task.id }, tabindex: '0', role: 'listitem',
    'aria-label': task.title,
  });

  const grip = h('span', { class: 'grip', html: icon('drag').outerHTML, title: 'Drag to reorder',
    'aria-hidden': 'true' });
  grip.addEventListener('pointerdown', () => { li.draggable = true; });
  grip.addEventListener('pointerup', () => { li.draggable = false; });

  const check = h('button', {
    class: 'check', type: 'button', role: 'checkbox',
    'aria-checked': String(task.done), 'aria-label': `Mark “${task.title}” ${task.done ? 'not done' : 'done'}`,
    onclick: (e) => { e.stopPropagation(); onToggle(task.id); },
  }, icon('check'));

  const title = h('div', { class: 'task-title' }, task.title);
  const body = h('div', {
    class: 'task-body',
    onclick: () => openDetail(task.id),
    ondblclick: (e) => { e.stopPropagation(); startInlineEdit(task.id); },
  }, title, metaRow(task));

  const actions = h('div', { class: 'task-actions' },
    h('button', { class: 'icon-btn xs', type: 'button', title: 'Edit (E)',
      'aria-label': `Edit ${task.title}`,
      onclick: (e) => { e.stopPropagation(); openDetail(task.id); } }, icon('edit')),
    h('button', { class: 'icon-btn xs', type: 'button', title: 'Delete (⌫)',
      'aria-label': `Delete ${task.title}`,
      onclick: (e) => { e.stopPropagation(); onDelete(task.id); } }, icon('trash')));

  li.append(grip, check, body, actions);
  if (task._pending) li.style.opacity = '.7';
  return li;
}

function metaRow(task) {
  const bits = [];
  if (task.dueAt) {
    const state = dueState(task.dueAt, task.done);
    bits.push(h('span', { class: `meta due-${state}` }, icon(task.allDay ? 'calendar' : 'clock'),
      formatDue(task.dueAt, task.allDay)));
  }
  if (task.priority < 4 && !task.done) {
    const names = { 1: 'Urgent', 2: 'High', 3: 'Medium' };
    const colors = { 1: 'var(--danger)', 2: 'var(--amber-500)', 3: 'var(--blue-500)' };
    bits.push(h('span', { class: 'meta', style: `color:${colors[task.priority]}` },
      icon('flag'), names[task.priority]));
  }
  if (task.projectId && store.prefs.view !== 'project') {
    const p = S.projectById(task.projectId);
    if (p) {
      bits.push(h('span', { class: 'meta' },
        h('span', { class: 'swatch', style: `color:${PROJECT_COLORS[p.color] || p.color};width:8px;height:8px` }),
        p.name));
    }
  }
  for (const tag of task.tags) {
    bits.push(h('button', {
      class: 'meta tag', type: 'button', title: `Filter by #${tag}`,
      onclick: (e) => { e.stopPropagation(); S.savePrefs({ tag, view: store.prefs.view }); },
    }, `#${tag}`));
  }
  if (task.notes.trim()) bits.push(h('span', { class: 'meta notes', title: 'Has notes' }, icon('notes')));
  return h('div', { class: 'task-meta' }, bits);
}

function onToggle(id) {
  const task = store.tasks.get(id);
  const nowDone = S.toggleTask(id);
  if (nowDone) {
    const remaining = buildGroups([...store.tasks.values()], store.prefs, store.query).total;
    toast({
      message: remaining === 0 ? 'That’s everything. Nice work. 🎉' : `Completed “${trim(task.title)}”`,
      action: { label: 'Undo', onClick: () => S.undo() },
      duration: 5000,
    });
  }
}

function onDelete(id) {
  const task = store.tasks.get(id);
  if (!task) return;
  S.deleteTask(id);
  toast({ message: `Deleted “${trim(task.title)}”`, action: { label: 'Undo', onClick: () => S.undo() } });
}

const trim = (s, n = 38) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

export function startInlineEdit(id) {
  const li = document.querySelector(`.task[data-id="${CSS.escape(id)}"]`);
  const task = store.tasks.get(id);
  if (!li || !task) return;
  const holder = li.querySelector('.task-title');
  const input = h('input', { type: 'text', value: task.title, 'aria-label': 'Edit title',
    maxlength: '500' });
  holder.replaceChildren(input);
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);

  let settled = false;
  const commit = (save) => {
    if (settled) return;
    settled = true;
    const value = input.value.trim();
    if (save && value && value !== task.title) S.updateTask(id, { title: value }, { undoable: true });
    else S.emit('cancel-edit');
  };
  input.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); commit(true); }
    if (e.key === 'Escape') { e.preventDefault(); commit(false); }
  });
  input.addEventListener('blur', () => commit(true));
}

/* ── drag & drop reordering ──────────────────────────────────────────── */
let dragId = null;

function wireDrag(list) {
  list.addEventListener('dragstart', (e) => {
    const li = e.target.closest('.task');
    if (!li) return;
    dragId = li.dataset.id;
    li.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/task-id', dragId);
    e.dataTransfer.setData('text/plain', store.tasks.get(dragId)?.title || '');
  });

  list.addEventListener('dragover', (e) => {
    if (!dragId) return;
    e.preventDefault();
    const li = e.target.closest('.task');
    list.querySelectorAll('.drop-before,.drop-after')
      .forEach((n) => n.classList.remove('drop-before', 'drop-after'));
    if (!li || li.dataset.id === dragId) return;
    const rect = li.getBoundingClientRect();
    li.classList.add(e.clientY < rect.top + rect.height / 2 ? 'drop-before' : 'drop-after');
  });

  list.addEventListener('drop', (e) => {
    if (!dragId) return;
    e.preventDefault();
    const target = list.querySelector('.drop-before, .drop-after');
    list.querySelectorAll('.dragging').forEach((n) => n.classList.remove('dragging'));
    if (!target) return;
    const before = target.classList.contains('drop-before');
    target.classList.remove('drop-before', 'drop-after');

    if (store.prefs.sort !== 'manual') {
      toast({ message: 'Switch to “My order” to reorder tasks by hand.',
        action: { label: 'Switch', onClick: () => S.savePrefs({ sort: 'manual' }) } });
      dragId = null;
      return;
    }
    const ids = visibleIds.filter((id) => id !== dragId);
    const at = ids.indexOf(target.dataset.id);
    ids.splice(before ? at : at + 1, 0, dragId);
    S.reorderTasks(ids);
    dragId = null;
  });

  list.addEventListener('dragend', () => {
    list.querySelectorAll('.dragging,.drop-before,.drop-after')
      .forEach((n) => n.classList.remove('dragging', 'drop-before', 'drop-after'));
    list.querySelectorAll('.task').forEach((n) => { n.draggable = false; });
    dragId = null;
  });
}

/* ── empty states ────────────────────────────────────────────────────── */
const ART = {
  clear: `<svg viewBox="0 0 120 96" fill="none"><ellipse cx="60" cy="83" rx="38" ry="6" fill="currentColor" opacity=".1"/><rect x="30" y="20" width="60" height="58" rx="9" fill="currentColor" opacity=".08"/><path d="M44 47l10 10 22-24" stroke="var(--accent)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="96" cy="24" r="4" fill="var(--accent)" opacity=".5"/><circle cx="24" cy="34" r="2.6" fill="var(--accent)" opacity=".35"/></svg>`,
  empty: `<svg viewBox="0 0 120 96" fill="none"><ellipse cx="60" cy="83" rx="38" ry="6" fill="currentColor" opacity=".1"/><rect x="28" y="18" width="64" height="60" rx="9" fill="currentColor" opacity=".08"/><rect x="40" y="34" width="40" height="5" rx="2.5" fill="currentColor" opacity=".28"/><rect x="40" y="47" width="30" height="5" rx="2.5" fill="currentColor" opacity=".2"/><rect x="40" y="60" width="22" height="5" rx="2.5" fill="currentColor" opacity=".14"/></svg>`,
  search: `<svg viewBox="0 0 120 96" fill="none"><ellipse cx="60" cy="83" rx="34" ry="6" fill="currentColor" opacity=".1"/><circle cx="55" cy="42" r="22" stroke="currentColor" stroke-width="5" opacity=".22"/><path d="M71 58l14 14" stroke="var(--accent)" stroke-width="6" stroke-linecap="round"/></svg>`,
};

function emptyState() {
  const { view } = store.prefs;
  if (store.query) {
    return { art: ART.search, title: 'No matches', body: `Nothing matches “${store.query}”.`,
      action: { label: 'Clear search', onClick: () => { store.query = ''; S.emit('query'); } } };
  }
  if (store.prefs.tag) {
    return { art: ART.search, title: `Nothing tagged #${store.prefs.tag}`,
      body: 'Try clearing the tag filter.',
      action: { label: 'Clear filter', onClick: () => S.savePrefs({ tag: null }) } };
  }
  const byView = {
    today: { art: ART.clear, title: 'Today is clear', body: 'Nothing is due today. Enjoy the quiet — or pull something forward.' },
    upcoming: { art: ART.empty, title: 'Nothing scheduled', body: 'Add a due date to a task and it will show up here.' },
    inbox: { art: ART.clear, title: 'Inbox zero', body: 'Everything is filed into a list. Capture new thoughts above.' },
    all: { art: ART.empty, title: 'No tasks yet', body: 'Add your first one above — try “draft the brief tomorrow 9am !p2 #work”.' },
    completed: { art: ART.empty, title: 'Nothing completed yet', body: 'Finished tasks collect here.' },
    project: { art: ART.empty, title: 'This list is empty', body: 'Add a task above to get started.' },
  };
  return byView[view] || byView.all;
}

/* ── list ────────────────────────────────────────────────────────────── */
export function renderList() {
  const list = $('taskList');
  const area = $('listArea');
  area.setAttribute('aria-busy', 'false');

  if (!list.dataset.wired) { wireDrag(list); list.dataset.wired = '1'; }

  const { groups } = buildGroups([...store.tasks.values()], store.prefs, store.query);
  visibleIds = groups.flatMap((g) => g.tasks.map((t) => t.id));

  const nodes = [];
  for (const g of groups) {
    if (g.label) {
      nodes.push(h('li', { class: `group-head ${g.tone || ''}`, role: 'presentation' },
        g.label, h('span', { class: 'n' }, String(g.tasks.length)), h('span', { class: 'rule' }),
        g.key === 'done'
          ? h('button', { class: 'btn ghost sm', type: 'button', onclick: clearCompletedFlow },
            'Clear')
          : null));
    }
    g.tasks.forEach((t) => nodes.push(taskEl(t)));
  }

  const previous = new Set([...list.children].map((n) => n.dataset?.id).filter(Boolean));
  list.replaceChildren(...nodes);
  for (const n of list.children) {
    if (n.dataset?.id && !previous.has(n.dataset.id) && previous.size) n.classList.add('entering');
  }

  const isEmpty = visibleIds.length === 0;
  const empty = $('empty');
  empty.hidden = !isEmpty;
  if (isEmpty) {
    const e = emptyState();
    empty.replaceChildren(
      h('div', { html: e.art, 'aria-hidden': 'true' }).firstElementChild,
      h('h3', {}, e.title),
      h('p', {}, e.body),
      e.action ? h('button', { class: 'btn sm', type: 'button', onclick: e.action.onClick },
        e.action.label) : null);
  }
}

export function clearCompletedFlow() {
  const scope = store.prefs.view === 'project' ? store.prefs.projectId : null;
  const n = [...store.tasks.values()].filter((t) => t.done && (!scope || t.projectId === scope)).length;
  if (!n) { toast({ message: 'No completed tasks to clear.' }); return; }
  confirmDialog({
    title: `Clear ${n} completed task${n === 1 ? '' : 's'}?`,
    lede: 'They move to the trash. You can undo right after.',
    confirmLabel: 'Clear them',
    onConfirm: () => {
      S.clearCompleted(scope);
      toast({ message: `Cleared ${n} task${n === 1 ? '' : 's'}`,
        action: { label: 'Undo', onClick: () => S.undo() } });
    },
  });
}

/* ── detail drawer ───────────────────────────────────────────────────── */
export function openDetail(id) {
  store.detailId = id;
  S.emit('detail');
  setTimeout(() => $('detail').querySelector('.detail-title')?.focus(), 0);
}

export function closeDetail() {
  const id = store.detailId;
  store.detailId = null;
  S.emit('detail');
  document.querySelector(`.task[data-id="${id ? CSS.escape(id) : ''}"]`)?.focus();
}

export function renderDetail() {
  const panel = $('detail');
  const task = store.detailId ? store.tasks.get(store.detailId) : null;
  document.getElementById('app').classList.toggle('detail-open', !!task);
  panel.hidden = !task;
  if (!task) { panel.replaceChildren(); return; }

  const set = (patch, opts) => S.updateTask(task.id, patch, opts);

  const titleBox = h('textarea', { class: 'detail-title', rows: '1', 'aria-label': 'Title',
    maxlength: '500' });
  titleBox.value = task.title;
  const autosize = () => { titleBox.style.height = 'auto'; titleBox.style.height = titleBox.scrollHeight + 'px'; };
  titleBox.addEventListener('input', autosize);
  titleBox.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') { e.preventDefault(); titleBox.blur(); }
    if (e.key === 'Escape') { titleBox.value = task.title; titleBox.blur(); }
  });
  titleBox.addEventListener('blur', () => {
    const v = titleBox.value.trim();
    if (v && v !== task.title) set({ title: v }, { undoable: true });
    else titleBox.value = task.title;
  });

  /* due date */
  const dueInput = h('input', {
    type: task.allDay ? 'date' : 'datetime-local',
    value: toInputValue(task.dueAt, task.allDay), 'aria-label': 'Due date',
    onchange: (e) => {
      const parsed = fromInputValue(e.target.value);
      set(parsed ? { dueAt: parsed.dueAt, allDay: task.allDay } : { dueAt: null });
    },
  });
  dueInput.addEventListener('keydown', (e) => e.stopPropagation());

  const quick = (label, dateFn) => h('button', { class: 'btn ghost sm', type: 'button',
    onclick: () => set({ dueAt: dateFn(), allDay: true }) }, label);

  const dueField = h('div', { class: 'field' },
    h('label', {}, 'Due'),
    h('div', { class: 'control' }, icon('calendar'), dueInput,
      task.dueAt ? h('button', { class: 'icon-btn xs', type: 'button', 'aria-label': 'Clear due date',
        onclick: () => set({ dueAt: null, allDay: true }) }, icon('close')) : null),
    h('div', { style: 'display:flex;gap:4px;flex-wrap:wrap' },
      quick('Today', () => toISO(startOfDay())),
      quick('Tomorrow', () => toISO(addDays(startOfDay(), 1))),
      quick('Next week', () => toISO(addDays(startOfDay(), 7))),
      h('button', {
        class: 'btn ghost sm', type: 'button', 'aria-pressed': String(!task.allDay),
        onclick: () => {
          const base = task.dueAt ? new Date(task.dueAt) : startOfDay();
          if (task.allDay) base.setHours(9, 0, 0, 0);
          else base.setHours(0, 0, 0, 0);
          set({ allDay: !task.allDay, dueAt: toISO(base) });
        },
      }, task.allDay ? 'Add time' : 'All-day')));

  /* priority */
  const priorityField = h('div', { class: 'field' }, h('label', {}, 'Priority'),
    h('div', { class: 'seg', role: 'group', 'aria-label': 'Priority' },
      [[1, 'Urgent'], [2, 'High'], [3, 'Medium'], [4, 'None']].map(([p, label]) =>
        h('button', { type: 'button', dataset: { p: String(p) },
          'aria-pressed': String(task.priority === p),
          onclick: () => set({ priority: p }, { undoable: true }) }, label))));

  /* list */
  const select = h('select', { 'aria-label': 'List',
    onchange: (e) => set({ projectId: e.target.value || null }, { undoable: true }) },
  h('option', { value: '' }, 'Inbox (no list)'),
  store.projects.map((p) => h('option', { value: p.id, selected: p.id === task.projectId }, p.name)));
  const listField = h('div', { class: 'field' }, h('label', {}, 'List'),
    h('div', { class: 'control' }, icon('list'), select));

  /* tags */
  const tagInput = h('input', { type: 'text', placeholder: task.tags.length ? 'Add tag…' : 'work, home…',
    'aria-label': 'Add tag', list: 'tagOptions' });
  tagInput.addEventListener('keydown', (e) => {
    e.stopPropagation();
    const value = tagInput.value.trim().replace(/^#/, '');
    if ((e.key === 'Enter' || e.key === ',') && value) {
      e.preventDefault();
      if (!task.tags.some((t) => t.toLowerCase() === value.toLowerCase())) {
        set({ tags: [...task.tags, value] });
      }
      tagInput.value = '';
    } else if (e.key === 'Backspace' && !tagInput.value && task.tags.length) {
      set({ tags: task.tags.slice(0, -1) });
    }
  });
  tagInput.addEventListener('blur', () => {
    const value = tagInput.value.trim().replace(/^#/, '');
    if (value && !task.tags.some((t) => t.toLowerCase() === value.toLowerCase())) {
      set({ tags: [...task.tags, value] });
    }
    tagInput.value = '';
  });
  const tagField = h('div', { class: 'field' }, h('label', {}, 'Tags'),
    h('div', { class: 'tag-editor', onclick: (e) => { if (e.target.classList.contains('tag-editor')) tagInput.focus(); } },
      task.tags.map((t) => h('span', { class: 'chip' }, `#${t}`,
        h('button', { type: 'button', 'aria-label': `Remove ${t}`,
          onclick: () => set({ tags: task.tags.filter((x) => x !== t) }) }, icon('close')))),
      tagInput),
    h('datalist', { id: 'tagOptions' }, store.tags.map((t) => h('option', { value: t }))));

  /* notes */
  const notes = h('textarea', { placeholder: 'Add notes, links, next steps…', 'aria-label': 'Notes' });
  notes.value = task.notes;
  notes.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Escape') notes.blur();
  });
  let notesTimer;
  notes.addEventListener('input', () => {
    clearTimeout(notesTimer);
    notesTimer = setTimeout(() => set({ notes: notes.value }), 600);
  });
  notes.addEventListener('blur', () => { clearTimeout(notesTimer); if (notes.value !== task.notes) set({ notes: notes.value }); });
  const notesField = h('div', { class: 'field' }, h('label', {}, 'Notes'),
    h('div', { class: 'control' }, notes));

  panel.replaceChildren(
    h('div', { class: 'detail-head' },
      h('h2', {}, 'Task'),
      h('div', { style: 'display:flex;gap:2px' },
        h('button', { class: 'btn ghost sm', type: 'button',
          onclick: () => { S.toggleTask(task.id); } },
        task.done ? 'Reopen' : 'Mark done'),
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Close details',
          onclick: closeDetail }, icon('close')))),
    h('div', { class: 'detail-body' }, titleBox, dueField, priorityField, listField, tagField, notesField),
    h('div', { class: 'detail-foot' },
      h('span', { class: 'stamp' }, `Added ${formatStamp(task.createdAt)}` +
        (task.completedAt ? ` · done ${formatStamp(task.completedAt)}` : '')),
      h('button', { class: 'btn danger sm', type: 'button',
        onclick: () => { onDelete(task.id); } }, icon('trash'), 'Delete')));
  autosize();
}

/* ── project dialogs ─────────────────────────────────────────────────── */
export function newProjectDialog() {
  let color = 'violet';
  const input = h('input', { type: 'text', placeholder: 'e.g. Home renovation', maxlength: '80',
    'aria-label': 'List name' });
  const swatches = h('div', { class: 'color-grid' },
    Object.entries(PROJECT_COLORS).map(([name, hex]) => h('button', {
      class: 'color-dot', type: 'button', 'aria-label': name, 'aria-pressed': String(name === color),
      style: `background:${hex}`,
      onclick: (e) => {
        color = name;
        swatches.querySelectorAll('.color-dot').forEach((b) => b.setAttribute('aria-pressed', 'false'));
        e.currentTarget.setAttribute('aria-pressed', 'true');
      },
    })));

  const form = h('form', { class: 'field', onsubmit: (e) => { e.preventDefault(); submit(); } },
    h('div', { class: 'control' }, input), h('div', { style: 'height:6px' }), swatches);
  input.addEventListener('keydown', (e) => e.stopPropagation());

  let close;
  function submit() {
    const name = input.value.trim();
    if (!name) { input.focus(); return; }
    S.createProject({ name, color });
    close?.();
    toast({ message: `Created “${name}”` });
  }
  close = openModal({
    title: 'New list', lede: 'Group related tasks together.', body: form,
    actions: [{ label: 'Cancel', variant: 'ghost' },
      { label: 'Create list', variant: 'primary', close: false, onClick: submit }],
  });
}

export function editProjectDialog(project) {
  const input = h('input', { type: 'text', value: project.name, maxlength: '80', 'aria-label': 'List name' });
  input.addEventListener('keydown', (e) => e.stopPropagation());
  let color = project.color;
  const swatches = h('div', { class: 'color-grid' },
    Object.entries(PROJECT_COLORS).map(([name, hex]) => h('button', {
      class: 'color-dot', type: 'button', 'aria-label': name, 'aria-pressed': String(name === color),
      style: `background:${hex}`,
      onclick: (e) => {
        color = name;
        swatches.querySelectorAll('.color-dot').forEach((b) => b.setAttribute('aria-pressed', 'false'));
        e.currentTarget.setAttribute('aria-pressed', 'true');
      },
    })));
  const body = h('div', { class: 'field' }, h('div', { class: 'control' }, input),
    h('div', { style: 'height:6px' }), swatches);

  const n = [...store.tasks.values()].filter((t) => t.projectId === project.id).length;
  openModal({
    title: 'Edit list', body,
    actions: [
      { label: 'Delete', variant: 'danger', onClick: () => confirmDialog({
        title: `Delete “${project.name}”?`,
        lede: n ? `${n} task${n === 1 ? '' : 's'} in this list will be deleted too.`
          : 'This list is empty.',
        confirmLabel: 'Delete list',
        onConfirm: () => { S.deleteProject(project.id); toast({ message: `Deleted “${project.name}”` }); },
      }) },
      { label: 'Cancel', variant: 'ghost' },
      { label: 'Save', variant: 'primary', onClick: () => {
        const name = input.value.trim() || project.name;
        S.updateProject(project.id, { name, color });
      } },
    ],
  });
}

/* ── orchestration ───────────────────────────────────────────────────── */
export function render() {
  renderSidebar();
  renderHeader();
  renderList();
  renderDetail();
}
