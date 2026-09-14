/* Application state.
 *
 * Every mutation is optimistic: the UI updates immediately, the request goes
 * out behind it, and a failure either rolls back (a real error) or parks the
 * change in an outbox and retries (offline). Undo is a stack of inverse ops.
 */

import { api, ApiError, connectEvents } from './api.js';

const PREFS_KEY = 'todo.prefs.v1';

const defaultPrefs = {
  view: 'today', projectId: null, tag: null, sort: 'manual',
  showCompleted: false, theme: 'system',
};

function loadPrefs() {
  try { return { ...defaultPrefs, ...JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') }; }
  catch { return { ...defaultPrefs }; }
}

export const store = {
  tasks: new Map(),
  projects: [],
  tags: [],
  prefs: loadPrefs(),
  query: '',
  detailId: null,
  focusId: null,
  connection: 'connecting',
  pending: 0,
  ready: false,
  _listeners: new Set(),
  _undo: [],
  _outbox: [],
  _retryTimer: null,
  _flushing: false,
};

/* ── subscriptions ───────────────────────────────────────────────────── */
export function subscribe(fn) { store._listeners.add(fn); return () => store._listeners.delete(fn); }

let frame = null;
export function emit(reason = 'change') {
  if (frame) return;
  const run = () => { frame = null; store._listeners.forEach((fn) => fn(reason)); };
  // requestAnimationFrame is paused in a background tab, which would strand
  // live updates from other clients until the tab is looked at again.
  frame = document.hidden ? setTimeout(run, 16) : requestAnimationFrame(run);
}

export function savePrefs(patch) {
  Object.assign(store.prefs, patch);
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(store.prefs)); } catch { /* private mode */ }
  emit('prefs');
}

/* ── error reporting (wired to toasts in app.js) ─────────────────────── */
let onError = () => {};
export function setErrorHandler(fn) { onError = fn; }

/* ── boot ────────────────────────────────────────────────────────────── */
export async function load() {
  const snap = await api.state();
  store.tasks = new Map(snap.tasks.map((t) => [t.id, t]));
  store.projects = snap.projects;
  store.tags = snap.tags;
  store.ready = true;
  emit('load');
}

export function startLiveSync() {
  connectEvents({
    onStatus(status) {
      store.connection = status;
      if (status === 'live') flushOutbox();
      emit('connection');
    },
    onEvent(msg) {
      applyRemote(msg.type, msg.data);
    },
    // Missed events: re-fetch the snapshot instead of drifting out of sync.
    onGap() {
      load().catch(() => {});
    },
  });
}

function applyRemote(type, data) {
  switch (type) {
    case 'task.created':
    case 'task.updated':
      store.tasks.set(data.id, data);
      break;
    case 'tasks.deleted':
      data.ids.forEach((id) => store.tasks.delete(id));
      if (data.ids.includes(store.detailId)) store.detailId = null;
      break;
    case 'tasks.reordered':
      data.ids.forEach((id, i) => {
        const t = store.tasks.get(id);
        if (t) t.position = i * 1024;
      });
      break;
    case 'project.created':
    case 'project.updated': {
      const i = store.projects.findIndex((p) => p.id === data.id);
      if (i === -1) store.projects.push(data); else store.projects[i] = data;
      store.projects.sort((a, b) => a.position - b.position);
      break;
    }
    case 'project.deleted':
      store.projects = store.projects.filter((p) => p.id !== data.id);
      for (const [id, t] of store.tasks) if (t.projectId === data.id) store.tasks.delete(id);
      if (store.prefs.projectId === data.id) savePrefs({ view: 'all', projectId: null });
      break;
  }
  refreshTags();
  emit('remote');
}

function refreshTags() {
  const seen = new Map();
  for (const t of store.tasks.values()) for (const tag of t.tags) {
    if (!seen.has(tag.toLowerCase())) seen.set(tag.toLowerCase(), tag);
  }
  store.tags = [...seen.values()].sort((a, b) => a.localeCompare(b));
}

/* ── request plumbing: optimistic + outbox ───────────────────────────── */
async function send(op) {
  store.pending++;
  emit('pending');
  try {
    const result = await op.run();
    op.commit?.(result);
    return result;
  } catch (err) {
    if (err instanceof ApiError && err.offline) {
      store._outbox.push(op);
      store.connection = 'offline';
      scheduleRetry();
      onError({ message: 'Offline — your change is saved and will sync.', tone: 'info' });
      return null;
    }
    op.rollback?.();
    onError({ message: err.message || 'That change could not be saved.', tone: 'error' });
    emit('error');
    return null;
  } finally {
    store.pending--;
    emit('pending');
  }
}

function scheduleRetry() {
  if (store._retryTimer) return;
  store._retryTimer = setTimeout(() => { store._retryTimer = null; flushOutbox(); }, 4000);
}

export async function flushOutbox() {
  if (!store._outbox.length || store._flushing) return;
  store._flushing = true;
  const queue = store._outbox.splice(0);
  const stillQueued = [];
  let offline = false;
  try {
    for (const op of queue) {
      if (offline) { stillQueued.push(op); continue; }   // keep the rest, in order
      try {
        op.commit?.(await op.run());
      } catch (err) {
        if (err instanceof ApiError && err.offline) { offline = true; stillQueued.push(op); }
        else op.rollback?.();
      }
    }
  } finally {
    store._flushing = false;
  }
  if (stillQueued.length) {
    store._outbox.unshift(...stillQueued);
    scheduleRetry();
    emit('sync');
    return;
  }
  await load().catch(() => {});
  onError({ message: 'Back online — everything is synced.', tone: 'info' });
  emit('sync');
}

addEventListener('online', () => flushOutbox());

/* ── undo ────────────────────────────────────────────────────────────── */
export function pushUndo(entry) {
  store._undo.push(entry);
  if (store._undo.length > 50) store._undo.shift();
}

export async function undo() {
  const entry = store._undo.pop();
  if (!entry) return null;
  await entry.run();
  return entry.label;
}

export const canUndo = () => store._undo.length > 0;

/** Writes waiting for the network. Drives the connection pill. */
export const queuedWrites = () => store._outbox.length;

/* ── task mutations ──────────────────────────────────────────────────── */
let tempSeq = 0;

export function createTask(data) {
  const tempId = `tmp_${++tempSeq}`;
  const min = Math.min(0, ...[...store.tasks.values()].map((t) => t.position));
  const optimistic = {
    id: tempId, projectId: data.projectId ?? null, title: data.title, notes: data.notes || '',
    done: false, priority: data.priority || 4, dueAt: data.dueAt || null,
    allDay: data.allDay !== false, position: min - 1024, tags: data.tags || [],
    createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), completedAt: null,
    _pending: true,
  };
  store.tasks.set(tempId, optimistic);
  refreshTags();
  emit('create');

  send({
    run: () => api.createTask(data),
    commit: (task) => {
      store.tasks.delete(tempId);
      if (task) store.tasks.set(task.id, task);
      if (store.detailId === tempId) store.detailId = task?.id ?? null;
      refreshTags();
      emit('create');
    },
    rollback: () => { store.tasks.delete(tempId); refreshTags(); emit('create'); },
  });
  return optimistic;
}

export function updateTask(id, patch, { undoable = false, label = 'Task updated' } = {}) {
  const before = store.tasks.get(id);
  if (!before) return;
  const snapshot = { ...before, tags: [...before.tags] };
  const next = { ...before, ...patch, updatedAt: new Date().toISOString() };
  if ('done' in patch) next.completedAt = patch.done ? new Date().toISOString() : null;
  store.tasks.set(id, next);
  refreshTags();
  emit('update');

  if (undoable) {
    const inverse = {};
    for (const key of Object.keys(patch)) inverse[key] = snapshot[key];
    pushUndo({ label, run: () => updateTask(id, inverse) });
  }

  if (id.startsWith('tmp_')) return next;      // create is still in flight
  send({
    run: () => api.updateTask(id, patch),
    commit: (task) => { if (task) { store.tasks.set(task.id, task); refreshTags(); emit('update'); } },
    rollback: () => { store.tasks.set(id, snapshot); refreshTags(); emit('update'); },
  });
  return next;
}

export function toggleTask(id) {
  const task = store.tasks.get(id);
  if (!task) return;
  updateTask(id, { done: !task.done });
  pushUndo({ label: task.done ? 'Task reopened' : 'Task completed',
    run: () => updateTask(id, { done: task.done }) });
  return !task.done;
}

export function deleteTask(id) {
  const task = store.tasks.get(id);
  if (!task) return null;
  store.tasks.delete(id);
  if (store.detailId === id) store.detailId = null;
  refreshTags();
  emit('delete');
  send({
    run: () => api.deleteTask(id),
    rollback: () => { store.tasks.set(id, task); refreshTags(); emit('delete'); },
  });
  pushUndo({ label: 'Task restored', run: () => restoreTask(id, task) });
  return task;
}

export function restoreTask(id, snapshot) {
  if (snapshot) { store.tasks.set(id, snapshot); refreshTags(); emit('restore'); }
  return send({
    run: () => api.restoreTask(id),
    commit: (task) => { if (task) { store.tasks.set(task.id, task); refreshTags(); emit('restore'); } },
    rollback: () => { store.tasks.delete(id); emit('restore'); },
  });
}

export function reorderTasks(orderedIds) {
  const before = orderedIds.map((id) => [id, store.tasks.get(id)?.position]);
  orderedIds.forEach((id, i) => {
    const t = store.tasks.get(id);
    if (t) t.position = i * 1024;
  });
  emit('reorder');
  send({
    run: () => api.reorder(orderedIds),
    rollback: () => {
      before.forEach(([id, pos]) => { const t = store.tasks.get(id); if (t) t.position = pos; });
      emit('reorder');
    },
  });
}

export function clearCompleted(projectId) {
  const removed = [...store.tasks.values()].filter(
    (t) => t.done && (!projectId || t.projectId === projectId));
  removed.forEach((t) => store.tasks.delete(t.id));
  emit('delete');
  send({
    run: () => api.clearCompleted(projectId),
    rollback: () => { removed.forEach((t) => store.tasks.set(t.id, t)); emit('delete'); },
  });
  pushUndo({
    label: `${removed.length} task${removed.length === 1 ? '' : 's'} restored`,
    run: async () => {
      removed.forEach((t) => store.tasks.set(t.id, t));
      emit('restore');
      await Promise.all(removed.map((t) => api.restoreTask(t.id).catch(() => {})));
    },
  });
  return removed;
}

/* ── project mutations ───────────────────────────────────────────────── */
export function createProject(data) {
  const temp = { id: `tmpp_${++tempSeq}`, name: data.name, color: data.color || 'violet',
    position: 1e9, createdAt: new Date().toISOString() };
  store.projects.push(temp);
  emit('project');
  return send({
    run: () => api.createProject(data),
    commit: (project) => {
      store.projects = store.projects.filter((p) => p.id !== temp.id);
      if (project) store.projects.push(project);
      store.projects.sort((a, b) => a.position - b.position);
      emit('project');
    },
    rollback: () => {
      store.projects = store.projects.filter((p) => p.id !== temp.id);
      emit('project');
    },
  });
}

export function updateProject(id, patch) {
  const i = store.projects.findIndex((p) => p.id === id);
  if (i === -1) return;
  const before = store.projects[i];
  store.projects[i] = { ...before, ...patch };
  emit('project');
  send({
    run: () => api.updateProject(id, patch),
    rollback: () => { store.projects[i] = before; emit('project'); },
  });
}

export function deleteProject(id) {
  const before = store.projects;
  const tasks = [...store.tasks.values()].filter((t) => t.projectId === id);
  store.projects = store.projects.filter((p) => p.id !== id);
  tasks.forEach((t) => store.tasks.delete(t.id));
  if (store.prefs.projectId === id) savePrefs({ view: 'all', projectId: null });
  refreshTags();
  emit('project');
  send({
    run: () => api.deleteProject(id),
    rollback: () => {
      store.projects = before;
      tasks.forEach((t) => store.tasks.set(t.id, t));
      refreshTags();
      emit('project');
    },
  });
}

export const projectById = (id) => store.projects.find((p) => p.id === id) || null;
