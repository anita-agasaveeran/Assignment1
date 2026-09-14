/* Selectors: which tasks a view shows, in what order, grouped how. */

import { dayDiff, formatDue, startOfDay } from './dates.js';

export const VIEWS = [
  { id: 'today', label: 'Today', icon: 'today' },
  { id: 'upcoming', label: 'Upcoming', icon: 'upcoming' },
  { id: 'inbox', label: 'Inbox', icon: 'inbox' },
  { id: 'all', label: 'All tasks', icon: 'all' },
  { id: 'completed', label: 'Completed', icon: 'done' },
];

export const SORTS = [
  { id: 'manual', label: 'My order' },
  { id: 'due', label: 'Due date' },
  { id: 'priority', label: 'Priority' },
  { id: 'created', label: 'Recently added' },
  { id: 'alpha', label: 'Alphabetical' },
];

const isOverdue = (t) => t.dueAt && !t.done && dayDiff(t.dueAt) < 0;
const isToday = (t) => t.dueAt && dayDiff(t.dueAt) === 0;

export function matchesView(task, prefs) {
  switch (prefs.view) {
    case 'inbox': return !task.projectId && !task.done;
    case 'today': return !task.done && !!task.dueAt && dayDiff(task.dueAt) <= 0;
    case 'upcoming': return !task.done && !!task.dueAt && dayDiff(task.dueAt) > 0;
    case 'all': return !task.done;
    case 'completed': return task.done;
    case 'project': return task.projectId === prefs.projectId && !task.done;
    default: return !task.done;
  }
}

export function counts(tasks, projects) {
  const c = { today: 0, upcoming: 0, inbox: 0, all: 0, completed: 0, overdue: 0, byProject: {},
    byTag: {} };
  for (const p of projects) c.byProject[p.id] = 0;
  for (const t of tasks) {
    if (t.done) { c.completed++; continue; }
    c.all++;
    if (!t.projectId) c.inbox++;
    if (t.dueAt && dayDiff(t.dueAt) <= 0) c.today++;
    if (t.dueAt && dayDiff(t.dueAt) > 0) c.upcoming++;
    if (isOverdue(t)) c.overdue++;
    if (t.projectId && t.projectId in c.byProject) c.byProject[t.projectId]++;
    for (const tag of t.tags) c.byTag[tag] = (c.byTag[tag] || 0) + 1;
  }
  return c;
}

/** Completed vs. open within the current view's subject matter — what the
 *  progress ring means. A view's "scope" ignores the done/not-done filter. */
export function scopeCounts(tasks, prefs) {
  const inScope = (t) => {
    switch (prefs.view) {
      case 'inbox': return !t.projectId;
      case 'today': return !!t.dueAt && dayDiff(t.dueAt) <= 0;
      case 'upcoming': return !!t.dueAt && dayDiff(t.dueAt) > 0;
      case 'project': return t.projectId === prefs.projectId;
      case 'completed': return t.done;
      default: return true;
    }
  };
  let open = 0, done = 0;
  for (const t of tasks) {
    if (!inScope(t)) continue;
    if (t.done) done++; else open++;
  }
  return { open, done };
}

const comparators = {
  manual: (a, b) => a.position - b.position,
  due: (a, b) => (a.dueAt ? Date.parse(a.dueAt) : Infinity) - (b.dueAt ? Date.parse(b.dueAt) : Infinity)
    || a.priority - b.priority,
  priority: (a, b) => a.priority - b.priority
    || (a.dueAt ? Date.parse(a.dueAt) : Infinity) - (b.dueAt ? Date.parse(b.dueAt) : Infinity),
  created: (a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt),
  alpha: (a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: 'base' }),
};

export function searchScore(task, query) {
  if (!query) return 1;
  const q = query.toLowerCase();
  const title = task.title.toLowerCase();
  if (title.includes(q)) return 3 - title.indexOf(q) / 1000;
  if (task.notes.toLowerCase().includes(q)) return 1.5;
  if (task.tags.some((t) => t.toLowerCase().includes(q))) return 1.2;
  return 0;
}

/** → [{ key, label, tone, tasks }] */
export function buildGroups(allTasks, prefs, query) {
  const { view, tag, sort, showCompleted } = prefs;
  let tasks = allTasks.filter((t) => matchesView(t, prefs));
  if (tag) tasks = tasks.filter((t) => t.tags.some((x) => x.toLowerCase() === tag.toLowerCase()));
  if (query) tasks = tasks.filter((t) => searchScore(t, query) > 0);

  let completed = [];
  if (showCompleted && view !== 'completed') {
    completed = allTasks.filter((t) => t.done
      && (view !== 'project' || t.projectId === prefs.projectId)
      && (view !== 'inbox' || !t.projectId)
      && (!tag || t.tags.some((x) => x.toLowerCase() === tag.toLowerCase()))
      && (!query || searchScore(t, query) > 0))
      .sort((a, b) => Date.parse(b.completedAt || 0) - Date.parse(a.completedAt || 0))
      .slice(0, 50);
  }

  const cmp = comparators[sort] || comparators.manual;
  tasks.sort(cmp);

  const groups = [];
  if (view === 'today') {
    const overdue = tasks.filter((t) => dayDiff(t.dueAt) < 0);
    const rest = tasks.filter((t) => dayDiff(t.dueAt) === 0);
    if (overdue.length) groups.push({ key: 'overdue', label: 'Overdue', tone: 'danger', tasks: overdue });
    if (rest.length) groups.push({ key: 'today', label: 'Today', tasks: rest });
  } else if (view === 'upcoming') {
    const byDay = new Map();
    for (const t of tasks) {
      const key = startOfDay(new Date(t.dueAt)).toDateString();
      if (!byDay.has(key)) byDay.set(key, []);
      byDay.get(key).push(t);
    }
    [...byDay.entries()]
      .sort((a, b) => Date.parse(a[0]) - Date.parse(b[0]))
      .forEach(([key, list]) => groups.push({
        key, label: formatDue(list[0].dueAt, true), tasks: list,
      }));
  } else if (view === 'completed') {
    const byDay = new Map();
    for (const t of tasks.sort((a, b) => Date.parse(b.completedAt || 0) - Date.parse(a.completedAt || 0))) {
      const key = t.completedAt ? startOfDay(new Date(t.completedAt)).toDateString() : 'unknown';
      if (!byDay.has(key)) byDay.set(key, []);
      byDay.get(key).push(t);
    }
    for (const [key, list] of byDay) {
      groups.push({ key, label: key === 'unknown' ? 'Completed' : formatDue(list[0].completedAt, true), tasks: list });
    }
  } else if (tasks.length) {
    groups.push({ key: 'main', label: null, tasks });
  }

  if (completed.length) groups.push({ key: 'done', label: 'Completed', tasks: completed, collapsedByDefault: true });
  return { groups, total: tasks.length, completedShown: completed.length };
}

export const viewMeta = (prefs, projects) => {
  if (prefs.view === 'project') {
    const p = projects.find((x) => x.id === prefs.projectId);
    return { title: p ? p.name : 'List', icon: 'list', color: p?.color };
  }
  const v = VIEWS.find((x) => x.id === prefs.view) || VIEWS[0];
  return { title: v.label, icon: v.icon };
};

export { isOverdue, isToday };
