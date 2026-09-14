/* Natural-language quick add.
 *
 *   "email tessa friday 2pm !p1 #work +Personal"
 *      → { title:"email tessa", dueAt, allDay:false, priority:1,
 *          tags:["work"], projectName:"Personal", tokens:[…] }
 *
 * Tokens carry their source range so the composer can show live chips.
 */

import { startOfDay, addDays, toISO } from './dates.js';

const DOW = { sunday: 0, sun: 0, monday: 1, mon: 1, tuesday: 2, tue: 2, tues: 2, wednesday: 3,
  wed: 3, thursday: 4, thu: 4, thur: 4, thurs: 4, friday: 5, fri: 5, saturday: 6, sat: 6 };
const MONTHS = { jan: 0, january: 0, feb: 1, february: 1, mar: 2, march: 2, apr: 3, april: 3,
  may: 4, jun: 5, june: 5, jul: 6, july: 6, aug: 7, august: 7, sep: 8, sept: 8, september: 8,
  oct: 9, october: 9, nov: 10, november: 10, dec: 11, december: 11 };

const DOW_RE = Object.keys(DOW).join('|');
const MON_RE = Object.keys(MONTHS).join('|');

function nextDow(target, includeToday = false) {
  const today = startOfDay();
  let delta = (target - today.getDay() + 7) % 7;
  if (delta === 0 && !includeToday) delta = 7;
  return addDays(today, delta);
}

function withTime(date, h, m) {
  const d = new Date(date);
  d.setHours(h, m, 0, 0);
  return d;
}

/* ── time of day ─────────────────────────────────────────────────────── */
const TIME_RULES = [
  [/\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b/i, (m) => {
    let h = +m[1] % 12;
    if (/pm/i.test(m[3])) h += 12;
    return [h, +(m[2] || 0)];
  }],
  [/\b(?:at\s+)?(\d{1,2}):(\d{2})\b/, (m) => (+m[1] < 24 && +m[2] < 60 ? [+m[1], +m[2]] : null)],
  [/\bnoon\b/i, () => [12, 0]],
  [/\bmidnight\b/i, () => [0, 0]],
  [/\bmorning\b/i, () => [9, 0]],
  [/\bafternoon\b/i, () => [14, 0]],
  [/\b(?:evening|tonight)\b/i, () => [19, 0]],
];

/* ── day ─────────────────────────────────────────────────────────────── */
const DAY_RULES = [
  [/\btoday\b|\btod\b/i, () => startOfDay()],
  [/\btonight\b/i, () => startOfDay()],
  [/\btomorrow\b|\btmrw?\b|\btmr\b/i, () => addDays(startOfDay(), 1)],
  [/\byesterday\b/i, () => addDays(startOfDay(), -1)],
  [/\bnext\s+week\b/i, () => nextDow(1)],
  [/\bnext\s+month\b/i, () => { const d = startOfDay(); d.setMonth(d.getMonth() + 1); return d; }],
  [/\b(?:end\s+of\s+week|eow)\b/i, () => nextDow(5, true)],
  [new RegExp(`\\bnext\\s+(${DOW_RE})\\b`, 'i'), (m) => nextDow(DOW[m[1].toLowerCase()])],
  [new RegExp(`\\b(?:this\\s+|on\\s+)?(${DOW_RE})\\b`, 'i'),
    (m) => nextDow(DOW[m[1].toLowerCase()], false)],
  [/\bin\s+(\d{1,3})\s*(day|days|d|week|weeks|w|month|months)\b/i, (m) => {
    const n = +m[1], u = m[2][0].toLowerCase();
    const d = startOfDay();
    if (u === 'd') return addDays(d, n);
    if (u === 'w') return addDays(d, n * 7);
    d.setMonth(d.getMonth() + n);
    return d;
  }],
  [/\b(\d{4})-(\d{2})-(\d{2})\b/, (m) => new Date(+m[1], +m[2] - 1, +m[3])],
  [/\b(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?\b/, (m) => {
    const y = m[3] ? (m[3].length === 2 ? 2000 + +m[3] : +m[3]) : new Date().getFullYear();
    const d = new Date(y, +m[1] - 1, +m[2]);
    return d.getMonth() === +m[1] - 1 ? d : null;
  }],
  [new RegExp(`\\b(${MON_RE})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b`, 'i'), (m) => {
    const d = new Date(new Date().getFullYear(), MONTHS[m[1].toLowerCase()], +m[2]);
    return d < startOfDay() ? new Date(d.setFullYear(d.getFullYear() + 1)) : d;
  }],
  [new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(${MON_RE})\\b`, 'i'), (m) => {
    const d = new Date(new Date().getFullYear(), MONTHS[m[2].toLowerCase()], +m[1]);
    return d < startOfDay() ? new Date(d.setFullYear(d.getFullYear() + 1)) : d;
  }],
];

const PRIORITY_LABEL = { 1: 'Urgent', 2: 'High', 3: 'Medium', 4: 'Normal' };

export function parse(input, { projects = [] } = {}) {
  const text = input || '';
  const taken = [];                       // [start, end) ranges consumed by tokens
  const tokens = [];
  const free = (s, e) => taken.every(([a, b]) => e <= a || s >= b);
  const claim = (s, e, token) => { taken.push([s, e]); tokens.push({ ...token, start: s, end: e }); };

  const result = { title: '', notes: '', dueAt: null, allDay: true, priority: 4,
    tags: [], projectId: null, projectName: null, tokens };

  /* tags — #work */
  for (const m of text.matchAll(/(^|\s)#([\p{L}\p{N}_-]{1,32})/gu)) {
    const s = m.index + m[1].length;
    if (!free(s, s + m[0].length - m[1].length)) continue;
    claim(s, s + m[2].length + 1, { type: 'tag', label: '#' + m[2] });
    if (!result.tags.some((t) => t.toLowerCase() === m[2].toLowerCase())) result.tags.push(m[2]);
  }

  /* priority — !p1 / !1 / !! */
  for (const m of text.matchAll(/(^|\s)!(?:p)?([1-4])\b/gi)) {
    const s = m.index + m[1].length;
    if (!free(s, s + m[0].length - m[1].length)) continue;
    claim(s, s + m[0].length - m[1].length, { type: 'priority', value: +m[2],
      label: PRIORITY_LABEL[+m[2]] });
    result.priority = +m[2];
  }

  /* list — +Work (matched against real lists, longest name first) */
  const sorted = [...projects].sort((a, b) => b.name.length - a.name.length);
  for (const p of sorted) {
    const re = new RegExp(`(^|\\s)\\+(${p.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})\\b`, 'i');
    const m = text.match(re);
    if (m) {
      const s = m.index + m[1].length;
      if (!free(s, s + m[0].length - m[1].length)) continue;
      claim(s, s + m[2].length + 1, { type: 'list', label: p.name });
      result.projectId = p.id;
      result.projectName = p.name;
      break;
    }
  }

  /* day */
  let day = null;
  for (const [re, fn] of DAY_RULES) {
    const m = text.match(re);
    if (!m || !free(m.index, m.index + m[0].length)) continue;
    const d = fn(m);
    if (!d || Number.isNaN(+d)) continue;
    day = d;
    claim(m.index, m.index + m[0].length, { type: 'date' });
    break;
  }

  /* time of day (implies a day — defaults to today, or tomorrow if already past) */
  let time = null;
  for (const [re, fn] of TIME_RULES) {
    const m = text.match(re);
    if (!m || !free(m.index, m.index + m[0].length)) continue;
    const hm = fn(m);
    if (!hm) continue;
    time = hm;
    claim(m.index, m.index + m[0].length, { type: 'time' });
    break;
  }

  if (day || time) {
    let d = day || startOfDay();
    if (time) {
      d = withTime(d, time[0], time[1]);
      if (!day && d < new Date()) d = addDays(d, 1);
      result.allDay = false;
    }
    result.dueAt = toISO(d);
    for (const t of tokens) {
      if (t.type === 'date' || t.type === 'time') {
        t.label = t.label || null;
        t.due = result.dueAt;
        t.allDay = result.allDay;
      }
    }
  }

  /* title = whatever is left */
  let title = '';
  let cursor = 0;
  const ranges = taken.slice().sort((a, b) => a[0] - b[0]);
  for (const [s, e] of ranges) {
    title += text.slice(cursor, Math.max(cursor, s));
    cursor = Math.max(cursor, e);
  }
  title += text.slice(cursor);
  result.title = title.replace(/\s{2,}/g, ' ').replace(/\s+([,.!?])/g, '$1').trim();

  /* a bare "#tag" with no words left is a title, not a tag */
  if (!result.title && result.tags.length) {
    result.title = text.trim();
    result.tags = [];
    result.tokens.length = 0;
  }
  return result;
}

export { PRIORITY_LABEL };
