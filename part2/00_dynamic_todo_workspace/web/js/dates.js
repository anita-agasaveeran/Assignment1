/* Date helpers. Everything is stored as UTC ISO; everything is shown local. */

export const MS_DAY = 86400000;

export const startOfDay = (d = new Date()) =>
  new Date(d.getFullYear(), d.getMonth(), d.getDate());

export const addDays = (d, n) => {
  const x = new Date(d);
  x.setDate(x.getDate() + n);
  return x;
};

export const toISO = (d) => new Date(d).toISOString().replace(/\.\d{3}Z$/, 'Z');

export const dayDiff = (iso, from = new Date()) =>
  Math.round((startOfDay(new Date(iso)) - startOfDay(from)) / MS_DAY);

export const isSameDay = (a, b) => startOfDay(a).getTime() === startOfDay(b).getTime();

const timeFmt = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' });
const dowFmt = new Intl.DateTimeFormat(undefined, { weekday: 'long' });
const shortFmt = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' });
const longFmt = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' });

export function formatTime(iso) {
  return timeFmt.format(new Date(iso)).replace(/\s?([AP])M/i, (_, p) => p.toLowerCase() + 'm');
}

/** "Today · 3pm", "Overdue — Mon", "In 4 days" … */
export function formatDue(iso, allDay = true) {
  if (!iso) return '';
  const d = new Date(iso);
  const diff = dayDiff(iso);
  const time = allDay ? '' : ' ' + formatTime(iso);
  let day;
  if (diff === 0) day = 'Today';
  else if (diff === 1) day = 'Tomorrow';
  else if (diff === -1) day = 'Yesterday';
  else if (diff > 1 && diff < 7) day = dowFmt.format(d);
  else if (diff < -1 && diff > -7) day = 'Last ' + dowFmt.format(d);
  else day = d.getFullYear() === new Date().getFullYear() ? shortFmt.format(d) : longFmt.format(d);
  return day + time;
}

export function dueState(iso, done) {
  if (!iso || done) return '';
  const diff = dayDiff(iso);
  if (diff < 0) return 'overdue';
  if (diff === 0) return 'today';
  return 'upcoming';
}

export function formatStamp(iso) {
  if (!iso) return '';
  const diff = Math.round((Date.now() - new Date(iso)) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return shortFmt.format(new Date(iso));
}

/** Value for <input type="datetime-local"> / "date" in local time. */
export function toInputValue(iso, allDay) {
  if (!iso) return '';
  const d = new Date(iso);
  const p = (n) => String(n).padStart(2, '0');
  const date = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
  return allDay ? date : `${date}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

export function fromInputValue(value) {
  if (!value) return null;
  const allDay = !value.includes('T');
  const d = allDay ? new Date(value + 'T00:00:00') : new Date(value);
  return Number.isNaN(+d) ? null : { dueAt: toISO(d), allDay };
}
