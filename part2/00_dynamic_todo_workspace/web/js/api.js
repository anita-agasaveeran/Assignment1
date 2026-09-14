/* Thin REST client + live event stream. */

export const clientId = (() => {
  let id = sessionStorage.getItem('todo.clientId');
  if (!id) {
    id = 'c_' + Math.random().toString(36).slice(2, 10);
    sessionStorage.setItem('todo.clientId', id);
  }
  return id;
})();

export class ApiError extends Error {
  constructor(status, message, field) {
    super(message);
    this.status = status;
    this.field = field;
    this.offline = status === 0;
  }
}

export async function request(method, path, body) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body !== undefined
        ? { 'Content-Type': 'application/json', 'X-Client-Id': clientId }
        : { 'X-Client-Id': clientId },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(0, 'You appear to be offline. Changes are saved locally.');
  }
  if (res.status === 204) return null;
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = data?.error || {};
    throw new ApiError(res.status, err.message || `Request failed (${res.status}).`, err.field);
  }
  return data;
}

export const api = {
  state: () => request('GET', '/api/state'),
  createTask: (t) => request('POST', '/api/tasks', t),
  updateTask: (id, patch) => request('PATCH', `/api/tasks/${id}`, patch),
  deleteTask: (id) => request('DELETE', `/api/tasks/${id}`),
  restoreTask: (id) => request('POST', `/api/tasks/${id}/restore`, {}),
  reorder: (ids) => request('POST', '/api/tasks/reorder', { ids }),
  clearCompleted: (projectId) => request('POST', '/api/tasks/clear-completed', { projectId }),
  createProject: (p) => request('POST', '/api/projects', p),
  updateProject: (id, patch) => request('PATCH', `/api/projects/${id}`, patch),
  deleteProject: (id) => request('DELETE', `/api/projects/${id}`),
};

/** Server-sent events: other tabs/devices show up here within milliseconds. */
export function connectEvents({ onEvent, onStatus, onGap }) {
  let source = null;
  let retry = 1000;
  let lastSeq = null;
  let hadConnection = false;

  const open = () => {
    source = new EventSource('/api/events');
    source.onopen = () => {
      retry = 1000;
      onStatus('live');
      // Nothing replays events sent while we were disconnected, so a
      // reconnect means our snapshot may be stale.
      if (hadConnection) { lastSeq = null; onGap?.(); }
      hadConnection = true;
    };
    source.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        // Sequence numbers cover every client's events, so a jump means we
        // missed frames (a full queue on the server drops slow subscribers).
        if (lastSeq !== null && msg.seq > lastSeq + 1) onGap?.();
        lastSeq = msg.seq;
        if (msg.origin !== clientId) onEvent(msg);
      } catch { /* ignore malformed frame */ }
    };
    source.onerror = () => {
      onStatus(navigator.onLine ? 'reconnecting' : 'offline');
      source.close();
      setTimeout(open, retry);
      retry = Math.min(retry * 1.8, 15000);
    };
  };

  open();
  addEventListener('online', () => {
    // Don't claim we're reconnecting if the stream never actually dropped.
    if (source?.readyState !== EventSource.OPEN) onStatus('reconnecting');
  });
  addEventListener('offline', () => onStatus('offline'));
  return () => source?.close();
}
