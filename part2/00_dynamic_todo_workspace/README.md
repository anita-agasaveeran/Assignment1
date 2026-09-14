# Tasks — a full-stack todo app

A complete, working todo application: Python/SQLite REST API on the back, a
keyboard-first single-page app on the front, live-synced over server-sent
events. **No dependencies, no build step, no install** — the whole thing runs
on a stock Python 3.

```bash
python3 run.py
```

Then open <http://127.0.0.1:8000>. That's it.

Design rationale, alternatives considered and failure-mode analysis live in
[DESIGN.md](DESIGN.md).

---

## Why it's built this way

The machine this was written on has no Node toolchain, so the usual
React + Vite + Express stack wasn't available. Rather than make the project
un-runnable, it targets what is guaranteed present: Python's standard library
(`http.server`, `sqlite3`) and the browser's own module system. The result
starts instantly, has nothing to install, and nothing to keep up to date —
while still being a real client/server application with a real database.

## What it does

**Capture**
- **Natural-language quick add** — type `email the deck friday 2pm !p1 #work +Work`
  and it becomes a task due Friday at 2 pm, urgent, tagged `work`, filed in the
  Work list. Chips under the input show what was understood *before* you commit.
- Dates understood: `today`, `tonight`, `tomorrow`, `friday`, `next monday`,
  `next week`, `end of week`, `in 3 days`, `jan 5`, `5 jan`, `12/25`, `2026-01-05`;
  times: `2pm`, `14:30`, `at 5`, `noon`, `midnight`, `morning`, `evening`.

**Organize**
- Smart views: Today (overdue surfaced first), Upcoming (grouped by day), Inbox,
  All, Completed.
- Lists with colors, `#tags`, four priority levels, notes, due dates with or
  without a time.
- Drag to reorder; drag onto a list in the sidebar to move a task there.
- Sort by manual order, due date, priority, recency, or alphabetically.

**Move fast**
- `⌘K` command palette — search every task, jump to a list, run any command.
- Full keyboard control: `J`/`K` to move, `X` to complete, `E` to rename in
  place, `1`–`4` for priority, `⌫` to delete, `⌘Z` to undo, `?` for the list.
- Undo on every destructive action, via toast or `⌘Z`.

**Trust it**
- **Optimistic UI** — every action lands instantly; the request follows behind.
- **Offline-tolerant** — lose the network and your changes stay on screen,
  queue up ("3 changes waiting to sync"), and flush automatically on reconnect.
- **Live sync** — open two tabs, or a phone on the same Wi-Fi: a change in one
  appears in the others in milliseconds over SSE, no polling.
- Deletes are soft, so undo actually works; a purge runs at startup for
  anything trashed over 30 days ago.

**Details**
- Light/dark/system themes, remembered.
- Progress ring scoped to the view you're in.
- Responsive down to a phone; the sidebar becomes a drawer.
- Accessible: landmarks, live regions, focus traps in dialogs, visible focus
  rings, `prefers-reduced-motion` respected, every control reachable by keyboard.
- Export everything as JSON.

## Layout

```
fullstack-test/
├── run.py                 entry point
├── server/
│   ├── app.py             HTTP: routing, SSE stream, static hosting
│   ├── api.py             REST endpoints + validation (transport-free)
│   ├── db.py              SQLite schema, queries, seed data
│   └── events.py          in-process pub/sub for live updates
├── web/
│   ├── index.html         app shell
│   ├── styles.css         design tokens → layout → components
│   └── js/
│       ├── app.js         bootstrap, keyboard, composer
│       ├── store.js       state, optimistic mutations, undo, outbox
│       ├── api.js         fetch client + EventSource
│       ├── render.js      all DOM rendering
│       ├── views.js       filtering, sorting, grouping
│       ├── parse.js       natural-language quick add
│       ├── palette.js     ⌘K palette
│       ├── dates.js       formatting helpers
│       ├── dom.js         element helper + icon set
│       └── ui.js          toasts, modals, focus management
├── tests/test_api.py      end-to-end tests over real HTTP
└── data/todo.db           created on first run (WAL mode)
```

The client is deliberately layered: `store.js` never touches the DOM,
`render.js` never talks to the network, `api.js` never knows what a task means.

## API

All JSON. Send `X-Client-Id` and the server will tag your own events so your
tab can ignore its own echo.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/state` | everything needed to boot, one round trip |
| `GET` | `/api/health` | status + connected live listeners |
| `GET` | `/api/events` | SSE stream of every mutation |
| `GET` | `/api/export` | full JSON export |
| `POST` | `/api/tasks` | create |
| `GET` `PATCH` `DELETE` | `/api/tasks/{id}` | read / partial update / soft delete |
| `POST` | `/api/tasks/{id}/restore` | undo a delete |
| `POST` | `/api/tasks/reorder` | `{ids: [...]}` |
| `POST` | `/api/tasks/clear-completed` | optional `{projectId}` |
| `GET` `POST` | `/api/projects` | list / create |
| `PATCH` `DELETE` | `/api/projects/{id}` | rename, recolor, delete (cascades) |

Errors come back as `{"error": {"status", "message", "field"}}` with a message
written for a person, and the field that caused it.

```bash
curl -X POST localhost:8000/api/tasks \
  -H 'Content-Type: application/json' \
  -d '{"title":"Try the API","priority":2,"tags":["demo"]}'
```

## Running

```bash
python3 run.py                  # http://127.0.0.1:8000
python3 run.py --port 3000      # somewhere else (auto-picks if busy)
python3 run.py --reset          # wipe and re-seed the demo data
python3 run.py --no-seed        # start empty
python3 run.py --host 0.0.0.0   # reachable from your phone on the same Wi-Fi
```

The database lives at `data/todo.db`; set `TODO_DB` to move it.

## Tests

```bash
python3 tests/test_api.py
```

21 end-to-end tests against a real server and a real database: task and project
lifecycles, soft delete and restore, reordering, tag normalization, every
validation path, SSE broadcast, plus static hosting, SPA deep links and path
traversal.

## Notes and limits

- Single user, no auth — it binds to localhost by default and is meant to run
  on your own machine. Putting it on a network would need accounts and TLS.
- SSE fan-out is in-process, so it's one server process; scaling out would mean
  moving the pub/sub to Redis or similar.
- `data/todo.db` holds all state. Copy it to back up, delete it to start over.
