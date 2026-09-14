# Design — Tasks

A design record for the todo application in this directory: what was built, why
it was built that way, what was rejected, and where it breaks.

---

## 1. Context and constraints

The target machine has **no Node toolchain** — no `node`, no `nvm`, no Homebrew
node. Python 3.14 is present with `sqlite3`. PyPI is reachable but slow (an
8-second `curl` to the simple index timed out mid-transfer).

That single fact drove the whole design. The usual answer to "build a modern
full-stack todo app" — React + Vite + Express + Prisma — was not runnable, and
a project that cannot be started is not a deliverable.

**Decision: use only what is guaranteed present.** Python's standard library on
the server (`http.server`, `sqlite3`), and the browser's native module system on
the client. No package manager, no build step, no lockfile, nothing to install.

Consequences accepted up front: no JSX, no TypeScript, no bundler, no component
library, no test runner beyond `unittest`. Everything below is shaped by that.

## 2. Goals and non-goals

**Goals**

- A genuine client/server application with a real database — not a
  `localStorage` toy that only looks full-stack.
- Starts with one command and no install, on a machine with nothing set up.
- Interaction quality of a modern task app: optimistic updates, undo, keyboard
  control, live multi-client sync.
- Honest failure behavior — the UI should never lie about whether your data
  is saved.
- Testable without a browser driver.

**Non-goals** (deliberately out of scope, not oversights)

- Authentication and multi-tenancy. Single user, bound to localhost.
- Conflict resolution beyond last-write-wins per field.
- True offline-first persistence in the browser (no IndexedDB mirror).
- Recurring tasks, subtasks, attachments, reminders, collaboration.
- Internationalization; dates render in the viewer's locale but UI text is English.

## 3. Alternatives considered

| Option | Verdict |
|---|---|
| React + Vite + Express | **Rejected.** No Node on the machine. Non-starter, not a preference. |
| Flask / FastAPI + uvicorn | **Rejected.** Needs `pip install` over a slow, flaky link. Buys routing and validation worth ~140 lines here. A dependency that can fail to install is a liability for a project whose main promise is "it runs". |
| Django | **Rejected.** ORM, migrations, admin, templates — nearly all unused. Cost far exceeds benefit at this size. |
| SPA with `localStorage`, no server | **Rejected.** Not end-to-end. No shared source of truth, no real API, nothing to test on the server side. |
| WebSockets for live updates | **Rejected.** The stdlib has no server-side WebSocket implementation, so this would reintroduce a dependency. The traffic is one-way (server → client); WebSockets solve a problem this app does not have. |
| Polling for live updates | **Rejected.** Wasteful when idle, laggy when not. SSE is strictly better for server-push over plain HTTP and needs no library on either end. |
| Client-side rendering with a virtual DOM | **Rejected.** See §8.4 — full re-render is correct by construction and fast enough at this scale. |

## 4. System architecture

```
┌──────────────────────── browser ────────────────────────┐
│  app.js      bootstrap, global keyboard, composer       │
│  store.js    state, optimistic mutations, undo, outbox  │  ← no DOM access
│  render.js   all DOM rendering                          │  ← no network access
│  api.js      fetch client + EventSource                 │  ← no domain logic
│  views.js / parse.js / dates.js / dom.js / ui.js        │
└───────┬─────────────────────┬───────────────────┬───────┘
        │ JSON over HTTP      │ SSE (read-only)   │ static assets
        ▼                     ▼                   ▼
┌────────────────────────── server ───────────────────────┐
│  app.py     HTTP transport, SSE stream, static hosting  │
│  api.py     routing + validation  (transport-free)      │
│  events.py  in-process pub/sub                          │
│  db.py      SQLite access + schema                      │
└─────────────────────────────┬───────────────────────────┘
                              ▼
                    data/todo.db  (WAL)
```

**Threading model.** `ThreadingHTTPServer` gives one thread per request. Each
thread gets its own SQLite connection via `threading.local()`, because SQLite
connections are not safely shared across threads. All writes take a single
process-wide lock and run inside `BEGIN IMMEDIATE`; reads never block, thanks
to WAL. This is not a high-concurrency design, and does not need to be — the
expected load is one human and a handful of tabs.

**Layering rule on the client.** `store.js` never touches the DOM; `render.js`
never touches the network; `api.js` knows nothing about what a task means. This
is what makes the optimistic-update logic testable by reading it, and it is the
reason the store could be swapped for a different backend without touching
rendering.

## 5. Data model

```sql
projects(id, name, color, position, created_at)
tasks(id, project_id→projects, title, notes, done, priority, due_at,
      all_day, position, created_at, updated_at, completed_at, deleted_at)
tags(id, name UNIQUE COLLATE NOCASE)
task_tags(task_id→tasks, tag_id→tags)      -- many-to-many
```

Decisions worth recording:

- **Prefixed string ids** (`tsk_`, `prj_`, `tag_`) rather than integers. They
  are self-describing in logs, URLs and error messages, and they make it
  impossible to pass a project id where a task id belongs and have it "work".
- **Soft delete** (`deleted_at`) rather than `DELETE`. Undo is a product
  requirement, and undo built on a tombstone is trivially correct, whereas undo
  built on re-inserting a reconstructed row is not. A purge of anything trashed
  more than 30 days ago runs at startup.
- **`position REAL`** for manual ordering. Reordering currently renumbers the
  visible list as `i * 1024`; new tasks get `min(position) − 1024` so they land
  on top. Floats leave room for fractional insertion (`(prev+next)/2`) without
  a schema change if renumbering ever gets expensive — it does not at this size.
- **`priority` as 1–4, urgent → none.** Ascending numeric sort is the display
  order, so priority sorting needs no case analysis.
- **Tags normalized** into their own table with a case-insensitive unique name,
  so `#Work` and `#work` cannot both exist. Input is trimmed, `#`-stripped,
  deduplicated and capped at 32 characters on write.
- **`due_at` stored as UTC ISO-8601, always**, with a separate `all_day` flag.
  An all-day task is stored at local midnight converted to UTC and rendered back
  in local time, so it displays on the intended date. The flag is what stops the
  UI from printing a meaningless "12:00 AM" on every dateless deadline.

**A timezone bug this caused.** The demo seed originally built timestamps in
UTC, so "water the plants today at 6pm" was written as `18:00Z` — which is
11am *the same day* in UTC−7 but lands on the *following* day for a task seeded
near midnight UTC. Since the server and browser are by definition the same
machine here, the seed now anchors to the server's local timezone. Storage
stays UTC; only the seed's notion of "today at 6pm" is local.

## 6. API design

Resource-oriented JSON. The full contract is in the README; the design points:

- **`GET /api/state` returns everything needed to boot in one round trip** —
  tasks, projects, tags. With a dataset this size, one request beats three
  and removes all boot-time ordering questions between them.
- **`PATCH` applies partial updates.** Only the keys present are written, which
  is what makes two tabs editing different fields of the same task both win.
- **Errors are an envelope**: `{"error": {"status", "message", "field"}}`. The
  message is written for a person to read and is displayed verbatim in a toast;
  `field` lets the client point at the offending input. Validation is
  centralized in `api.py` so the same rules apply to the UI and to `curl`.
- **Delete is soft and paired with `POST /api/tasks/{id}/restore`**, so undo is
  a first-class API operation rather than a client-side illusion.
- **`X-Client-Id`** on every mutating request. The server echoes it on the
  broadcast event so the originating tab can ignore its own change (§7).
- **No authentication.** Stated plainly rather than half-built: the server binds
  to `127.0.0.1` by default, and `--host 0.0.0.0` is documented as a LAN-only
  convenience. Anything beyond that needs accounts and TLS.

## 7. Live sync

The mechanism: every mutation calls `events.publish(type, payload, origin)`,
which fans the JSON frame out to one `queue.Queue` per connected SSE client.
Each client's HTTP thread blocks on its queue and writes frames as they arrive.

Design points:

- **Writers must never block on a slow reader.** Each subscriber queue is
  bounded at 256 frames and `put_nowait` is used; a client that cannot keep up
  is dropped from the subscriber set rather than being allowed to stall a write.
  Availability of the write path beats delivery to a stalled listener.
- **20-second heartbeat comments** (`: ping`) keep intermediaries and the
  browser from treating an idle stream as dead.
- **Echo suppression by origin id.** Without it, the tab that just made an
  optimistic change would immediately receive its own change back and re-render
  from it, which both wastes work and can visibly clobber in-flight edits.
- **A monotonic `seq` on every frame**, counted across all clients. The client
  tracks the last sequence number it saw, and a jump means it missed frames.

**Delivery guarantee: at-most-once, with gap-triggered resync.** There is no
replay buffer, so anything published while a client is disconnected is gone.
Rather than pretend otherwise, the client detects the two ways it can fall
behind — a sequence gap, or a reconnect after a previous connection — and
recovers by re-fetching `/api/state` wholesale. Cheap at this data size, and
correct regardless of how many events were missed.

This was the last real defect found: the `seq` field was being published and
ignored, so a tab whose stream dropped went stale permanently and silently. It
now resyncs; verified by killing the server, creating a task from `curl` while
the tab was disconnected, and watching the tab reconnect and pick it up without
a reload.

The upgrade path, if this ever needed to be exact: keep a bounded ring buffer of
recent events server-side and honor the `Last-Event-ID` header that EventSource
already sends on reconnect, replaying the tail instead of resyncing.

## 8. Client architecture

### 8.1 Data flow

Strictly one-directional: **user event → store mutation → `emit()` → render**.
Rendering never mutates state, and mutations never touch the DOM. Remote events
enter at exactly the same place as local ones (`applyRemote` → `emit`), so a
change from another tab and a change from this tab travel identical paths.

### 8.2 Optimistic mutation lifecycle

Every mutation applies to the store immediately, then sends the request:

```
apply locally ─→ emit ─→ render          (instant)
      └─→ request ─┬─ ok      → commit server truth, re-emit
                   ├─ 4xx/5xx → rollback snapshot + error toast
                   └─ network → keep change, push to outbox, mark offline
```

Creates use a temporary `tmp_` id and reconcile to the server id on commit,
including moving the detail-drawer selection if that task is open. Updates to a
task whose create is still in flight skip the network and ride along with it.

### 8.3 Undo and the outbox

**Undo** is a stack of inverse closures (cap 50), not a snapshot of the world.
A completion pushes "set done back"; a delete pushes "restore this id". This
keeps undo O(1) in memory and makes it composable with the optimistic path,
because an undo is just another mutation.

**The outbox** is what makes the offline claim true. A request that fails with a
network error (not an HTTP error) parks its operation in a queue; the pill in
the sidebar reports the depth (`"3 changes waiting to sync"`), and a flush is
attempted on a 4-second retry timer, on the browser's `online` event, and
whenever the event stream reports itself live again. The flush is ordered and
stops at the first still-offline operation, re-queueing the remainder ahead of
anything newer.

That last detail was a bug: the original flush `return`ed on the first failure
and silently discarded the rest of the batch. Three queued tasks would sync one
and lose two. Found by stubbing `fetch` to reject, queueing three creates, and
asserting all three arrived, in order, exactly once.

### 8.4 Rendering strategy

The visible list is re-rendered in full on every change, coalesced so that many
mutations in one tick produce one render. No virtual DOM, no diffing, no
component framework.

Justification: at a realistic size (hundreds to low thousands of rows) rebuilding
a list of DOM nodes is around a millisecond, and *correct by construction* —
there is no reconciliation logic to get subtly wrong, and no stale-state class of
bug at all. The two places that genuinely need continuity are handled explicitly:
newly-arrived rows are detected by comparing ids against the previous render (to
animate only those), and an in-progress inline edit is committed before its row
is replaced. If the dataset ever outgrew this, the fix is windowing the list, not
adopting a diffing layer.

**Scheduling.** Renders are queued with `requestAnimationFrame` — except when
`document.hidden`, where a timer is used instead. This is not a micro-optimization:
`rAF` is *paused* in a background tab, so a hidden tab receiving live updates
would queue one frame and then coalesce every subsequent change into it forever,
appearing frozen and — worse — stalling the store's own change notifications.
The same trap applied to dialog autofocus, which used `rAF` and therefore opened
dialogs with no focus in a background tab. Both now use timers.

## 9. Natural-language quick add

`parse.js` turns `email the deck friday 2pm !p1 #work +Work` into a structured
task. The algorithm is **range claiming**: each rule matches against the raw
string and claims a character range; a rule may only claim a range that does not
overlap one already taken; whatever text is left over, joined and whitespace-
collapsed, becomes the title.

This ordering matters and is deliberate — unambiguous sigils first (`#tag`,
`!p1`, `+List`, matched longest-list-name-first), then a day, then a time.
Running dates last means `friday` inside `#friday-standup` is already claimed
by the tag rule and cannot be stolen by the date rule.

Two behavioral rules worth stating:

- **A time implies a day.** `2pm` alone means today, unless today at 2pm has
  already passed, in which case tomorrow — matching what people mean.
- **A parse that consumes the entire title is wrong.** Typing just `#work`
  should create a task *called* "#work", not a titleless task with a tag, so
  that case falls back to treating the raw text as the title.

**Parsing lives on the client only.** The alternative — parse on the server,
call it per keystroke for the preview chips — adds a round trip to every
character typed. Duplicating the grammar in both languages was rejected outright
as two implementations that would drift. So the server's API accepts structured
fields only, and remains the single authority on validation.

## 10. Interaction principles

- **Optimistic, then undo — not confirm-then-act.** Confirmation dialogs are
  reserved for actions that are destructive *at scale* (clear all completed,
  delete a list and its tasks). Single-item actions just happen, with an undo
  offered in a toast and on `⌘Z`.
- **Keyboard-first.** Every operation has a shortcut; `⌘K` searches tasks, jumps
  to lists and runs commands from one input.
- **Show the parse before committing it.** The composer renders chips for the
  date, priority, tag and list it understood, so the natural-language input is
  never a guess the user discovers after pressing Enter.
- **Progress is scoped to the current view.** A ring that counts globally while
  you are looking at Today is noise; it counts completed-vs-open within whatever
  the view is about.
- **Empty states are per-view and actionable**, because "Today is clear" and
  "no search results" call for different words and different next steps.

## 11. Accessibility

Semantic landmarks (`aside`/`main`/`header`), a skip link, `role="checkbox"` with
`aria-checked` on task toggles, `aria-current` on the active view, live regions
for toasts and for announcements like "Added … due Friday". Dialogs trap Tab,
close on Escape, and restore focus to the element that opened them. Focus rings
are visible and never suppressed without replacement. `prefers-reduced-motion`
disables all animation. Colour is never the only carrier of meaning — overdue
items say "Overdue" as well as turning red.

## 12. Failure modes

| Failure | Designed behavior |
|---|---|
| Validation error (422) | Optimistic change rolled back; server's message shown in a toast; offending field named |
| Server error (5xx) | Same rollback path; generic message, traceback stays in the server log |
| Network down during a write | Change stays on screen, queues in the outbox, pill shows the count, auto-flushes on reconnect |
| Event stream drops | Backoff reconnect (1s, ×1.8, capped at 15s), then full resync |
| Client too slow to drain its queue | Server drops that subscriber rather than blocking writers; client sees a `seq` gap on the next frame and resyncs |
| Server restarts | Indistinguishable from a stream drop; verified end-to-end |
| Two tabs edit the same task | Last write wins **per field** — `PATCH` only writes the keys it is given, so edits to different fields both survive |
| SQLite lock contention | `busy_timeout=5000` plus a single-writer lock; WAL keeps readers unblocked |
| Malformed / oversized request | 400 and 413 respectively, before any handler runs |
| Path traversal in a static URL | Normalized and rejected with 403; covered by a test |

## 13. Testing strategy

**21 end-to-end tests** (`tests/test_api.py`) run against a real
`ThreadingHTTPServer` on an ephemeral port with a real SQLite file — no mocks,
no fakes. They cover task and project lifecycles, soft delete and restore,
reordering, tag normalization, every validation branch, SSE broadcast including
the origin tag, and the static layer (SPA deep links, missing assets, traversal).

What that suite deliberately does not cover is the browser: there is no driver
available, so the UI was verified by driving a real browser manually — every
view, both themes, mobile width, drag-to-reorder, the palette, dialogs, offline
behavior with a stubbed `fetch`, and two-tab live sync.

That manual pass earned its keep. It found five real defects, none of which the
server-side suite could have caught:

1. `[hidden]` silently overridden by `display:flex`, so the command palette
   rendered over the app on first load.
2. `requestAnimationFrame` paused in background tabs, stranding live updates.
3. The outbox dropping the tail of a batch after the first failed write.
4. The connection indicator reporting "Live sync on" while writes were queued.
5. `seq` published but unused, so a dropped stream left a tab stale forever.

The honest read: the automated suite proves the API is correct, and the parts of
this app that are hardest to get right — optimistic state, reconnection, render
scheduling — are exactly the parts it does not reach. Browser-level tests are
the first thing this project should gain.

## 14. Known limits and next steps

**Limits, in rough order of how much they'd matter**

- Single user, no auth. The data model has no owner column; adding one touches
  every query.
- One server process. The SSE fan-out is in-memory, so a second process would
  not see the first's events.
- Conflict resolution is last-write-wins per field. Simultaneous edits to the
  same field lose one silently.
- `/api/state` returns everything. Fine for thousands of tasks, wrong for
  hundreds of thousands — that boundary needs pagination and incremental sync.
- No browser-side persistence: a reload while offline loses the outbox.
- Reordering renumbers the whole visible list rather than inserting fractionally.

**Next steps, in the order I would do them**

1. Browser-level tests for the optimistic, offline and reconnect paths.
2. `Last-Event-ID` replay so reconnects are exact instead of resyncing.
3. Accounts and per-user scoping, then TLS, before this is exposed anywhere.
4. IndexedDB mirror of the outbox for real offline durability.
5. Recurring tasks and subtasks — the schema additions are small; the
   interaction design is not.
