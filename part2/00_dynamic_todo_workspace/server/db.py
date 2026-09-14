"""SQLite persistence layer.

One connection per thread (the HTTP server is threaded), WAL mode so readers
never block the writer, and foreign keys on so cascades actually happen.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "TODO_DB",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "todo.db"),
)

_local = threading.local()
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    color       TEXT NOT NULL DEFAULT 'violet',
    position    REAL NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id           TEXT PRIMARY KEY,
    project_id   TEXT REFERENCES projects(id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    done         INTEGER NOT NULL DEFAULT 0,
    priority     INTEGER NOT NULL DEFAULT 4,   -- 1 urgent .. 4 none
    due_at       TEXT,                          -- ISO-8601, UTC
    all_day      INTEGER NOT NULL DEFAULT 1,
    position     REAL NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    completed_at TEXT,
    deleted_at   TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE IF NOT EXISTS task_tags (
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    tag_id  TEXT NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
    PRIMARY KEY (task_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id, done, position);
CREATE INDEX IF NOT EXISTS idx_tasks_due     ON tasks(due_at) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_live    ON tasks(deleted_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


def init(seed: bool = True) -> None:
    conn = connect()
    with _write_lock:
        conn.executescript(SCHEMA)
    if seed and conn.execute("SELECT COUNT(*) c FROM projects").fetchone()["c"] == 0:
        _seed(conn)


def reset() -> None:
    """Drop everything. Used by tests."""
    conn = connect()
    with _write_lock:
        conn.executescript(
            "DROP TABLE IF EXISTS task_tags; DROP TABLE IF EXISTS tasks;"
            "DROP TABLE IF EXISTS tags; DROP TABLE IF EXISTS projects;"
        )
    init(seed=False)


def close() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #

def project_row(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "name": r["name"], "color": r["color"],
            "position": r["position"], "createdAt": r["created_at"]}


def task_row(r: sqlite3.Row, tags: list[str]) -> dict:
    return {
        "id": r["id"],
        "projectId": r["project_id"],
        "title": r["title"],
        "notes": r["notes"],
        "done": bool(r["done"]),
        "priority": r["priority"],
        "dueAt": r["due_at"],
        "allDay": bool(r["all_day"]),
        "position": r["position"],
        "tags": tags,
        "createdAt": r["created_at"],
        "updatedAt": r["updated_at"],
        "completedAt": r["completed_at"],
    }


def _tag_map(conn, task_ids=None) -> dict[str, list[str]]:
    q = ("SELECT tt.task_id, t.name FROM task_tags tt "
         "JOIN tags t ON t.id = tt.tag_id ORDER BY t.name")
    out: dict[str, list[str]] = {}
    for row in conn.execute(q):
        out.setdefault(row["task_id"], []).append(row["name"])
    return out


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #

def snapshot() -> dict:
    """Everything the client needs to boot, in one round trip."""
    conn = connect()
    tags = _tag_map(conn)
    tasks = [task_row(r, tags.get(r["id"], []))
             for r in conn.execute(
                 "SELECT * FROM tasks WHERE deleted_at IS NULL ORDER BY position")]
    projects = [project_row(r)
                for r in conn.execute("SELECT * FROM projects ORDER BY position")]
    all_tags = [r["name"] for r in conn.execute("SELECT name FROM tags ORDER BY name")]
    return {"projects": projects, "tasks": tasks, "tags": all_tags, "serverTime": now_iso()}


def get_task(task_id: str, include_deleted: bool = False) -> dict | None:
    conn = connect()
    sql = "SELECT * FROM tasks WHERE id = ?"
    if not include_deleted:
        sql += " AND deleted_at IS NULL"
    r = conn.execute(sql, (task_id,)).fetchone()
    if r is None:
        return None
    tags = [x["name"] for x in conn.execute(
        "SELECT t.name FROM task_tags tt JOIN tags t ON t.id = tt.tag_id "
        "WHERE tt.task_id = ? ORDER BY t.name", (task_id,))]
    return task_row(r, tags)


def get_project(project_id: str) -> dict | None:
    r = connect().execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return project_row(r) if r else None


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #

def _next_position(conn, project_id) -> float:
    row = conn.execute(
        "SELECT MIN(position) p FROM tasks WHERE deleted_at IS NULL AND "
        "(project_id IS ? OR ? IS NULL)", (project_id, project_id)).fetchone()
    return (row["p"] or 0) - 1024.0


def _sync_tags(conn, task_id: str, names: list[str]) -> None:
    clean, seen = [], set()
    for raw in names or []:
        name = str(raw).strip().lstrip("#")[:32]
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            clean.append(name)
    conn.execute("DELETE FROM task_tags WHERE task_id = ?", (task_id,))
    for name in clean:
        row = conn.execute("SELECT id FROM tags WHERE name = ? COLLATE NOCASE",
                           (name,)).fetchone()
        tag_id = row["id"] if row else new_id("tag")
        if not row:
            conn.execute("INSERT INTO tags (id, name) VALUES (?, ?)", (tag_id, name))
        conn.execute("INSERT OR IGNORE INTO task_tags (task_id, tag_id) VALUES (?, ?)",
                     (task_id, tag_id))


def create_task(data: dict) -> dict:
    conn = connect()
    ts = now_iso()
    tid = new_id("tsk")
    with _write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            pos = data.get("position")
            if pos is None:
                pos = _next_position(conn, data.get("projectId"))
            conn.execute(
                "INSERT INTO tasks (id, project_id, title, notes, done, priority, due_at,"
                " all_day, position, created_at, updated_at, completed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (tid, data.get("projectId"), data["title"].strip()[:500],
                 (data.get("notes") or "")[:20000], 1 if data.get("done") else 0,
                 int(data.get("priority") or 4), data.get("dueAt"),
                 1 if data.get("allDay", True) else 0, float(pos), ts, ts,
                 ts if data.get("done") else None))
            _sync_tags(conn, tid, data.get("tags") or [])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return get_task(tid)


_FIELDS = {
    "title": ("title", lambda v: str(v).strip()[:500]),
    "notes": ("notes", lambda v: str(v)[:20000]),
    "priority": ("priority", lambda v: max(1, min(4, int(v)))),
    "dueAt": ("due_at", lambda v: v or None),
    "allDay": ("all_day", lambda v: 1 if v else 0),
    "projectId": ("project_id", lambda v: v or None),
    "position": ("position", float),
}


def update_task(task_id: str, patch: dict) -> dict | None:
    conn = connect()
    if get_task(task_id) is None:
        return None
    sets, args = [], []
    for key, (col, cast) in _FIELDS.items():
        if key in patch:
            sets.append(f"{col} = ?")
            args.append(cast(patch[key]))
    if "done" in patch:
        done = bool(patch["done"])
        sets += ["done = ?", "completed_at = ?"]
        args += [1 if done else 0, now_iso() if done else None]
    sets.append("updated_at = ?")
    args.append(now_iso())
    args.append(task_id)
    with _write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", args)
            if "tags" in patch:
                _sync_tags(conn, task_id, patch["tags"])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return get_task(task_id)


def delete_task(task_id: str) -> dict | None:
    """Soft delete so the client can offer undo."""
    task = get_task(task_id)
    if task is None:
        return None
    with _write_lock:
        connect().execute("UPDATE tasks SET deleted_at = ?, updated_at = ? WHERE id = ?",
                          (now_iso(), now_iso(), task_id))
    return task


def restore_task(task_id: str) -> dict | None:
    conn = connect()
    if conn.execute("SELECT 1 FROM tasks WHERE id = ? AND deleted_at IS NOT NULL",
                    (task_id,)).fetchone() is None:
        return None
    with _write_lock:
        conn.execute("UPDATE tasks SET deleted_at = NULL, updated_at = ? WHERE id = ?",
                     (now_iso(), task_id))
    return get_task(task_id)


def purge_deleted(older_than_days: int = 30) -> int:
    cutoff = time.time() - older_than_days * 86400
    cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    with _write_lock:
        cur = connect().execute(
            "DELETE FROM tasks WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff_iso,))
    return cur.rowcount


def reorder(ids: list[str]) -> list[dict]:
    """Assign evenly spaced positions in the given order."""
    conn = connect()
    ts = now_iso()
    with _write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for i, tid in enumerate(ids):
                conn.execute("UPDATE tasks SET position = ?, updated_at = ? WHERE id = ?",
                             (float(i * 1024), ts, tid))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return [t for t in (get_task(i) for i in ids) if t]


def clear_completed(project_id: str | None = None) -> list[str]:
    conn = connect()
    sql = "SELECT id FROM tasks WHERE done = 1 AND deleted_at IS NULL"
    args: list = []
    if project_id:
        sql += " AND project_id = ?"
        args.append(project_id)
    ids = [r["id"] for r in conn.execute(sql, args)]
    ts = now_iso()
    with _write_lock:
        conn.executemany("UPDATE tasks SET deleted_at = ?, updated_at = ? WHERE id = ?",
                         [(ts, ts, i) for i in ids])
    return ids


# --------------------------------------------------------------------------- #
# projects
# --------------------------------------------------------------------------- #

def create_project(data: dict) -> dict:
    conn = connect()
    pid = new_id("prj")
    row = conn.execute("SELECT MAX(position) p FROM projects").fetchone()
    pos = (row["p"] or 0) + 1024.0
    with _write_lock:
        conn.execute("INSERT INTO projects (id, name, color, position, created_at)"
                     " VALUES (?,?,?,?,?)",
                     (pid, str(data.get("name", "Untitled")).strip()[:80] or "Untitled",
                      data.get("color") or "violet", pos, now_iso()))
    return get_project(pid)


def update_project(pid: str, patch: dict) -> dict | None:
    if get_project(pid) is None:
        return None
    sets, args = [], []
    if "name" in patch:
        sets.append("name = ?"); args.append(str(patch["name"]).strip()[:80] or "Untitled")
    if "color" in patch:
        sets.append("color = ?"); args.append(patch["color"])
    if "position" in patch:
        sets.append("position = ?"); args.append(float(patch["position"]))
    if sets:
        args.append(pid)
        with _write_lock:
            connect().execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", args)
    return get_project(pid)


def delete_project(pid: str) -> bool:
    conn = connect()
    if get_project(pid) is None:
        return False
    with _write_lock:
        conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
    return True


# --------------------------------------------------------------------------- #
# seed
# --------------------------------------------------------------------------- #

def _seed(conn) -> None:
    from datetime import timedelta
    # Anchored to the machine's local time: the browser rendering these is on the
    # same machine, so "today at 6pm" means what it says. All-day items sit at
    # local noon, which lands on the intended date in any nearby timezone.
    today = datetime.now().astimezone()

    def at(days, hour=None):
        d = (today + timedelta(days=days)).replace(
            hour=12 if hour is None else hour, minute=0, second=0, microsecond=0)
        return d.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    inbox = create_project({"name": "Personal", "color": "violet"})
    work = create_project({"name": "Work", "color": "blue"})
    create_project({"name": "Reading list", "color": "amber"})

    demo = [
        ("Try the command palette — press ⌘K", None, 4, None, ["tips"], False, True),
        ("Type a due date right into the title: “review deck tomorrow 3pm !p1”",
         None, 4, None, ["tips"], False, True),
        ("Water the plants", inbox["id"], 3, at(0, 18), ["home"], False, False),
        ("Book dentist appointment", inbox["id"], 2, at(1, 9), ["health"], False, False),
        ("Renew passport", inbox["id"], 1, at(-2), ["admin"], False, True),
        ("Draft Q3 retrospective", work["id"], 2, at(0, 15), ["writing"], False, False),
        ("Review PR #418 — sync engine", work["id"], 1, at(2, 11), ["code"], False, False),
        ("Ship the onboarding email", work["id"], 3, at(5), ["growth"], False, True),
        ("Pay the internet bill", inbox["id"], 4, at(-4, 9), ["admin"], True, False),
        ("Plan weekend hike", inbox["id"], 4, at(6), ["fun"], False, True),
    ]
    for title, pid, prio, due, tags, done, allday in demo:
        create_task({"title": title, "projectId": pid, "priority": prio, "dueAt": due,
                     "tags": tags, "done": done, "allDay": allday})
