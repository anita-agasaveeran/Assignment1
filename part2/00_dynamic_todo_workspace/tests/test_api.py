"""End-to-end tests: a real HTTP server, a real SQLite file, real requests.

    python3 tests/test_api.py          # or: python3 -m unittest discover tests
"""

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TMP = tempfile.mkdtemp(prefix="todo-test-")
os.environ["TODO_DB"] = os.path.join(TMP, "test.db")
os.environ["TODO_QUIET"] = "1"

from server import db, events           # noqa: E402
from server.app import Handler          # noqa: E402

db.DB_PATH = os.environ["TODO_DB"]


class Client:
    def __init__(self, base):
        self.base = base

    def __call__(self, method, path, body=None, client_id="test-client"):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("X-Client-Id", client_id)
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                raw = res.read()
                return res.status, (json.loads(raw) if raw and
                                    "json" in res.headers.get("Content-Type", "") else raw)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except json.JSONDecodeError:
                return e.code, raw


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init(seed=False)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        cls.server._shutting_down = False
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.req = Client(f"http://127.0.0.1:{cls.server.server_address[1]}")

    @classmethod
    def tearDownClass(cls):
        cls.server._shutting_down = True
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        db.reset()

    # ── basics ──────────────────────────────────────────────────────────
    def test_health(self):
        status, body = self.req("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    def test_empty_state_shape(self):
        status, body = self.req("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(body["tasks"], [])
        self.assertEqual(body["projects"], [])
        self.assertIn("serverTime", body)

    # ── task lifecycle ──────────────────────────────────────────────────
    def test_create_and_fetch(self):
        status, task = self.req("POST", "/api/tasks",
                                {"title": "  Write the spec  ", "priority": 2,
                                 "dueAt": "2030-05-04T09:00:00Z", "allDay": False,
                                 "tags": ["Work", "work", "  writing "]})
        self.assertEqual(status, 201)
        self.assertEqual(task["title"], "Write the spec")
        self.assertEqual(task["priority"], 2)
        self.assertFalse(task["allDay"])
        self.assertEqual(task["tags"], ["Work", "writing"])   # deduped, trimmed
        self.assertFalse(task["done"])

        status, fetched = self.req("GET", f"/api/tasks/{task['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["id"], task["id"])

    def test_update_marks_completed_at(self):
        _, task = self.req("POST", "/api/tasks", {"title": "Ship it"})
        status, updated = self.req("PATCH", f"/api/tasks/{task['id']}", {"done": True})
        self.assertEqual(status, 200)
        self.assertTrue(updated["done"])
        self.assertIsNotNone(updated["completedAt"])

        _, reopened = self.req("PATCH", f"/api/tasks/{task['id']}", {"done": False})
        self.assertIsNone(reopened["completedAt"])

    def test_partial_update_leaves_other_fields(self):
        _, task = self.req("POST", "/api/tasks",
                           {"title": "Keep me", "notes": "context", "priority": 1,
                            "tags": ["a"]})
        _, updated = self.req("PATCH", f"/api/tasks/{task['id']}", {"title": "Renamed"})
        self.assertEqual(updated["title"], "Renamed")
        self.assertEqual(updated["notes"], "context")
        self.assertEqual(updated["priority"], 1)
        self.assertEqual(updated["tags"], ["a"])

    def test_delete_is_soft_and_restorable(self):
        _, task = self.req("POST", "/api/tasks", {"title": "Oops"})
        status, _ = self.req("DELETE", f"/api/tasks/{task['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(self.req("GET", f"/api/tasks/{task['id']}")[0], 404)
        self.assertEqual(self.req("GET", "/api/state")[1]["tasks"], [])

        status, restored = self.req("POST", f"/api/tasks/{task['id']}/restore", {})
        self.assertEqual(status, 200)
        self.assertEqual(restored["title"], "Oops")
        self.assertEqual(len(self.req("GET", "/api/state")[1]["tasks"]), 1)

    def test_restore_unknown_is_404(self):
        self.assertEqual(self.req("POST", "/api/tasks/tsk_nope/restore", {})[0], 404)

    def test_reorder(self):
        ids = [self.req("POST", "/api/tasks", {"title": f"T{i}"})[1]["id"] for i in range(4)]
        shuffled = [ids[2], ids[0], ids[3], ids[1]]
        status, _ = self.req("POST", "/api/tasks/reorder", {"ids": shuffled})
        self.assertEqual(status, 200)
        order = [t["id"] for t in self.req("GET", "/api/state")[1]["tasks"]]
        self.assertEqual(order, shuffled)

    def test_clear_completed_only_touches_done(self):
        a = self.req("POST", "/api/tasks", {"title": "done", "done": True})[1]
        b = self.req("POST", "/api/tasks", {"title": "open"})[1]
        status, body = self.req("POST", "/api/tasks/clear-completed", {})
        self.assertEqual(status, 200)
        self.assertEqual(body["ids"], [a["id"]])
        remaining = [t["id"] for t in self.req("GET", "/api/state")[1]["tasks"]]
        self.assertEqual(remaining, [b["id"]])

    # ── validation ──────────────────────────────────────────────────────
    def test_rejects_empty_title(self):
        status, body = self.req("POST", "/api/tasks", {"title": "   "})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["field"], "title")

    def test_rejects_bad_priority_and_date(self):
        self.assertEqual(self.req("POST", "/api/tasks",
                                  {"title": "x", "priority": 9})[0], 422)
        self.assertEqual(self.req("POST", "/api/tasks",
                                  {"title": "x", "dueAt": "next tuesday"})[0], 422)

    def test_rejects_unknown_project(self):
        status, body = self.req("POST", "/api/tasks", {"title": "x", "projectId": "prj_ghost"})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["field"], "projectId")

    def test_rejects_malformed_json(self):
        req = urllib.request.Request(self.req.base + "/api/tasks", data=b"{not json",
                                     method="POST")
        req.add_header("Content-Type", "application/json")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_route_is_404(self):
        self.assertEqual(self.req("GET", "/api/nope")[0], 404)
        self.assertEqual(self.req("GET", "/api/tasks/tsk_missing")[0], 404)

    # ── projects ────────────────────────────────────────────────────────
    def test_project_crud_and_cascade(self):
        status, project = self.req("POST", "/api/projects", {"name": "Home", "color": "teal"})
        self.assertEqual(status, 201)
        _, task = self.req("POST", "/api/tasks",
                           {"title": "Fix sink", "projectId": project["id"]})

        _, renamed = self.req("PATCH", f"/api/projects/{project['id']}", {"name": "House"})
        self.assertEqual(renamed["name"], "House")

        self.assertEqual(self.req("DELETE", f"/api/projects/{project['id']}")[0], 200)
        state = self.req("GET", "/api/state")[1]
        self.assertEqual(state["projects"], [])
        self.assertEqual(state["tasks"], [], "tasks in a deleted list should cascade")

    def test_project_requires_name(self):
        self.assertEqual(self.req("POST", "/api/projects", {"name": " "})[0], 422)

    # ── live events ─────────────────────────────────────────────────────
    def test_sse_broadcasts_mutations(self):
        received = []
        ready = threading.Event()

        def listen():
            with urllib.request.urlopen(self.req.base + "/api/events", timeout=6) as res:
                ready.set()
                for raw in res:
                    line = raw.decode().strip()
                    if line.startswith("data:"):
                        received.append(json.loads(line[5:]))
                        return

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        ready.wait(3)
        for _ in range(40):                       # wait for the subscription to register
            if events.subscriber_count():
                break
            threading.Event().wait(0.05)

        self.req("POST", "/api/tasks", {"title": "Broadcast me"}, client_id="tab-1")
        t.join(4)
        self.assertTrue(received, "expected an SSE frame")
        self.assertEqual(received[0]["type"], "task.created")
        self.assertEqual(received[0]["origin"], "tab-1")
        self.assertEqual(received[0]["data"]["title"], "Broadcast me")

    # ── static hosting ──────────────────────────────────────────────────
    def test_serves_spa(self):
        status, body = self.req("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Tasks</title>", body)

    def test_deep_link_falls_back_to_index(self):
        self.assertEqual(self.req("GET", "/some/deep/link")[0], 200)

    def test_missing_asset_is_404(self):
        self.assertEqual(self.req("GET", "/js/nope.js")[0], 404)

    def test_blocks_path_traversal(self):
        status, _ = self.req("GET", "/../server/app.py")
        self.assertIn(status, (403, 404))


if __name__ == "__main__":
    unittest.main(verbosity=2)
