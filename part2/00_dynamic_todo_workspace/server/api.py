"""REST routes. Pure functions of (request) -> (status, body); the HTTP layer
in app.py handles transport concerns only."""

import json
import re

from . import db, events


class ApiError(Exception):
    def __init__(self, status: int, message: str, field: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.field = field


ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


def _validate_task(data: dict, partial: bool = False) -> None:
    if not partial or "title" in data:
        title = str(data.get("title", "")).strip()
        if not title:
            raise ApiError(422, "Title can't be empty.", "title")
        if len(title) > 500:
            raise ApiError(422, "Title is too long (500 characters max).", "title")
    if data.get("dueAt") is not None and "dueAt" in data:
        if not ISO_RE.match(str(data["dueAt"])):
            raise ApiError(422, "dueAt must be an ISO-8601 timestamp.", "dueAt")
    if "priority" in data and data["priority"] is not None:
        try:
            p = int(data["priority"])
        except (TypeError, ValueError):
            raise ApiError(422, "priority must be 1-4.", "priority")
        if not 1 <= p <= 4:
            raise ApiError(422, "priority must be 1-4.", "priority")
    if "projectId" in data and data["projectId"]:
        if db.get_project(data["projectId"]) is None:
            raise ApiError(422, "That list no longer exists.", "projectId")
    if "tags" in data and not isinstance(data["tags"], list):
        raise ApiError(422, "tags must be a list of strings.", "tags")


# --------------------------------------------------------------------------- #

def route(method: str, path: str, body: dict, query: dict, origin: str | None):
    parts = [p for p in path.strip("/").split("/") if p]      # e.g. ['api','tasks','id']
    if not parts or parts[0] != "api":
        raise ApiError(404, "Not found.")
    parts = parts[1:]

    if parts == ["health"] and method == "GET":
        return 200, {"ok": True, "listeners": events.subscriber_count(),
                     "time": db.now_iso()}

    if parts == ["state"] and method == "GET":
        return 200, db.snapshot()

    if parts == ["export"] and method == "GET":
        return 200, {"exportedAt": db.now_iso(), **db.snapshot()}

    # ---- tasks ----------------------------------------------------------- #
    if parts == ["tasks"]:
        if method == "GET":
            return 200, {"tasks": db.snapshot()["tasks"]}
        if method == "POST":
            _validate_task(body)
            task = db.create_task(body)
            events.publish("task.created", task, origin)
            return 201, task

    if parts == ["tasks", "reorder"] and method == "POST":
        ids = body.get("ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ApiError(422, "ids must be a list of task ids.", "ids")
        tasks = db.reorder(ids)
        events.publish("tasks.reordered", {"ids": ids}, origin)
        return 200, {"tasks": tasks}

    if parts == ["tasks", "clear-completed"] and method == "POST":
        ids = db.clear_completed(body.get("projectId"))
        events.publish("tasks.deleted", {"ids": ids}, origin)
        return 200, {"ids": ids}

    if len(parts) == 2 and parts[0] == "tasks":
        tid = parts[1]
        if method == "GET":
            task = db.get_task(tid)
            if not task:
                raise ApiError(404, "That task no longer exists.")
            return 200, task
        if method in ("PATCH", "PUT"):
            _validate_task(body, partial=True)
            task = db.update_task(tid, body)
            if not task:
                raise ApiError(404, "That task no longer exists.")
            events.publish("task.updated", task, origin)
            return 200, task
        if method == "DELETE":
            task = db.delete_task(tid)
            if not task:
                raise ApiError(404, "That task no longer exists.")
            events.publish("tasks.deleted", {"ids": [tid]}, origin)
            return 200, {"id": tid}

    if len(parts) == 3 and parts[0] == "tasks" and parts[2] == "restore" and method == "POST":
        task = db.restore_task(parts[1])
        if not task:
            raise ApiError(404, "Nothing to restore.")
        events.publish("task.created", task, origin)
        return 200, task

    # ---- projects -------------------------------------------------------- #
    if parts == ["projects"]:
        if method == "GET":
            return 200, {"projects": db.snapshot()["projects"]}
        if method == "POST":
            if not str(body.get("name", "")).strip():
                raise ApiError(422, "Give the list a name.", "name")
            project = db.create_project(body)
            events.publish("project.created", project, origin)
            return 201, project

    if len(parts) == 2 and parts[0] == "projects":
        pid = parts[1]
        if method in ("PATCH", "PUT"):
            project = db.update_project(pid, body)
            if not project:
                raise ApiError(404, "That list no longer exists.")
            events.publish("project.updated", project, origin)
            return 200, project
        if method == "DELETE":
            if not db.delete_project(pid):
                raise ApiError(404, "That list no longer exists.")
            events.publish("project.deleted", {"id": pid}, origin)
            return 200, {"id": pid}

    raise ApiError(404, "Not found.")
