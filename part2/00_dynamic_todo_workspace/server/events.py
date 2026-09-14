"""In-process pub/sub for Server-Sent Events.

Every mutation publishes an event; connected browsers apply it live, so two
tabs (or two devices on the LAN) stay in sync without polling. Each event
carries the originating client id so the tab that made the change can ignore
the echo and keep its optimistic state.
"""

import json
import queue
import threading

_subscribers: set[queue.Queue] = set()
_lock = threading.Lock()
_seq = 0


def subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=256)
    with _lock:
        _subscribers.add(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    with _lock:
        _subscribers.discard(q)


def subscriber_count() -> int:
    with _lock:
        return len(_subscribers)


def publish(kind: str, payload, origin: str | None = None) -> None:
    global _seq
    with _lock:
        _seq += 1
        message = json.dumps(
            {"seq": _seq, "type": kind, "origin": origin, "data": payload},
            separators=(",", ":"))
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(message)
            except queue.Full:      # a stalled client must not block writers
                dead.append(q)
        for q in dead:
            _subscribers.discard(q)
