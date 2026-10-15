"""Per-run request cache with single-flight semantics.

Several agents need the same feed payload (momentum, volume, risk and behaviour all read the
price series). Without a cache each one pays the network latency; with it, the first caller
fetches and concurrent callers wait for that single in-flight request.
"""
from __future__ import annotations

import threading


class _Entry:
    __slots__ = ("event", "value", "exc")

    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.exc = None


class RunCache:
    def __init__(self):
        self._d: dict = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key, fn):
        with self._lock:
            e = self._d.get(key)
            owner = e is None
            if owner:
                e = self._d[key] = _Entry()
                self.misses += 1
            else:
                self.hits += 1
        if owner:
            try:
                e.value = fn()
            except BaseException as ex:          # cache failures too, so waiters fail fast
                e.exc = ex
            finally:
                e.event.set()
        else:
            e.event.wait()
        if e.exc is not None:
            raise e.exc
        return e.value
