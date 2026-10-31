"""r9: GPU-busy accounting. `with timeline.mark("qwen"):` appends [category, t0, t1] (wall clock) to $R9_TIMELINE when it is set (no-op otherwise);
scripts/r9/gpu_busy.py then integrates the 100-ms NVML utilization samples over each category's intervals (the innermost open one wins)."""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time

_LOCK = threading.Lock()


@contextlib.contextmanager
def mark(cat: str):
    path = os.environ.get("R9_TIMELINE")
    if not path:
        yield; return
    t0 = time.time()
    try:
        yield
    finally:
        with _LOCK, open(path, "a") as f:
            f.write(json.dumps([cat, round(t0, 3), round(time.time(), 3)]) + "\n")


def wrap(obj, name: str, cat: str):
    """obj.name(...) runs inside mark(cat)"""
    f = getattr(obj, name)
    def g(*a, **k):
        with mark(cat): return f(*a, **k)
    setattr(obj, name, g)
