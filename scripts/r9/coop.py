"""r9 speed (audio lever U): the units of a stage run CONCURRENTLY so their GPU calls batch together, with the per-unit code unchanged.

Each unit's selection (Qwen candidates -> judge -> Chatterbox takes -> dry-run cost -> pace takes ...) is a chain of small GPU calls; run unit
by unit, every Qwen call decodes 1 sequence and every Chatterbox round 2-4 takes. Here every unit is a worker thread, but COOPERATIVE: exactly
one worker runs Python at any time (no races, the per-unit code and its shared stats need no locks), and a worker that needs the GPU parks its
request (`call(kind, payload)`). When no worker can run, the scheduler executes all parked requests of one kind as ONE batch (handlers[kind] gets
the payload list; kinds are tried in `order`), then resumes those workers in unit order. `map(fn, items)` inside a worker runs independent
sub-tasks (e.g. the judge calls of one unit) as child workers. The result is deterministic: the same inputs give the same batches.
GPU work only ever runs inside a handler (or in the one running worker), never concurrently.
    co = Coop({"qwen": run_qwen, "tts": run_tts}, order=("qwen", "tts")); results = co.map(unit_fn, units)
"""
from __future__ import annotations

import threading


class _W:
    __slots__ = ("fn", "item", "out", "err", "done", "parent", "left", "res", "th")

    def __init__(self, fn, item, parent):
        self.fn, self.item, self.parent = fn, item, parent; self.out = self.err = self.res = None; self.done = False; self.left = 0


class Coop:
    def __init__(self, handlers: dict, order=None, wrap=None):
        self.h = handlers; self.order = list(order or handlers); self.wrap = wrap   # wrap(): context manager entered in every worker thread (torch grad mode)
        self.cv = threading.Condition(); self.turn = None; self.ready = []; self.pending = {k: [] for k in handlers}
        self.tl = threading.local(); self.batches = {k: [] for k in handlers}

    def current(self):
        return getattr(self.tl, "w", None)

    # ---- scheduler (always called with the lock held, by the thread that just gave up its turn)
    def _next(self):
        while True:
            if self.ready:
                self.turn = self.ready.pop(0); self.cv.notify_all(); return
            kind = next((k for k in self.order if self.pending[k]), None)
            if kind is None:
                self.turn = None; self.cv.notify_all(); return
            reqs = self.pending[kind]; self.pending[kind] = []; self.batches[kind].append(len(reqs))
            try:
                outs = self.h[kind]([p for _, p in reqs])
                assert len(outs) == len(reqs), f"handler {kind}: {len(outs)} results for {len(reqs)} requests"
                for (w, _), o in zip(reqs, outs): w.res = ("ok", o)
            except BaseException as e:   # the whole batch failed: every caller sees the exception
                for w, _ in reqs: w.res = ("err", e)
            self.ready += [w for w, _ in reqs]

    def _wait_turn(self, w):
        while self.turn is not w: self.cv.wait()

    def _body(self, w):
        self.tl.w = w
        with self.cv: self._wait_turn(w)
        try:
            if self.wrap:
                with self.wrap(): w.out = w.fn(w.item)
            else:
                w.out = w.fn(w.item)
        except BaseException as e:
            w.err = e
        with self.cv:
            w.done = True
            if w.parent is not None:
                w.parent.left -= 1
                if w.parent.left == 0: self.ready.append(w.parent)
            self._next()

    def call(self, kind: str, payload):
        w = self.current()
        if w is None:   # not inside a worker (no concurrency): run it alone
            return self.h[kind]([payload])[0]
        with self.cv:
            self.pending[kind].append((w, payload)); self._next(); self._wait_turn(w)
            st, v = w.res; w.res = None
        if st == "err": raise v
        return v

    def map(self, fn, items):
        items = list(items); parent = self.current()
        ws = [_W(fn, it, parent) for it in items]
        if not ws: return []
        for w in ws:
            w.th = threading.Thread(target=self._body, args=(w,), daemon=True); w.th.start()
        with self.cv:
            if parent is not None:   # children run, the parent parks until the last one is done
                parent.left = len(ws); self.ready += ws; self._next(); self._wait_turn(parent)
            else:
                assert self.turn is None, "Coop.map from outside a worker while workers run"
                self.ready += ws; self._next()
                while not all(w.done for w in ws) or self.turn is not None: self.cv.wait()
        for w in ws: w.th.join()
        for w in ws:
            if w.err is not None: raise w.err
        return [w.out for w in ws]


def qwen_handler(payloads):
    """[(llm, msgs, max_tokens)] -> greedy answers: batched on the resident model (DUB_QWEN_BATCH=1, qwen_batch.py), else one by one"""
    import os
    if os.environ.get("DUB_QWEN_BATCH", "0") == "1":
        import qwen_batch
        out = [None] * len(payloads); by = {}
        for k, (l_, _, _) in enumerate(payloads): by.setdefault(id(l_), (l_, []))[1].append(k)
        for l_, ks in by.values():
            for k, t in zip(ks, qwen_batch.engine(l_).chat([(payloads[k][1], payloads[k][2]) for k in ks])): out[k] = t
        return out
    return [l_.create_chat_completion(ms, max_tokens=mt, temperature=0.0)["choices"][0]["message"]["content"] for l_, ms, mt in payloads]


def torch_wrap():
    """the caller's torch grad / inference mode (thread-local in torch) for every worker thread"""
    import contextlib
    import torch
    gi, inf = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
    def wrap():
        st = contextlib.ExitStack(); st.enter_context(torch.inference_mode(inf)); st.enter_context(torch.set_grad_enabled(gi)); return st
    return wrap
