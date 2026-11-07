"""r15 T: the generator stage without whole-video frame buffers (DUB_GEN_STREAM=1).

A 5-min 720p job held ~45 GB in the generator (all master frames + all LatentSync replacement frames in RAM) and wrote 2 x 8.9 GB of raw
frame memmaps (--dump-ls-raw). Nothing about the generation changes here (same LatentSync calls, same inputs, same order); only where the
frames live:
  LazyFrames   the master frames of read_frames(p) as a sequence decoded on demand (the SAME decoder: read_frames = list(iter_frames(p))):
               forward access decodes sequentially into a sliding cache; frames below the low-water mark (release_before) or more than
               `back` frames behind the newest one are dropped; an access below the decoded range re-decodes from the start (rare: split
               segments). Iteration (for f in frames) is a separate sequential decode, one frame in memory.
  PatchFrames  exact frames stored as their difference box against the master frame of the same index (the mouth-only composite and the
               LatentSync paste-back change one face region): get(k, master) returns the identical full frame.
"""
from __future__ import annotations
import pathlib
import numpy as np


def _colorspace(f_):
    cs = getattr(f_, "colorspace", None)
    return "ITU709" if cs == 1 else ("ITU601" if cs in (5, 6) else ("ITU709" if f_.height >= 720 else "ITU601"))


def iter_frames(p):
    """the frames of face_aware_latentsync.read_frames(p), one at a time (read_frames is list(iter_frames(p)))"""
    import cv2, os
    if os.environ.get("LS_CMS_READ") == "1":   # r13: the master decoded with its OWN matrix (see read_frames)
        import av
        c = av.open(str(p))
        try:
            for f_ in c.decode(c.streams.video[0]):
                yield f_.reformat(format="bgr24", src_colorspace=_colorspace(f_), src_color_range="MPEG").to_ndarray()
        finally:
            c.close()
        return
    cap = cv2.VideoCapture(str(p))
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
    finally:
        cap.release()


def count_frames(p) -> int:
    """decoded frame count of p (one sequential decode; exact by construction)"""
    n = 0
    for _ in iter_frames(p): n += 1
    return n


class LazyFrames:
    def __init__(self, p, n: int | None = None, back: int = 1024):
        self.p = pathlib.Path(p); self.n = int(n) if n is not None else count_frames(self.p); self.back = int(back)
        self._it = None; self._next = 0; self._cache: dict = {}; self._lo = 0; self.redecodes = 0; self.decoded = 0; self.peak_cached = 0

    def __len__(self): return self.n

    def _restart(self):
        self._it = iter_frames(self.p); self._next = 0; self._cache.clear(); self.redecodes += 1

    def release_before(self, k: int):
        """frames < k are not needed any more (they can still be read: that re-decodes from the start)"""
        self._lo = max(self._lo, int(k))
        for j in [j for j in self._cache if j < self._lo]: del self._cache[j]

    def __getitem__(self, i):
        if isinstance(i, slice): return [self[j] for j in range(*i.indices(self.n))]
        i = int(i)
        if i < 0: i += self.n
        if not 0 <= i < self.n: raise IndexError(i)
        f = self._cache.get(i)
        if f is not None: return f
        if self._it is None or i < self._next:   # not decoded yet / already dropped: (re)start the decoder
            if i < self._lo: self._lo = 0          # an explicit backward read lowers the mark (correctness first)
            self._restart()
        keep = max(self._lo, i - self.back)
        while self._next <= i:
            try: fr = next(self._it)
            except StopIteration: raise RuntimeError(f"{self.p}: only {self._next} frames decoded, {self.n} expected")
            j = self._next; self._next += 1; self.decoded += 1
            if j >= keep: self._cache[j] = fr
        for j in [j for j in self._cache if j < keep]: del self._cache[j]
        self.peak_cached = max(self.peak_cached, len(self._cache))
        return self._cache[i]

    def __iter__(self):
        n = 0
        for f in iter_frames(self.p):
            n += 1; yield f
        if n != self.n: raise RuntimeError(f"{self.p}: {n} frames decoded, {self.n} expected")

    def stats(self) -> dict:
        return {"frames": self.n, "decoded": self.decoded, "redecodes": self.redecodes, "peak_cached_frames": self.peak_cached, "back": self.back}


class PatchFrames:
    """n = what len() reports (None: the number of stored frames); put(k, frame, master); get(k, master, default)"""

    def __init__(self, n: int | None = None):
        self.n = n; self.p: dict = {}; self.bytes = 0

    def put(self, k, frame, master):
        frame = np.asarray(frame)
        if master is None or frame.shape != np.asarray(master).shape or frame.dtype != np.asarray(master).dtype:
            self.p[k] = ("full", np.array(frame, copy=True)); self.bytes += frame.nbytes; return
        d = frame != master
        d = d.any(2) if d.ndim == 3 else d
        if not d.any():
            self.p[k] = None; return
        ys = np.flatnonzero(d.any(1)); xs = np.flatnonzero(d.any(0))
        y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
        a = np.ascontiguousarray(frame[y0:y1, x0:x1]); self.p[k] = (y0, y1, x0, x1, a); self.bytes += a.nbytes

    def get(self, k, master, default=None):
        if k not in self.p: return default
        e = self.p[k]
        if e is None: return master
        if e[0] == "full": return e[1]
        y0, y1, x0, x1, a = e; f = np.array(master, copy=True); f[y0:y1, x0:x1] = a; return f

    def __contains__(self, k): return k in self.p

    def __len__(self): return self.n if self.n is not None else len(self.p)

    def close(self): self.p.clear()
