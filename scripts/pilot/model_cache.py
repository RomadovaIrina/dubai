"""Resident-worker model cache (2026-10-13, pilot 0.4 "держим резидентно и экономим на перезагрузках").

DUB_RESIDENT=1: every GPU model of the pipeline is created ONCE per process and reused by the following jobs (run_worker.py runs several
videos in one process, as one worker per GPU node does in production, TZ 3.4); the runner's free_cuda() calls only drop local references.
Qwen is NOT cached (DUB_RESIDENT_KEYS below): the CUDA llama.cpp build loads the 7.5 GiB GGUF from the page cache in ~2 s (pilot 0.9 addendum), and releasing it
before LatentSync keeps the video-stage peak at ~24-26 GiB instead of 29-31 GiB.
DUB_RESIDENT unset: get() calls the factory every time (the frozen per-run behaviour)."""
from __future__ import annotations
import os

_CACHE: dict = {}


def resident() -> bool:
    return os.environ.get("DUB_RESIDENT") == "1"


# what stays loaded: slow to load AND small (measured 2026-10-13). Whisper ASR (3 s, 1.8 GiB) and the TTS QA models (2.6 s, 3.4 GiB) are
# reloaded per job: holding them too would put the LatentSync peak at ~32 GiB on the 32 GiB card (all-resident probe + 03 peak).
DEFAULT_KEYS = "chatterbox_v3,latentsync,router,pyannote,demucs,codeformer,silero_vad"


def _name(key) -> str:
    return key[0] if isinstance(key, tuple) else str(key)


def get(key, factory):
    keys = set(os.environ.get("DUB_RESIDENT_KEYS", DEFAULT_KEYS).split(","))
    if not resident() or _name(key) not in keys:
        return factory()
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def cached(name: str) -> bool:
    """True when an object with this key name is held by the cache (the caller must then NOT close / free it)."""
    return any(_name(k) == name for k in _CACHE)


def loaded() -> list:
    return [str(k) for k in _CACHE]
