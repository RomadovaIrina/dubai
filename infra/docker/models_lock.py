#!/usr/bin/env python3
"""DabAI image (r15 T): the weight list of the models volume = every file of the commercially clean set, with its size and sha256.
    python infra/docker/models_lock.py make  --models DIR --out infra/docker/models.lock.json      (on the reference server)
    python infra/docker/models_lock.py check --models DIR [--deep]                                 (check.sh: sizes; --deep: sha256 too)
Exit 0 = every listed file present (and matching); 1 = something missing / different (printed)."""
import argparse, hashlib, json, os, pathlib, sys

LOCK = pathlib.Path(__file__).resolve().parent / "models.lock.json"
# models/<dir> -> (licence, what uses it); "student" is our own LoRA (mode fast), "cache/*" are runtime caches of gated / lazy downloads
SET = {
    "whisper-large-v3": ("MIT", "ASR (6.1), subtitle timing checks"), "qwen2.5-14b-instruct-gguf": ("Apache-2.0", "translation (6.4)"),
    "chatterbox": ("MIT", "TTS (7.2)"), "latentsync": ("OpenRAIL++ (user-approved)", "lip-sync (8.1) + SyncNet"),
    "sd-vae-ft-mse": ("MIT", "LatentSync VAE"), "wav2vec2-base-960h": ("Apache-2.0", "CTC word timing (9.2), TTS QA"),
    "mediapipe": ("Apache-2.0", "faces / head pose (5.4)"), "realesrgan": ("BSD-3-Clause", "composite layer SR"),
    "spleeter": ("MIT", "background bed"), "cache/pyannote": ("MIT (gated)", "diarization (6.3)"), "cache/pkuseg": ("MIT", "Chatterbox zh segmenter"),
    "student": ("own (LatentSync OpenRAIL++ derivative)", "mode fast"),
}
OPTIONAL = {"cache/pkuseg", "student"}
EXCLUDE = {"latentsync/stable_syncnet.pt"}   # training / metric only (distillation, sync_speech), not fetched for the worker
EXCLUDE_PREFIX = ("chatterbox/models--",)      # a stray HF cache copy inside the local snapshot


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["make", "check"]); ap.add_argument("--models", required=True)
    ap.add_argument("--out", default=str(LOCK)); ap.add_argument("--deep", action="store_true"); a = ap.parse_args(); M = pathlib.Path(a.models)
    if a.cmd == "make":
        rows = []
        for d, (lic, use) in SET.items():
            base = M / d
            if not base.exists(): print("absent on this server:", d); continue
            fs = [pathlib.Path(r_) / n for r_, _, ns in os.walk(base, followlinks=True) for n in ns]   # the set dirs may be symlinks (a clean-room view)
            for f in sorted(x for x in fs if x.is_file() and not {".cache", ".locks"} & set(x.relative_to(M).parts) and not x.name.startswith("._") and not x.name.endswith((".lock", ".metadata", ".part")) and str(x.relative_to(M)) not in EXCLUDE and not str(x.relative_to(M)).startswith(EXCLUDE_PREFIX)):
                rows.append({"path": str(f.relative_to(M)), "bytes": f.stat().st_size, "sha256": sha(f), "set": d})
        out = {"schema": "dabai-models/1", "sets": {d: {"licence": l, "used_by": u, "optional": d in OPTIONAL} for d, (l, u) in SET.items()},
               "total_gb": round(sum(r["bytes"] for r in rows) / 1e9, 2), "files": rows}
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1)); print(f"{len(rows)} files, {out['total_gb']} GB -> {a.out}"); return 0
    lock = json.loads(LOCK.read_text()); bad = []; have = 0
    for r in lock["files"]:
        f = M / r["path"]; opt = lock["sets"].get(r["set"], {}).get("optional")
        if not f.is_file(): (print("optional missing:", r["path"]) if opt else bad.append(f"missing {r['path']}")); continue
        if f.stat().st_size != r["bytes"]: bad.append(f"size {r['path']}: {f.stat().st_size} != {r['bytes']}"); continue
        if a.deep and sha(f) != r["sha256"]: bad.append(f"sha256 {r['path']}"); continue
        have += 1
    print(f"weights: {have} / {len(lock['files'])} files ok ({lock['total_gb']} GB listed){' (sha256 checked)' if a.deep else ''}")
    for b in bad[:30]: print("  BAD", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
