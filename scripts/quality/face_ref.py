#!/usr/bin/env python3
"""LatentSync-mask reference WITHOUT a video run (Agent C r2, 2026-10-17): the canonical baseline manifests (with their LatentSync segments) were
not available on dub-C2, so dub_gaps.py's "LS frames" need another fixed mask. This builds it the way the production video stage would:
master_25fps.mp4 -> MediaPipe router classes per frame (clean_face_stack, the video agents' face stack) x speech gate of a REFERENCE dub track
(source VAD u dub VAD, margin 0.24 s / merge 0.6 s = OPTIMIZED_DEFAULTS) -> VALID_FACE runs >= 25 frames = LATENT_SYNC. Writes a copy of the
reference manifest with a `lipsync` block (master_frames, segments) that dub_gaps.py --ref-manifest accepts. No LatentSync, no GPU.

    python scripts/quality/face_ref.py --manifest RUN_CAND5/final/03_final.manifest.json --out FACE/03.ref.manifest.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality"): sys.path.insert(0, str(HERE.parent / d))
FPS = 25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--min-ls-frames", type=int, default=25)
    ap.add_argument("--margin", type=float, default=0.24); ap.add_argument("--merge", type=float, default=0.6)
    a = ap.parse_args()
    import clean_face_stack as cfs, face_aware_latentsync as fa
    import run_clean_pipeline_05 as rp
    from face_aware_latentsync_accel import build_speech_gate
    man = json.loads(pathlib.Path(a.manifest).read_text()); W = pathlib.Path(man["work_dir"]); out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    master = W / "master_25fps.mp4"
    if not master.exists(): master, _ = fa.make_master(pathlib.Path(man["input"]), W)
    npz = cfs.run_mediapipe(master, W / "master_mediapipe.npz"); lm = np.load(npz)["lm"]; n = len(lm)
    router = cfs.MediaPipeRouter(50, 80); cls = [router.row(f if np.isfinite(f[0, 0]) else None)["cls"] for f in lm]
    dur = man["source"]["duration_s"]; src = [(v["start"], v["end"]) for v in man["speech_intervals"]]
    dub = [(v["start"], v["end"]) for v in rp.run_vad(pathlib.Path(man["dubbed_track"]["wav16k"]))]
    gate = build_speech_gate([src, dub], dur, a.margin, a.merge); g = np.zeros(n, bool)
    for s, e in gate: g[int(round(s * FPS)):min(n, int(round(e * FPS)))] = True
    segs = []; i = 0
    while i < n:   # runs of (face class, gated)
        j = i
        while j < n and cls[j] == cls[i] and g[j] == g[i]: j += 1
        ls = cls[i] == "VALID_FACE" and g[i] and j - i >= a.min_ls_frames
        segs.append({"start_frame": i, "end_frame": j, "classification": cls[i] if g[i] else "NO_SPEECH", "face_class": cls[i],
                     "action": "LATENT_SYNC" if ls else "PASS_THROUGH"}); i = j
    for s in segs:   # NO_SPEECH runs keep their real face class for dub_gaps' regate (it only reads non-NO_SPEECH classes)
        pass
    man2 = dict(man); man2["lipsync"] = {"master_frames": n, "segments": segs, "source": "face_ref.py (MediaPipe router x speech gate; no LatentSync run)"}
    man2["speech_gate"] = {"enabled": True, "margin_s": a.margin, "merge_gap_s": a.merge, "intervals": gate}
    man2.setdefault("config", {})["min_ls_frames"] = a.min_ls_frames
    out.write_text(json.dumps(man2, indent=1, ensure_ascii=False))
    nls = sum(s["end_frame"] - s["start_frame"] for s in segs if s["action"] == "LATENT_SYNC")
    print(f"{pathlib.Path(man['input']).stem}: {n} frames, VALID_FACE {sum(c == 'VALID_FACE' for c in cls)}, LATENT_SYNC {nls} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
