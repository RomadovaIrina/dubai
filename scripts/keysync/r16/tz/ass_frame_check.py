#!/usr/bin/env python3
"""r16 TZ 9.1-9.3 on KeySync: do the BURNED subtitles appear / disappear on the frames the .ass says, on the 25 fps output grid?
Per frame: |burned - plain| (the same video without subtitles), minus each pixel's median difference over the whole video (a static
watermark / the re-encode bias cancel out); a frame shows text when > MIN_PX pixels differ by > THR. Every on / off edge of that mask is
compared with the .ass event edges (libass shows an event on frame n when start <= n / fps < end: first frame ceil(start*fps), last
ceil(end*fps) - 1). Back-to-back events (no gap) have no on / off edge between them, so only the edges next to a gap are scored.
    python ass_frame_check.py --ass X.ass --plain plain.mp4 --burned burned.mp4 --out check.json [--fps 25]"""
import argparse, json, math, re, subprocess, sys
import numpy as np


def frames(path, w=320):
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path],
                        capture_output=True, text=True, check=True)
    s = json.loads(pr.stdout)["streams"][0]; h = int(round(s["height"] * w / s["width"] / 2) * 2); fr = s["r_frame_rate"].split("/")
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w), float(fr[0]) / float(fr[1])


def ass_events(path):
    ev = []
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"Dialogue:\s*\d+,(\d+):(\d+):(\d+)\.(\d+),(\d+):(\d+):(\d+)\.(\d+),", ln)
        if m:
            g = [int(x) for x in m.groups()]; ev.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 100, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 100))
    return sorted(ev)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ass", required=True); ap.add_argument("--plain", required=True); ap.add_argument("--burned", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fps", type=float, default=0); ap.add_argument("--thr", type=int, default=40); ap.add_argument("--min-px", type=int, default=30)
    a = ap.parse_args()
    P, fp = frames(a.plain); B, fb = frames(a.burned); n = min(len(P), len(B)); fps = a.fps or fb
    d = np.abs(B[:n].astype(np.int16) - P[:n].astype(np.int16)); med = np.median(d[:: max(1, n // 400)], axis=0)
    on = ((d - med) > a.thr).reshape(n, -1).sum(1) > a.min_px
    edges = [(i, "on" if on[i] else "off") for i in range(1, n) if on[i] != on[i - 1]] + ([(0, "on")] if on[0] else [])
    ev = ass_events(a.ass); exp = []
    for k, (s, e) in enumerate(ev):
        gap_before = k == 0 or ev[k - 1][1] < s - 1e-3; gap_after = k == len(ev) - 1 or ev[k + 1][0] > e + 1e-3
        if gap_before: exp.append((math.ceil(s * fps - 1e-6), "on", k))
        if gap_after: exp.append((math.ceil(e * fps - 1e-6), "off", k))
    deltas = []
    for f, kind, k in exp:
        if f >= n: continue
        cand = [i for i, t in edges if t == kind and abs(i - f) <= 6]
        deltas.append({"event": k, "edge": kind, "expected_frame": f, "observed_frame": min(cand, key=lambda i: abs(i - f)) if cand else None})
    got = [x["observed_frame"] - x["expected_frame"] for x in deltas if x["observed_frame"] is not None]
    res = {"fps_plain": fp, "fps_burned": fb, "fps_used": fps, "frames": n, "events": len(ev), "scored_edges": len(deltas), "matched_edges": len(got),
           "exact_share": round(sum(g == 0 for g in got) / max(len(got), 1), 3), "within_1_frame_share": round(sum(abs(g) <= 1 for g in got) / max(len(got), 1), 3),
           "max_abs_frames": max((abs(g) for g in got), default=None), "mean_frames": round(float(np.mean(got)), 3) if got else None,
           "unmatched": [x for x in deltas if x["observed_frame"] is None], "text_frames": int(on.sum()), "edges_observed": len(edges), "deltas": deltas}
    json.dump(res, open(a.out, "w"), indent=1)
    print("ASS_FRAME_CHECK", json.dumps({k: v for k, v in res.items() if k not in ("deltas", "unmatched")}), "unmatched", len(res["unmatched"]))


if __name__ == "__main__":
    main()
