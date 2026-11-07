#!/usr/bin/env python3
"""r13 0.5: the e2e AUDIO stage (+ sep) per sub-step, GPU-busy and wall s per video-minute, from one e2e run made with R9_TIMELINE.
Samples = the NVML sampler of the run (work/nvml.jsonl, 100 ms, device SM util); every sample goes to the INNERMOST open foreground timeline
interval (scripts/r9/gpu_busy.py rule): pipe:<pipeline stage> / qwen / t3 / s3gen / qa / chain:* ...; bg:* (MediaPipe track, htdemucs stems
and remix of the sep overlap) run alongside and are reported apart (their samples are shared with the foreground). Window = the e2e audio +
sep stages (work/stages.tsv).
    python scripts/r13/prof_audio.py --runs DIR1 DIR2 ... [--json out.json]     (DIR = an e2e out dir)"""
import argparse, collections, json, pathlib


def one(d: pathlib.Path):
    W = d / "work"; rows = [l.rstrip("\n").split("\t") for l in open(W / "stages.tsv") if "\t" in l]
    st = {n: float(t) for n, t in rows}; order = [n for n, _ in rows]
    a0 = st["audio"]; a1 = st[order[order.index("sep") + 1]] if "sep" in st else st[order[order.index("audio") + 1]]
    dur = json.loads((W / "precheck.json").read_text())["probe"]["duration_s"]; vmin = dur / 60
    S = [json.loads(l) for l in open(W / "nvml.jsonl") if l.strip()]
    S = [(s["t"], float(s.get("util", 0))) for s in S if "t" in s]
    T = [json.loads(l) for l in open(d / "tl.jsonl") if l.strip()]
    fg = [t for t in T if not t[0].startswith("bg:")]; bg = [t for t in T if t[0].startswith("bg:")]
    wall, busy = collections.Counter(), collections.Counter(); bgw, bgb = collections.Counter(), collections.Counter(); tot = 0.0
    for k, (ts, u) in enumerate(S):
        if not (a0 <= ts <= a1): continue
        dt = min(0.2, (S[k + 1][0] - ts) if k + 1 < len(S) else 0.1)
        op = [t for t in fg if t[1] <= ts < t[2]]
        cat = max(op, key=lambda t: t[1])[0] if op else "other (start / loads outside a stage / bash / CPU chain)"
        wall[cat] += dt; busy[cat] += u / 100 * dt; tot += u / 100 * dt
        for b in bg:
            if b[1] <= ts < b[2]: bgw[b[0]] += dt; bgb[b[0]] += u / 100 * dt
    return {"id": d.name, "vmin": round(vmin, 4), "audio_sep_wall_s": round(a1 - a0, 1), "gpu_busy_s": round(tot, 1),
            "fg": {c: {"wall": round(wall[c], 1), "busy": round(busy[c], 1)} for c in wall}, "bg": {c: {"wall": round(bgw[c], 1), "busy_shared": round(bgb[c], 1)} for c in bgw}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--json"); a = ap.parse_args()
    R = [one(pathlib.Path(d)) for d in a.runs]; V = sum(r["vmin"] for r in R)
    cats = sorted({c for r in R for c in r["fg"]}, key=lambda c: -sum(r["fg"].get(c, {}).get("busy", 0) for r in R))
    print(f"videos {[r['id'] for r in R]}  {V:.3f} video-min;  audio+sep wall {sum(r['audio_sep_wall_s'] for r in R) / V:.1f} s/vmin, GPU-busy {sum(r['gpu_busy_s'] for r in R) / V:.1f} s/vmin")
    print("| sub-step | GPU-busy s/vmin | wall s/vmin | " + " | ".join(f"{r['id']} busy/wall s" for r in R) + " |")
    print("|---|---:|---:|" + "---|" * len(R))
    for c in cats:
        b = sum(r["fg"].get(c, {}).get("busy", 0) for r in R); w = sum(r["fg"].get(c, {}).get("wall", 0) for r in R)
        print(f"| {c} | {b / V:.1f} | {w / V:.1f} | " + " | ".join(f"{r['fg'].get(c, {}).get('busy', 0)}/{r['fg'].get(c, {}).get('wall', 0)}" for r in R) + " |")
    for c in sorted({c for r in R for c in r["bg"]}):
        print(f"| (alongside) {c} | {sum(r['bg'].get(c, {}).get('busy_shared', 0) for r in R) / V:.1f} shared | {sum(r['bg'].get(c, {}).get('wall', 0) for r in R) / V:.1f} | " + " | ".join(f"{r['bg'].get(c, {}).get('busy_shared', 0)}/{r['bg'].get(c, {}).get('wall', 0)}" for r in R) + " |")
    if a.json: pathlib.Path(a.json).write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
