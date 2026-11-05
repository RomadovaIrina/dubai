#!/usr/bin/env python3
"""r15 lever 1 (r5): markdown tables of a run, val per checkpoint (ckpt_val.json) + E2E per video and arm (eval_r14.sh outputs).
    python scripts/r15/r5_table.py --root /workspace/dub/r15x --run r5_s4 --base q2u --arms s4p1500c8u r5c8u r5fc8u [--ids old03 m03 m18]"""
import argparse, json, pathlib, re


def jl(p):
    try: return json.load(open(p))
    except Exception: return None


def f(v, n=3): return "–" if v is None else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--run", default="r5_s4"); ap.add_argument("--base", default="q2u")
    ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--ids", nargs="+", default=["old03", "m03", "m18"]); a = ap.parse_args()
    R = pathlib.Path(a.root); E = R / "eval"; cv = jl(R / "ckpt" / a.run / "ckpt_val.json") or {}
    print(f"### {a.run}: val per checkpoint (EMA; ckpt_val.py, {cv.get('val_windows', '?')} held-out windows)\n")
    print("| ckpt | latent MSE | mouth pix L1 | jitter | stable-SyncNet sim | opening / teacher |\n|---|---|---|---|---|---|")
    for r in cv.get("by_ckpt", []):
        print(f"| {r['ckpt']}{' **pick**' if r['ckpt'] == cv.get('pick') else ''} | {f(r.get('mse'), 4)} | {f(r.get('pix'), 4)} | {f(r.get('jitter'), 4)} | {f(r.get('sync'))} | {f(r.get('open_ratio'))} |")
    if cv: print(f"\npick: {cv.get('pick')} ({cv.get('pick_reason')})")
    print(f"\n### End-to-end (dabai_job, q2 + F3 composite, LS_FACE_UNITS=1; base = {a.base})\n")
    print(f"| video | arm | sync_speech | flicker mean | flicker pops | opening vs {a.base} (mean / speed) | near-closed voiced | GPU s / video-min |\n|---|---|---|---|---|---|---|---|")
    for id_ in a.ids:
        for t in [a.base] + a.arms:
            n = f"{t}_{id_}"; st = E / f"{n}.sync.txt"; sy = re.search(r": ([-0-9.]+) over", st.read_text()) if st.exists() else None
            lg = E / f"{n}.log"; gs = None
            if lg.exists():
                m = re.search(r'"video_minutes": ([0-9.]+).*?"gpu_busy_s": ([0-9.]+)', lg.read_text(), re.S)
                if m: gs = float(m.group(2)) / float(m.group(1))
            if not sy and gs is None: continue
            src = t if t != a.base else next((x for x in a.arms if (E / f"{x}_{id_}.flicker.json").exists()), t)   # the base's numbers live in each arm's json
            fv = (jl(E / f"{src}_{id_}.flicker.json") or {}).get("videos", {}).get(t, {}); mv = (jl(E / f"{src}_{id_}.mouth.json") or {}).get(t, {})
            vs = mv.get(f"vs_{a.base}", {})
            op = "1" if t == a.base else (f"{f(vs.get('mean'))} / {f(vs.get('speed'))}" if vs else "–")
            print(f"| {id_} | {t} | {sy.group(1) if sy else '–'} | {f(fv.get('mean'), 4)} | {f(fv.get('pops'), 0)} | {op} | {f(mv.get('closed_share'), 2)} | {f(gs, 0)} |")


if __name__ == "__main__":
    main()
