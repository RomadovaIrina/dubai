#!/usr/bin/env python3
"""Round 3 (track A) table over r3_eval.py results.  r3_table.py BASE... [--md OUT.md] [--worst K]
Columns: SyncNet dub-only / mix (AV), per-second conf_at0 mean / p10 (seconds with a lip-synced face), lip-gap vs dub-energy r (lag 0 / best lag),
mouth vel / jerk / resid / sharp vs the master (v4), lower-face shear, frozen-mouth seconds, nose slide, LatentSync s per LS frame.
--worst K: the K worst seconds by conf_at0 per base (with dub / source activity), for the per-second worst-case list."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def f(x, n=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def load(base):
    b = pathlib.Path(base); e = json.loads(b.with_suffix(".r3eval.json").read_text())
    pf = b.with_suffix(".syncpf.json"); ps = json.loads(pf.read_text())["per_second"] if pf.exists() else []
    v4 = b.with_suffix(".v4.json"); v4d = json.loads(v4.read_text()) if v4.exists() else {}
    return b.name, e, ps, v4d


def frozen_seconds(v4d):
    ps = v4d.get("per_second") or {}
    rows = ps if isinstance(ps, list) else ps.get("rows", [])
    fr = [r.get("frozen_mouth") for r in rows if isinstance(r, dict) and r.get("frozen_mouth") is not None]
    return (int(np.sum(np.array(fr) > 1.5)), len(fr)) if fr else (None, None)


def silent_frozen(base, e, v4d):
    """mean v4 frozen_mouth over seconds that are >= 50 % dub-silent-while-source-talks (r3_lipsync.silent_mask on the run's manifest)."""
    import sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import r3_lipsync as r3
    man = json.loads(pathlib.Path(base).with_suffix(".manifest.json").read_text()); f0 = e.get("f0") or 0; n = e.get("frames") or 0
    if "units" not in man:   # window manifests carry no units: the baseline manifest of the same video (same audio)
        vid = pathlib.Path(man["input"]).stem; man = json.loads(pathlib.Path(f"/workspace/dub/canonical/artifacts/baseline/{vid}/final/{vid}_final.manifest.json").read_text())
    sm = r3.silent_mask(man, f0 + n)[f0:f0 + n]; rows = v4d.get("per_second") or []
    fz = [r["frozen_mouth"] for r in rows if r.get("frozen_mouth") is not None and sm[int(r["t"]) * 25:int(r["t"]) * 25 + 25].mean() >= 0.5]
    return (float(np.mean(fz)) if fz else None), len(fz)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("bases", nargs="+"); ap.add_argument("--md"); ap.add_argument("--worst", type=int, default=0)
    a = ap.parse_args()
    L = ["| run | SyncNet dub / mix (AV) | per-s conf@0 mean / p10 | lip-dub r lag0 / best (lag) | mouth vel / jerk / resid / sharp | lf shear | frozen s (>1.5) | silent-s frozen (n) | joint jump med / >1.5 (src) | nose slide | LS s/frame |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    W = []
    for b in a.bases:
        try: name, e, ps, v4d = load(b)
        except FileNotFoundError: continue
        sy = e.get("sync", {}); d, m = sy.get("dub_only", {}), sy.get("mix", {})
        c0 = np.array([r["conf_at0"] for r in ps if "conf_at0" in r and r.get("n", 0) >= 20], float)
        pf = e.get("syncpf", {}); lags = pf.get("lip_env_r_by_lag") or {}
        best = max(lags.items(), key=lambda kv: kv[1] if kv[1] is not None else -9) if lags else (None, None)
        v4 = e.get("v4", {}); fz = frozen_seconds(v4d)
        sf = silent_frozen(b, e, v4d)
        bj = (v4d.get("latentsync_window_boundaries") or {}).get("boundary_d1_jump") or {}; bv, br = bj.get("video", {}), bj.get("reference", {})
        jj = f"{f(bv.get('median'))} / {f(bv.get('share_gt_1_5'))} ({f(br.get('median'))} / {f(br.get('share_gt_1_5'))}, n {bv.get('boundaries')})" if bv else "–"
        sl = (e.get("slide") or {}).get("regions", {}).get("nosebase", {}).get("slide_px", {}).get("mean")
        L.append(f"| {name} | {f(d.get('conf'))} / {f(m.get('conf'))} ({d.get('av_offset')}) | {f(float(c0.mean()) if len(c0) else None)} / {f(float(np.percentile(c0, 10)) if len(c0) else None)} | "
                 f"{f(pf.get('lip_env_r'))} / {f(best[1])} ({best[0]}) | {f(v4.get('mouth_vel'))} / {f(v4.get('mouth_jerk'))} / {f(v4.get('mouth_resid'))} / {f(v4.get('mouth_sharp'))} | "
                 f"{f(v4.get('lf_shear'))} | {fz[0] if fz[0] is not None else '–'}/{fz[1] if fz[1] is not None else '–'} | {f(sf[0])} ({sf[1]}) | {jj} | {f(sl, 3)} | {f((e.get('speed') or {}).get('ls_s_per_ls_frame'), 3)} |")
        if a.worst:
            t0 = (e.get("f0") or 0) / 25
            rows = sorted([r for r in ps if "conf_at0" in r and r.get("n", 0) >= 20], key=lambda r: r["conf_at0"])[:a.worst]
            W.append(f"* {name}: " + ", ".join(f"{t0 + r['t']:.0f}s conf {r['conf_at0']:.1f} (best {r['conf_best']:.1f}@{r['offset_best']}, dub {r.get('dub_act')}, src {r.get('src_act')}, lip r {r.get('lip_r0')})" for r in rows))
    txt = "\n".join(L + ([""] + W if W else [])); print(txt)
    if a.md: pathlib.Path(a.md).write_text(txt + "\n")


if __name__ == "__main__":
    main()
