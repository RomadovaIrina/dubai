#!/usr/bin/env python3
"""Track B comparison table: baseline (cand5) vs whole-video candidates, from the files written by run_candidates_full.sh /
candidate_eval.py and the Phase 0 baseline reports.
    python scripts/quality/track_b_table.py --ids 01 03 04 05 --tags C2T_nocf C1C2T_gfpgan [--out table.md]"""
from __future__ import annotations
import argparse, json, pathlib

BASE = pathlib.Path("/workspace/dub/artifacts/baseline"); FULL = pathlib.Path("/workspace/dub/artifacts/restoration-temporal/full/final")


def jl(p):
    p = pathlib.Path(p); return json.loads(p.read_text()) if p.exists() else None


def mj(d):
    if not d: return {}
    c = d.get("classes", {}).get("ls_core", {}); w = d.get("latentsync_window_boundaries", {}); sw = d.get("classes", {}).get("switch", {})
    g = lambda blk, k, reg="mouth": (blk.get(k) or {}).get(reg, {}) and blk[k][reg]["ratio"]
    return {"resid": g(c, "resid"), "sharp": g(c, "sharp"), "d1": g(c, "d1"), "vel": g(c, "vel"), "lower_sharp": g(c, "sharp", "lowerface"),
            "bnd": (w.get("d1", {}).get("video") or {}).get("phase0_over_inner"), "bnd15": (w.get("boundary_d1_jump", {}).get("video") or {}).get("share_gt_1_5"),
            "sw_resid": g(sw, "resid"), "sw_jerk": g(sw, "jerk"), "upper_resid": g(c, "resid", "upper")}


def ft(d):
    if not d: return {}
    s = d["summary"]; r = lambda k, reg="ls_core": (s[k].get(reg) or {}).get("ratio")
    return {"ft_mouth_flicker": r("mouth_pixel_flicker"), "ft_lip_jitter": r("mouth_landmark_jitter"), "ft_upper_flicker": r("upper_pixel_flicker"), "ft_open": r("mouth_open_jitter")}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args(); rows = []
    for i in a.ids:
        rep = (jl(BASE / i / "report_baseline_cand5.json") or [{}])[0]
        b = {"id": i, "variant": "cand5 (baseline)", "license": "NO (CodeFormer)", "sync_mix": (rep.get("syncnet") or {}).get("confidence") if isinstance(rep.get("syncnet"), dict) else None,
             "sync_dub": (rep.get("syncnet_dub_only") or {}).get("confidence"), "av": (rep.get("syncnet") or {}).get("av_offset") if isinstance(rep.get("syncnet"), dict) else None,
             "audio_identical": "ref", **mj(jl(FULL / f"{i}_cand5.mouth_jitter_hr.json")), **ft(jl(BASE / i / "face_temporal" / f"{i}_cand5.face_temporal.json"))}
        rows.append(b)
        for t in a.tags:
            e = jl(FULL / f"{i}_{t}.eval.json") or {}
            rows.append({"id": i, "variant": t, "license": "YES" if t == "C2T_nocf" else "YES* (GFPGAN, FFHQ data)", "sync_mix": (e.get("syncnet_mix") or {}).get("confidence"),
                         "sync_dub": (e.get("syncnet_dub_only") or {}).get("confidence"), "av": (e.get("syncnet_mix") or {}).get("av_offset"),
                         "audio_identical": (e.get("audio_md5") or {}).get("identical"), "frames_equal": (e.get("frames") or {}).get("equal"),
                         **mj(jl(FULL / f"{i}_{t}.mouth_jitter_hr.json")), **ft(jl(FULL / "face_temporal" / f"{i}_{t}.face_temporal.json"))})
    cols = [("video", "id"), ("variant", "variant"), ("license", "license"), ("mouth texture resid", "resid"), ("mouth sharp", "sharp"), ("lower-face sharp", "lower_sharp"),
            ("LS-window jump d1", "bnd"), ("jumps > 1.5x", "bnd15"), ("switch resid / jerk", None), ("mouth motion vs head", "vel"), ("upper resid", "upper_resid"),
            ("SyncNet mix / dub-only / AV", None), ("old face_temporal mouth flicker / lip jitter", None), ("audio bit-identical", "audio_identical")]
    def f(x): return "-" if x is None or x == {} else (f"{x:.2f}" if isinstance(x, float) else str(x))
    L = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        v = []
        for c, k in cols:
            if c.startswith("switch"): v.append(f"{f(r.get('sw_resid'))} / {f(r.get('sw_jerk'))}")
            elif c.startswith("SyncNet"): v.append(f"{f(r.get('sync_mix'))} / {f(r.get('sync_dub'))} / {f(r.get('av'))}")
            elif c.startswith("old"): v.append(f"{f(r.get('ft_mouth_flicker'))} / {f(r.get('ft_lip_jitter'))}")
            else: v.append(f(r.get(k)))
        L.append("| " + " | ".join(v) + " |")
    t = "\n".join(L) + "\n"; print(t)
    if a.out: pathlib.Path(a.out).write_text(t); pathlib.Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
