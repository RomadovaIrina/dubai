#!/usr/bin/env python3
"""Round 3 D: spec compliance status per reference (cand5 / clean0 / r2m / r2m-FIX30 ...), one row per spec item, PASS / FAIL / NOT MEASURED + numbers.
Sources: r3_scorecard.py score JSONs (+ their mouth_jitter_hr JSON), the run manifests, the MediaPipe npz (head pose), C's audio_stages detector
JSON of the audio actually muxed, and for cand5 (no mp4 left: video NOT MEASURED) the canonical baseline reports.
    python scripts/quality/r3_spec_status.py --score-dir S --label clean0 --label r2m25 --label r2mfix30 --out spec_status.md [--json x.json]"""
from __future__ import annotations
import argparse, json, pathlib, re, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
CANON = pathlib.Path("/workspace/dub/canonical/artifacts/baseline"); IDS = ["01", "02", "03", "04", "05"]


def cand5_report(id_):
    p = CANON / id_ / "report_baseline_cand5.md"
    if not p.exists(): return {}
    r = {}
    for line in p.read_text().splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", line)
        if m: r[m.group(1)] = m.group(2)
    return r


def audio_events(id_, man=None):
    """C's audio_stages detector: clicks in the muxed audio + hot unit edges (joints) + bed steps. cand5 audio = canonical JSON."""
    p = CANON / id_ / "audio_stages" / f"{id_}_cand5.audio_stages.json"
    dd = man and (man.get("audio_dir") or (man.get("candidate") or {}).get("dubbed_dir"))
    if dd and "/canonical/" not in dd and "/artifacts/baseline/" not in dd:   # another audio than cand5's: its own detector JSON only
        q = sorted(pathlib.Path(dd).glob("*audio_stages*.json"))
        p = q[0] if q else None
    if p is None or not p.exists(): return None
    d = json.loads(p.read_text()); c = d.get("counts", {})
    return {"clicks_mp4": d.get("clicks", {}).get("mp4"), "clicks_dub_voice": d.get("clicks", {}).get("dub_voice"), "hot_edges": c.get("hot_edge", 0),
            "bed_steps": c.get("bed_step", 0), "bed_voice": c.get("bed_voice", 0), "source": str(p)}


def pose(man):
    import clean_face_stack as cfs
    fs = man.get("face_stack") or {}; npz = fs.get("npz")
    if not npz or not pathlib.Path(npz).exists(): return None
    z = np.load(npz)
    if "matrix" not in z.files: return None
    P = cfs.pose_from_matrix(z["matrix"]); ok = np.isfinite(P[:, 0])
    if not ok.any(): return {"face_frames": 0}
    y, p = np.abs(P[ok, 0]), np.abs(P[ok, 1])
    pc = (fs.get("pose") or {}).get("precheck") or {}
    return {"face_frames": int(ok.sum()), "yaw_abs_p50_p99_max": [round(float(np.percentile(y, q)), 1) for q in (50, 99)] + [round(float(y.max()), 1)],
            "pitch_abs_p50_p99_max": [round(float(np.percentile(p, q)), 1) for q in (50, 99)] + [round(float(p.max()), 1)],
            "frac_yaw_gt30": round(float((y > 30).mean()), 4), "frac_pitch_gt30": round(float((p > 30).mean()), 4),
            "frac_any_gt30": round(float(((y > 30) | (p > 30)).mean()), 4), "precheck_fired": bool(pc.get("voice_only")), "precheck_enabled": pc.get("enabled")}


def collect(sd: pathlib.Path, label: str, id_: str):
    sp = sd / f"{id_}_{label}.score.json"
    if not sp.exists(): return None
    r = json.loads(sp.read_text()); man = json.loads(pathlib.Path(r["manifest"]).read_text())
    mj = sd / f"{id_}_{label}.mjhr.json"; sw = None
    if mj.exists():
        d = json.loads(mj.read_text()); c = (d.get("classes") or {}).get("switch") or {}
        d1 = (c.get("d1") or {}).get("mouth"); acc = (c.get("acc") or {}).get("mouth")
        sw = {"switch_frames": len(d.get("switch_frames") or []), "d1_ratio": d1 and d1.get("ratio"), "d1_p95_ratio": d1 and d1.get("p95_ratio"), "acc_ratio": acc and acc.get("ratio")}
    fs = man.get("face_stack") or {}; fix = man.get("fix30") or {}
    nframes = r["output"]["frames"]; held = fs.get("held_landmark_frames") or 0; fb = fix.get("fallback_frames") or 0
    return {"score": r, "switch": sw, "audio": audio_events(id_, man), "pose": pose(man),
            "robust": {"held_landmark_frames": held, "fix30_fallback_frames": fb, "replaced": held + fb, "frames": nframes,
                       "pct": round(100 * (held + fb) / max(nframes, 1), 3), "face_stack_failed": fs.get("face_stack_failed"), "failure": fs.get("failure")},
            "faces": {"tracks": (fs.get("faces") or {}).get("tracks"), "two_face_share": (fs.get("two_face_probe") or {}).get("share"),
                      "probed": (fs.get("two_face_probe") or {}).get("probed_frames_with_face")}}


def f2(x, nd=2):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--score-dir", required=True); ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--json", default=None); ap.add_argument("--syncnet-min", type=float, default=None)
    a = ap.parse_args(); sd = pathlib.Path(a.score_dir)
    data = {lab: {i: collect(sd, lab, i) for i in IDS} for lab in a.label}
    c5 = {i: cand5_report(i) for i in IDS}
    L = ["# Spec status (round 3, Agent D)", "",
         "Columns: one per reference run. Cell = verdict + numbers per video (01 / 02 / 03 / 04 / 05). cand5: the canonical baseline reports "
         "(its mp4s are gone and cannot be re-rendered without the non-commercial models), so its video items are NOT MEASURED unless the report has them.", ""]
    hdr = "| spec | item | cand5 | " + " | ".join(a.label) + " |"; L += [hdr, "|---|---|" + "---|" * (len(a.label) + 1)]

    def row(spec, item, c5cell, fn):
        cells = []
        for lab in a.label:
            vals = {i: data[lab][i] for i in IDS if data[lab][i] is not None}
            if not vals: cells.append("NOT MEASURED"); continue
            try: cells.append(fn(vals))
            except Exception as e: cells.append(f"NOT MEASURED ({type(e).__name__})")
        L.append(f"| {spec} | {item} | {c5cell} | " + " | ".join(cells) + " |")

    def per(vals, g): return " / ".join(f"{i}: {g(v)}" for i, v in vals.items())

    # 9.4 / 5.6
    def r94(v):
        ok = all(x["score"]["spec_9_4"]["pass"] for x in v.values())
        return ("PASS " if ok else "**FAIL** ") + per(v, lambda x: f"{x['score']['output']['fps']:g}fps {x['score']['output']['frames']}/{x['score']['source']['frames']}f "
                                                     f"{x['score']['output']['w']}x{x['score']['output']['h']} Δ{x['score']['spec_9_4']['duration_delta_s']:+.2f}s {x['score']['output']['codec']}")
    row("9.4 / 5.6", "fps, resolution, frames, duration, H.264 vs source", "**FAIL** (25 fps CFR master muxed; coordinator gap table 17.10)", r94)

    def r85w(v):
        sh = [x["score"]["face_v4"]["window_boundaries"]["video"]["share_gt_1_5"] for x in v.values() if (x["score"].get("face_v4") or {}).get("window_boundaries")]
        ref = [x["score"]["face_v4"]["window_boundaries"]["reference"]["share_gt_1_5"] for x in v.values() if (x["score"].get("face_v4") or {}).get("window_boundaries")]
        verdict = "**FAIL**" if sh and max(s - r for s, r in zip(sh, ref)) > 0.05 else "PASS"
        return f"{verdict} " + per(v, lambda x: (lambda w: f"{w['video']['boundaries']} starts, >1.5x {w['video']['share_gt_1_5']:.2f} (src {w['reference']['share_gt_1_5']:.2f}), >2x {w['video']['share_gt_2']:.2f}" if w else "no LS")(
            (x["score"].get("face_v4") or {}).get("window_boundaries")))
    row("8.5", "video jumps at LatentSync window starts (mouth d1 step vs inside the window; source = same frames)", "x1.38 step, 41 % > 1.5x (03 only, CLEAN0.md)", r85w)

    def r85s(v):
        rs = [x["switch"]["d1_ratio"] for x in v.values() if x.get("switch") and x["switch"]["d1_ratio"] is not None]
        return ("**FAIL** " if rs and max(rs) > 1.5 else "PASS " if rs else "") + per(v, lambda x: f"{x['switch']['switch_frames']} fr, d1 {f2(x['switch']['d1_ratio'])} p95 {f2(x['switch']['d1_p95_ratio'])}" if x.get("switch") and x["switch"]["d1_ratio"] is not None else "no switches")
    row("8.5", "LS ↔ original switches: switch-zone frames, mouth d1 / p95 vs source (1 = like the source)", "NOT MEASURED", r85s)

    def r85a(v):
        a_ = {i: x["audio"] for i, x in v.items() if x.get("audio")}
        if not a_: return "NOT MEASURED"
        bad = sum((y["clicks_mp4"] or 0) for y in a_.values())
        return ("PASS " if bad == 0 else "**FAIL** ") + " / ".join(f"{i}: clicks {y['clicks_mp4']}, hot edges {y['hot_edges']}, bed steps {y['bed_steps']}" for i, y in a_.items())
    c5a = " / ".join(f"{i}: clicks {audio_events(i)['clicks_mp4']}" for i in IDS if audio_events(i))
    row("8.5", "audio clicks at joints (C's detector on the muxed audio), hot unit edges, bed steps", c5a, r85a)

    def r84(v):
        px = {i: x["score"]["feather"].get("edge_10_90_px") for i, x in v.items()}
        okv = [p for p in px.values() if p is not None]
        verdict = "PASS" if okv and all(10 <= p <= 15 for p in okv) else "**FAIL**"
        return f"{verdict} " + " / ".join(f"{i}: {f2(p, 1)} px" for i, p in px.items())
    row("8.4", "mask feather 10-90 % edge on the output (px)", "NOT MEASURED (same Gaussian rule as clean0)", r84)

    def r121(v):
        def s(x):
            y = x["score"].get("syncnet")
            return "no LS" if not y else (f"mix {y['mix'][0]:.2f}/{y['mix'][1]:+d}, dub {y['dub_only'][0]:.2f}/{y['dub_only'][1]:+d}" if "mix" in y else "error")
        return "REPORT (threshold UNDECIDED) " + per(v, s)
    c5s = " / ".join(f"{i}: mix {c5[i].get('SyncNet','–')}, dub {c5[i].get('SyncNet dub-only','–')}, AV {c5[i].get('AV off','–')}" for i in IDS if c5[i])
    row("12.1", "SyncNet confidence / AV offset, final mix and dub-only", c5s, r121)

    def r122(v):
        return per(v, lambda x: f"WER {f2((x['score'].get('asr') or {}).get('wer_final_mix'), 3)}")
    c5w = " / ".join(f"{i}: WER {c5[i].get('WER','–')} CER(ctc) {c5[i].get('CTC CER','–')}" for i in IDS if c5[i])
    row("12.2", "Whisper large-v3 (GPU) back-transcription of the FINAL MIX vs the synthesised English text", c5w + " (per-unit dub voice)", r122)

    def r11(v):
        worst = max(x["robust"]["pct"] for x in v.values())
        return ("PASS " if worst <= 2 else "**FAIL** ") + per(v, lambda x: f"{x['robust']['replaced']} fr = {x['robust']['pct']:.2f} %" + (" FACE STACK FAILED" if x["robust"]["face_stack_failed"] else ""))
    row("11.1-11.4", "replaced / invalid frames (held landmarks + composite fallbacks) vs the 2 % rule", "NOT MEASURED", r11)

    def r55(v):
        fired = [i for i, x in v.items() if x["pose"] and x["pose"].get("precheck_fired")]
        return (f"pre-check fires on {fired} " if fired else "pre-check fires on none ") + per(v, lambda x: (
            f"yaw p99 {x['pose']['yaw_abs_p50_p99_max'][1]}° max {x['pose']['yaw_abs_p50_p99_max'][2]}°, pitch p99 {x['pose']['pitch_abs_p50_p99_max'][1]}°, >30° {100 * x['pose']['frac_any_gt30']:.1f} %"
            if x["pose"] and x["pose"].get("face_frames") else "no face"))
    row("5.4 / 5.5", "head yaw / pitch per frame (MediaPipe matrix), share > 30°, would the 30 % pre-check fire", "NOT MEASURED (no pose in cand5)", r55)

    def r82(v):
        two = {i: x["faces"] for i, x in v.items()}
        any2 = [i for i, f in two.items() if (f["two_face_share"] or 0) >= 0.10 or (f["tracks"] or 0) > 1]
        return ("two faces in " + str(any2) if any2 else "single face everywhere (8.2 path not exercised) ") + " " + per(v, lambda x: f"tracks {x['faces']['tracks']}, 2-face share {f2(x['faces']['two_face_share'])}")
    row("8.2", "second face present? only the active speaker touched?", "NOT MEASURED", r82)

    def rnorm(v):
        tot = {i: x["score"]["speed"]["s_per_video_min"].get("total") for i, x in v.items()}
        okv = [t for t in tot.values() if t]
        return ("PASS " if okv and max(okv) <= 300 else "**FAIL** ") + " / ".join(f"{i}: {f2(t, 0)}" for i, t in tot.items()) + " s/video-min (video stage only; audio + FIX30 see gpu_audit.md)"
    DUR = {"01": 101.667, "02": 110.867, "03": 90.333, "04": 53.9, "05": 52.367}
    c5n = "**FAIL** " + " / ".join(f"{i}: {float(c5[i]['wall s']) / DUR[i] * 60:.0f}" for i in IDS if c5[i].get("wall s")) + " s/video-min (full E2E incl. CodeFormer, Qwen CPU)"
    row("norm", "s / video-min (stage totals of the run)", c5n, rnorm)

    L += ["", "Per-video detail: `/workspace/dub/artifacts/r3d/score/<id>_<label>.score.json` (+ `.mjhr.json`); table of all metrics: `scorecard.md`."]
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))
    if a.json: pathlib.Path(a.json).write_text(json.dumps({"labels": data, "cand5": c5}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
