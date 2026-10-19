#!/usr/bin/env python3
"""Residual "source speaks, dub silent" gaps of one run (baseline final manifest or a recut_offline manifest), as a list for the video side
and as the timing-policy metrics (Agent C, 2026-10-16).

Speaking = Silero VAD (manifest speech_intervals for the source, Silero on the dub track, the pipeline's own call), 10 ms grid. Every run of
"source VAD on, dub VAD off" >= --min-gap-s becomes one gap with:
  t0 / t1 / seconds, master frames [f0, f1), LatentSync frames inside it (fixed video = the reference run's LS segments),
  where it sits in its source burst (head = dub starts late, tail = dub ends early, inner = between two dub parts, whole = burst undubbed),
  the cause by unit (pause_rhythm.py classes: late_start / inner_gap / short_fill / between_units / unit_missing),
  the unit's source sentence + translation, the dub parts before / after (text, atempo, end / start), the source words spoken in the gap,
  diagnostics for the timing layers: tempo needed to stretch the previous part over the gap, lead needed to start the next part earlier.
Frame metrics (25 fps master):
  ls_frames_dub_silent_while_orig_rms  = the pilot / baseline-table metric (LS frame, dub RMS < -50 dBFS, source audio16k RMS > -35 dB)
  ls_frames_dub_silent_while_orig_vad  = same LS frames, majority of the frame is source-VAD-on / dub-VAD-off
  *_regated = the same with the LS mask re-estimated from THIS dub track (speech gate = source VAD u dub VAD with margin / merge of the reference run, face classes of the
  reference run; NO_SPEECH frames inherit the neighbouring face class; runs < min-ls-frames dropped) -- an estimate, not a video run.

    python scripts/quality/dub_gaps.py --manifest RUN.manifest.json [--ref-manifest BASELINE.manifest.json] --tag x --out OUT_DIR
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE))
R = 0.01; FPS = 25; SR16 = 16000


def mask(ivs, n):
    m = np.zeros(n, bool)
    for s, e in ivs: m[max(0, int(round(s / R))):min(n, int(round(e / R)))] = True
    return m


def runs(m, step=R):
    out = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            out.append((round(i * step, 3), round(j * step, 3))); i = j
        else: i += 1
    return out


def frame_rms(y, n):
    hop = SR16 // FPS; y = np.pad(y, (0, max(0, n * hop - len(y))))[: n * hop]
    return np.sqrt(np.mean(y.reshape(n, hop) ** 2, axis=1))


def read16(p):
    import soundfile as sf
    y, sr = sf.read(str(p), dtype="float32"); assert sr == SR16, (p, sr)
    return y if y.ndim == 1 else y.mean(axis=1)


def dub_vad(wav16, min_speech_ms):
    """Silero on the dub track, the pipeline's model; min_speech_ms 60 (not Silero's 250 default) so a short isolated word ("with" after a
    TTS pause, 04 34.8 s) is not dropped when a policy shifts it by a few frames (P1: that alone flipped a 0.9 s gap to 1.2 s)."""
    import torch, model_cache
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    torch.set_num_threads(int(__import__("os").environ.get("DUB_THREADS", "6")))
    m = model_cache.get("silero_vad", load_silero_vad); au = read_audio(str(wav16), sampling_rate=SR16)
    return [(round(float(t["start"]), 3), round(float(t["end"]), 3)) for t in get_speech_timestamps(au, m, sampling_rate=SR16, return_seconds=True, min_speech_duration_ms=min_speech_ms)]


def ls_mask_of(segs, n):
    m = np.zeros(n, bool)
    for s in segs:
        if s["action"] == "LATENT_SYNC": m[s["start_frame"]:s["end_frame"]] = True
    return m


def regate_mask(ref, src_ivs, dub_ivs, dur, n):
    """LS mask estimate for a new dub track: speech gate (source VAD u dub VAD, reference margin / merge, as run_clean_pipeline_05) x reference
    face classes, runs >= min LS frames."""
    from face_aware_latentsync_accel import build_speech_gate
    sg = ref.get("speech_gate") or {}; margin, merge = sg.get("margin_s", 0.24), sg.get("merge_gap_s", 0.6)
    gate = build_speech_gate([src_ivs, dub_ivs], dur, margin, merge)
    g = np.zeros(n, bool)
    for s, e in gate: g[int(round(s * FPS)):min(n, int(round(e * FPS)))] = True
    segs = sorted(ref["lipsync"]["segments"], key=lambda s: s["start_frame"]); cls = [None] * n
    for s in segs:
        if s["classification"] != "NO_SPEECH":
            for i in range(s["start_frame"], min(n, s["end_frame"])): cls[i] = s["classification"]
    last = next((c for c in cls if c), "VALID_FACE")
    for i in range(n):   # NO_SPEECH: face class unknown -> previous known class (next one before the first)
        if cls[i] is None: cls[i] = last
        else: last = cls[i]
    valid = g & np.array([c == "VALID_FACE" for c in cls])
    min_ls = (ref.get("config") or {}).get("min_ls_frames", 25) or 25
    out = np.zeros(n, bool)
    for s, e in runs(valid, 1.0 / FPS):
        f0, f1 = int(round(s * FPS)), int(round(e * FPS))
        if f1 - f0 >= min_ls: out[f0:f1] = True
    return out, gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--ref-manifest", default=None, help="run whose LatentSync segments / face classes are used (default: --manifest)")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--min-gap-s", type=float, default=0.1, help="gaps shorter than this are not listed (still counted in the seconds)")
    ap.add_argument("--natural-s", type=float, default=0.35, help="gaps shorter than this are labelled natural (rule of thumb 0.3-0.4 s)")
    ap.add_argument("--dub-min-speech-ms", type=int, default=60, help="Silero min speech duration on the dub track (250 = pipeline / pause_rhythm default, used in P0)")
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); ref = json.loads(pathlib.Path(a.ref_manifest).read_text()) if a.ref_manifest else man
    import run_clean_pipeline_05 as rp
    dur = man["source"]["duration_s"]; W = pathlib.Path(man["work_dir"]); RW = pathlib.Path(ref["work_dir"])
    dub16 = pathlib.Path((man.get("dubbed_track") or {}).get("wav16k") or (W / "dubbed_16k.wav"))
    if not dub16.exists(): dub16 = W / "dubbed_16k.wav"
    orig = [(v["start"], v["end"]) for v in man["speech_intervals"]]; dubv = dub_vad(dub16, a.dub_min_speech_ms)
    n = int(dur / R) + 1; mo, md = mask(orig, n), mask(dubv, n); gapm = mo & ~md
    nf = ref["lipsync"]["master_frames"]; ls_fixed = ls_mask_of(ref["lipsync"]["segments"], nf); ls_regated, gate = regate_mask(ref, orig, dubv, dur, nf)
    # per-frame: majority of the 40 ms frame is gap (VAD definition), RMS definition (pilot metric)
    k = int(round(1 / FPS / R)); gm = np.pad(gapm, (0, max(0, nf * k - n)))[: nf * k].reshape(nf, k).mean(axis=1) >= 0.5
    rd = frame_rms(read16(dub16), nf); ro = frame_rms(read16(RW / "audio16k.wav"), nf)
    rms_sil = (rd < 10 ** (-50 / 20)) & (ro > 10 ** (-35 / 20))
    units = man["units"]; parts = []
    for u in units:
        for kk, pl in enumerate(u["alignment"]["placements"]):
            if pl.get("placed"):
                parts.append({"unit": u["id"], "k": kk, "t0": pl["placed"][0], "t1": pl["placed"][1], "atempo": pl.get("atempo", 1.0) or 1.0, "text": pl.get("text", ""),
                              "burst": pl.get("burst")})
    parts.sort(key=lambda p: p["t0"]); slot = {u["id"]: (u["slot"]["start"], u["slot"]["end"]) for u in units}
    by_unit = {}
    for p in parts: by_unit.setdefault(p["unit"], []).append(p)
    words = [w for s in man["asr"]["segments"] for w in s.get("words", [])]

    def cause_of(t0, t1):
        tm = 0.5 * (t0 + t1)
        for u in units:
            s, e = slot[u["id"]]
            if s - 0.05 <= tm <= e + 0.05:
                ps = by_unit.get(u["id"], [])
                if not ps: return "unit_missing", u
                if tm < ps[0]["t0"]: return "late_start", u
                if tm > ps[-1]["t1"]: return "short_fill", u
                return "inner_gap", u
        return "between_units", None

    gaps = []
    for t0, t1 in runs(gapm):
        d = round(t1 - t0, 2)
        if d < a.min_gap_s: continue
        cause, u = cause_of(t0, t1)
        src = next(((s, e) for s, e in orig if s - 0.02 <= t0 and t1 <= e + 0.02), None)
        if src:
            head, tail = t0 - src[0] < 0.06, src[1] - t1 < 0.06
            pos = "whole" if head and tail else "head" if head else "tail" if tail else "inner"
        else: pos = "spans_pause"
        prev = next((p for p in reversed(parts) if p["t1"] <= t0 + 0.03), None); nxt = next((p for p in parts if p["t0"] >= t1 - 0.03), None)
        f0, f1 = int(t0 * FPS), min(nf, int(np.ceil(t1 * FPS)))
        sw = [w["word"].strip() for w in words if min(w["end"], t1) - max(w["start"], t0) > 0.5 * max(1e-3, w["end"] - w["start"])]
        g = {"t0": t0, "t1": t1, "seconds": d, "frames": [f0, f1], "ls_frames_fixed": int(ls_fixed[f0:f1].sum()), "ls_frames_regated": int(ls_regated[f0:f1].sum()),
             "natural": d < a.natural_s, "position_in_burst": pos, "source_burst": list(src) if src else None, "cause": cause,
             "unit": u["id"] if u else None, "sentence_src": u["text"] if u else None, "sentence_dub": (u.get("translation_fitted") or u["translation"]) if u else None,
             "source_words_in_gap": " ".join(sw),
             "prev_part": {"unit": prev["unit"], "text": prev["text"][-60:], "end": prev["t1"], "atempo": prev["atempo"]} if prev else None,
             "next_part": {"unit": nxt["unit"], "text": nxt["text"][:60], "start": nxt["t0"], "atempo": nxt["atempo"]} if nxt else None}
        if prev and prev["t1"] > t0 - 0.3:   # tempo that would stretch the previous part to the gap end (start kept)
            natural_len = (prev["t1"] - prev["t0"]) * prev["atempo"]; g["tempo_to_fill_by_prev"] = round(natural_len / max(1e-3, t1 - prev["t0"]), 3)
        if nxt and nxt["t0"] < t1 + 0.3: g["lead_to_fill_by_next_s"] = round(nxt["t0"] - t0, 2)
        gaps.append(g)
    big = lambda th: [g for g in gaps if g["seconds"] > th]
    at = np.array([p["atempo"] for p in parts]) if parts else np.array([1.0])
    rep = {"tag": a.tag, "dub_min_speech_ms": a.dub_min_speech_ms, "manifest": a.manifest, "ref_manifest": a.ref_manifest or a.manifest, "dub_track": str(dub16), "duration_s": dur,
           "source_speech_s": round(float(mo.sum() * R), 2), "dub_speech_s": round(float(md.sum() * R), 2),
           "source_without_dub_s": round(float(gapm.sum() * R), 2), "dub_without_source_s": round(float((md & ~mo).sum() * R), 2),
           "vad_iou": round(float((mo & md).sum() / max(1, (mo | md).sum())), 3),
           "gaps_gt": {str(th): {"n": len(big(th)), "s": round(sum(g["seconds"] for g in big(th)), 2),
                                 "n_in_ls": sum(1 for g in big(th) if g["ls_frames_fixed"] > 0), "ls_frames": sum(g["ls_frames_fixed"] for g in big(th))} for th in (0.3, 0.5, 1.0)},
           "gaps_by_cause_s": {c: round(sum(g["seconds"] for g in gaps if g["cause"] == c and not g["natural"]), 2) for c in sorted({g["cause"] for g in gaps})},
           "gaps_by_position_s": {c: round(sum(g["seconds"] for g in gaps if g["position_in_burst"] == c and not g["natural"]), 2) for c in sorted({g["position_in_burst"] for g in gaps})},
           "ls_frames": {"fixed": int(ls_fixed.sum()), "regated": int(ls_regated.sum())},
           "ls_frames_dub_silent_while_orig_rms": {"fixed": int((ls_fixed & rms_sil).sum()), "regated": int((ls_regated & rms_sil).sum())},
           "ls_frames_dub_silent_while_orig_vad": {"fixed": int((ls_fixed & gm).sum()), "regated": int((ls_regated & gm).sum())},
           "atempo": {"parts": len(parts), "max": round(float(at.max()), 3), "min": round(float(at.min()), 3), "share_lt_0_9": round(float((at < 0.9 - 1e-3).mean()), 3),
                      "share_lt_1": round(float((at < 0.995).mean()), 3), "share_gt_1_15": round(float((at > 1.15 + 1e-3).mean()), 3), "n_gt_1_15": int((at > 1.15 + 1e-3).sum())},
           "regated_gate": gate, "gaps": gaps}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = f"{pathlib.Path(man['input']).stem}_{a.tag}"
    (out / f"{stem}.gaps.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    gg = rep["gaps_gt"]
    print(f"== {stem}: src w/o dub {rep['source_without_dub_s']}s IoU {rep['vad_iou']} | gaps >0.3 {gg['0.3']['n']} ({gg['0.3']['s']}s) >0.5 {gg['0.5']['n']} ({gg['0.5']['s']}s) "
          f">1.0 {gg['1.0']['n']} ({gg['1.0']['s']}s) | LS silent-while-orig rms {rep['ls_frames_dub_silent_while_orig_rms']} vad {rep['ls_frames_dub_silent_while_orig_vad']} "
          f"| atempo max {rep['atempo']['max']} min {rep['atempo']['min']} <0.9 {rep['atempo']['share_lt_0_9']} >1.15 {rep['atempo']['share_gt_1_15']}")
    print(f"   by cause {rep['gaps_by_cause_s']} | by position {rep['gaps_by_position_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
