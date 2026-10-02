#!/usr/bin/env python3
"""OLD vs NEW audio candidate of one video, BEFORE LatentSync (2026-10-02 coverage fix).

Inputs: the runner manifests (+ their dubbed_16k.wav) of two runs of the same video and the scripts/diag/eval_run.py JSONs of their dubbed
audio. Reports, per side and as a delta: dubbed speech duration (Silero VAD, the gate's call), original speech without dub, VAD IoU, dub outside
the original speech, bursts < 50 % covered / uncovered, silent-while-speaking (25-fps frames where the original speaks and the dub is silent:
VAD proxy over the whole timeline AND the pilot RMS metric inside the LATENT_SYNC segments of a reference run), TTS WER / CER / bad units,
QA retries, interjections (hallucinations), missing final words, repeated words, lost numbers, atempo distribution / max, cut groups,
fit-up / split / tiny-merge facts. The acceptance block applies the stated criteria (intelligibility not worse, no stray um/ah, no repeats,
no cut words, no lost numbers, atempo <= 1.15 preferred / <= 1.2 hard, original speech without dub < 10 s).

    python scripts/pilot/compare_audio_candidates.py --old-manifest A.manifest.json --old-dubbed A/dubbed_16k.wav --old-eval A_eval.json \
        --new-manifest B.manifest.json --new-dubbed B/dubbed_16k.wav --new-eval B_eval.json --source-audio work_03/audio16k.wav --json out.json --md out.md
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "quality"))
FPS = 25; SR = 16000
BINS = [(0.0, 0.995, "<1"), (0.995, 1.005, "1.0"), (1.005, 1.05, "<=1.05"), (1.05, 1.10, "<=1.10"), (1.10, 1.15, "<=1.15"), (1.15, 1.20, "<=1.20"), (1.20, 9.0, ">1.20")]


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def inter(a, b):
    out = []; i = j = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total(iv):
    return round(sum(e - s for s, e in iv), 3)


def frame_mask(iv, n):
    m = np.zeros(n, dtype=bool)
    for s, e in iv:
        a, b = max(0, int(np.floor(s * FPS))), min(n, int(np.ceil(e * FPS)))
        if b > a:
            m[a:b] = True
    return m


def rms_db_per_frame(y: np.ndarray, sr: int, n: int) -> np.ndarray:
    hop = sr // FPS; out = np.full(n, -120.0)
    for i in range(n):
        seg = y[i * hop:(i + 1) * hop]
        if len(seg):
            out[i] = 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-9)
    return out


def side(man: dict, dubbed16: pathlib.Path, ev: dict | None, source16: pathlib.Path | None, ls_segments: list) -> dict:
    import soundfile as sf
    from face_aware_latentsync_accel import silero_speech_intervals
    from burst_aware_audio import missing_numbers
    dur = float(man["source"]["duration_s"]); n = int(round(dur * FPS))
    orig = merge([(v["start"], v["end"]) for v in man["speech_intervals"]]); dub = merge(silero_speech_intervals(dubbed16))
    both = merge(inter(orig, dub)); t_o, t_d, t_i = total(orig), total(dub), total(both)
    per = [{"burst": [s, e], "covered": round(total(inter([(s, e)], dub)) / max(e - s, 1e-6), 3)} for s, e in orig]
    mo, md = frame_mask(orig, n), frame_mask(dub, n)
    sws_vad = int((mo & ~md).sum())
    ls_mask = np.zeros(n, dtype=bool)
    for s in ls_segments:
        if s.get("action") == "LATENT_SYNC":
            ls_mask[s["start_frame"]:s["end_frame"]] = True
    rms = None
    if source16 is not None and source16.exists():
        yo, sro = sf.read(str(source16), dtype="float32"); yd, srd = sf.read(str(dubbed16), dtype="float32")
        if yo.ndim > 1: yo = yo.mean(1)
        if yd.ndim > 1: yd = yd.mean(1)
        do, dd = rms_db_per_frame(yo, sro, n), rms_db_per_frame(yd, srd, n)
        silent_dub = dd < -50.0; orig_loud = do > -35.0
        rms = {"ls_frames": int(ls_mask.sum()), "ls_frames_dub_silent": int((ls_mask & silent_dub).sum()), "ls_frames_dub_silent_while_original_speaks": int((ls_mask & silent_dub & orig_loud).sum()),
               "all_frames_dub_silent_while_original_speaks": int((silent_dub & orig_loud).sum())}
    units = man.get("units") or []; st = (man.get("tts") or {}).get("stats") or {}; al = man.get("alignment") or {}; summ = al.get("summary") or {}
    ats = [p.get("atempo", 1.0) for u in units for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    hist = {lab: sum(1 for x in ats if lo < x <= hi) for lo, hi, lab in BINS}
    cuts = [p.get("cut_at_slot_end_s", 0) or 0 for u in units for p in (u.get("alignment") or {}).get("placements", [])]
    cut_s = round(sum(cuts), 3); cut_max = round(max(cuts), 3) if cuts else 0.0
    lost = []
    for u in units:
        final = " ".join(u["parts"]) if u.get("parts") else u.get("translation", "")
        miss = missing_numbers(u.get("text", ""), final)
        if miss:
            lost.append({"unit": u["id"], "missing": miss, "source": u.get("text"), "final": final})
    rep = None
    if ev:
        rep = {"units": [], "count": 0}
        for u in ev.get("units", []):
            for key in ("whisper", "ctc"):
                toks = re.findall(r"[a-z']+", (u.get(key) or "").lower()); exp = re.findall(r"[a-z']+", (u.get("text") or "").lower())
                dups = [a for a, b in zip(toks, toks[1:]) if a == b and len(a) > 2 and not any(x == y for x, y in zip(exp, exp[1:]) if x == a)]
                if dups:
                    rep["units"].append({"unit": u.get("id"), "decoder": key, "repeated": dups}); rep["count"] += len(dups)
    evs = (ev or {}).get("summary") or {}
    return {"duration_s": dur, "orig_speech_s": t_o, "dub_speech_s": t_d, "overlap_s": t_i, "iou": round(t_i / max(t_o + t_d - t_i, 1e-6), 4), "orig_without_dub_s": round(t_o - t_i, 3),
            "dub_outside_orig_s": round(t_d - t_i, 3), "bursts": len(orig), "bursts_covered_lt_50pct": sum(1 for p in per if p["covered"] < 0.5), "bursts_uncovered": sum(1 for p in per if p["covered"] < 0.05),
            "silent_while_speaking_frames_vad": sws_vad, "silent_while_speaking_s_vad": round(sws_vad / FPS, 2), "rms_metric": rms, "per_burst": per,
            "units": len(units), "tts_mode": man.get("tts_mode") or (man.get("tts") or {}).get("mode", "parts"), "segmentation": (man.get("segmentation") or {}).get("mode"),
            "tiny_merges": len((man.get("tiny_unit_merge") or {}).get("merges") or []), "tts_stats": st, "translation_split": (man.get("translation") or {}).get("burst_split"),
            "eval": {k: evs.get(k) for k in ("wer_whisper_mean", "cer_ctc_mean", "cer_whisper_mean", "units_bad(cer_ctc>=0.3)", "interjection_units", "final_word_missing_units", "tts_parts", "retried_parts", "retry_success")},
            "repeated_words": rep, "lost_numbers": lost, "atempo": {"max": summ.get("atempo_max", max(ats) if ats else None), "gt_1_15": sum(x > 1.15 + 1e-9 for x in ats), "gt_1_20": sum(x > 1.20 + 1e-9 for x in ats),
                                                                  "mean": round(float(np.mean(ats)), 4) if ats else None, "histogram": hist, "groups": len(ats)},
            "cut_groups": summ.get("cut_groups"), "cut_seconds": cut_s, "cut_max_s": cut_max, "overflow_groups": summ.get("overflow_groups"), "units_slot_fallback": al.get("units_slot_fallback"),
            "units_table": [{"id": u["id"], "slot": [u["start"], u["end"]], "source": u.get("text"), "final": (" ".join(u["parts"]) if u.get("parts") else u.get("translation")), "fitted": bool(u.get("translation_fitted")),
                             "parts": len(u.get("tts_parts") or []), "tts_s": (u.get("tts") or {}).get("seconds"), "ratio_src": (u.get("tts") or {}).get("ratio_tts_to_source_speech"), "qa_bad": sum(bool(t.get("qa_bad_final")) for t in u.get("tts_parts") or []),
                             "fit_up": [x.get("result") for x in u.get("fit_up", [])], "placed": [p.get("placed") for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]} for u in units]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-manifest", required=True); ap.add_argument("--old-dubbed", required=True); ap.add_argument("--old-eval", default=None)
    ap.add_argument("--new-manifest", required=True); ap.add_argument("--new-dubbed", required=True); ap.add_argument("--new-eval", default=None)
    ap.add_argument("--source-audio", default=None, help="16 kHz mono wav of the source (work dir audio16k.wav) for the pilot RMS metric")
    ap.add_argument("--ls-reference-manifest", default=None, help="manifest whose LATENT_SYNC segments define the LS frame set (default: old manifest)")
    ap.add_argument("--target-without-dub-s", type=float, default=10.0)
    ap.add_argument("--json", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    old_m = json.loads(pathlib.Path(a.old_manifest).read_text()); new_m = json.loads(pathlib.Path(a.new_manifest).read_text())
    ref_m = json.loads(pathlib.Path(a.ls_reference_manifest).read_text()) if a.ls_reference_manifest else old_m
    ls_segs = (ref_m.get("lipsync") or {}).get("segments") or []
    ev_o = json.loads(pathlib.Path(a.old_eval).read_text()) if a.old_eval else None; ev_n = json.loads(pathlib.Path(a.new_eval).read_text()) if a.new_eval else None
    src16 = pathlib.Path(a.source_audio) if a.source_audio else None
    O = side(old_m, pathlib.Path(a.old_dubbed), ev_o, src16, ls_segs); N = side(new_m, pathlib.Path(a.new_dubbed), ev_n, src16, ls_segs)
    acc = {}
    eo, en = O["eval"], N["eval"]
    if ev_o and ev_n:
        acc["intelligibility_not_worse"] = (en["cer_ctc_mean"] <= eo["cer_ctc_mean"] + 0.01) and (en["wer_whisper_mean"] <= eo["wer_whisper_mean"] + 0.01) and (en["units_bad(cer_ctc>=0.3)"] <= eo["units_bad(cer_ctc>=0.3)"])
        acc["no_systematic_interjections"] = (en["interjection_units"] or 0) <= max(1, eo["interjection_units"] or 0)
    acc["no_repeated_words"] = (N["repeated_words"] or {}).get("count", 0) <= (O["repeated_words"] or {}).get("count", 0)   # no NEW repetition (an ASR-side "twenty twenty" for "2020" present in both sides is not a regression)
    acc["no_cut_words"] = (N["cut_groups"] or 0) == 0 and N["cut_max_s"] < 0.05   # per-span trim at the track end below 50 ms = the post-roll margin, not a word
    acc["no_lost_numbers"] = len(N["lost_numbers"]) == 0
    acc["atempo_hard_le_1_20"] = (N["atempo"]["gt_1_20"] == 0); acc["atempo_preferred_le_1_15"] = (N["atempo"]["gt_1_15"] == 0)
    acc["orig_without_dub_lt_target"] = N["orig_without_dub_s"] < a.target_without_dub_s
    acc["coverage_improved"] = N["orig_without_dub_s"] < O["orig_without_dub_s"] - 1.0
    acc["no_qa_bad_parts"] = (N["tts_stats"] or {}).get("final_bad", 0) == 0
    hard = ["intelligibility_not_worse", "no_systematic_interjections", "no_repeated_words", "no_cut_words", "no_lost_numbers", "atempo_hard_le_1_20", "orig_without_dub_lt_target", "coverage_improved"]
    verdict = "PASS" if all(acc.get(k, False) for k in hard) else "FAIL"
    out = {"old": {k: v for k, v in O.items() if k != "per_burst"}, "new": {k: v for k, v in N.items() if k != "per_burst"}, "per_burst": {"old": O["per_burst"], "new": N["per_burst"]},
           "acceptance": acc, "hard_criteria": hard, "verdict": verdict, "target_without_dub_s": a.target_without_dub_s, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pathlib.Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))

    def f(x, d=2):
        return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    rows = [("dubbed speech s (VAD)", f(O["dub_speech_s"]), f(N["dub_speech_s"])), ("original speech s", f(O["orig_speech_s"]), f(N["orig_speech_s"])), ("overlap s", f(O["overlap_s"]), f(N["overlap_s"])),
            ("original speech WITHOUT dub s", f(O["orig_without_dub_s"]), f(N["orig_without_dub_s"])), ("dub outside original s", f(O["dub_outside_orig_s"]), f(N["dub_outside_orig_s"])), ("VAD IoU", f(O["iou"], 3), f(N["iou"], 3)),
            (f"bursts < 50 % covered / uncovered (of {O['bursts']})", f"{O['bursts_covered_lt_50pct']} / {O['bursts_uncovered']}", f"{N['bursts_covered_lt_50pct']} / {N['bursts_uncovered']}"),
            ("silent-while-speaking frames, VAD proxy, whole timeline", f(O["silent_while_speaking_frames_vad"]), f(N["silent_while_speaking_frames_vad"]))]
    if O["rms_metric"] and N["rms_metric"]:
        rows += [("LS frames (reference segments) dub-silent (RMS < -50 dB)", f(O["rms_metric"]["ls_frames_dub_silent"]), f(N["rms_metric"]["ls_frames_dub_silent"])),
                 ("… of which original speaking (> -35 dB) = pilot metric", f(O["rms_metric"]["ls_frames_dub_silent_while_original_speaks"]), f(N["rms_metric"]["ls_frames_dub_silent_while_original_speaks"]))]
    rows += [("units / tts mode / segmentation / tiny merges", f"{O['units']} / {O['tts_mode']} / {O['segmentation']} / {O['tiny_merges']}", f"{N['units']} / {N['tts_mode']} / {N['segmentation']} / {N['tiny_merges']}"),
             ("TTS parts (generations) / attempts / retried (ok) / bad final", f"{O['tts_stats'].get('parts')} / {O['tts_stats'].get('attempts')} / {O['tts_stats'].get('retried_parts')} ({O['tts_stats'].get('retry_success')}) / {O['tts_stats'].get('final_bad')}",
              f"{N['tts_stats'].get('parts')} / {N['tts_stats'].get('attempts')} / {N['tts_stats'].get('retried_parts')} ({N['tts_stats'].get('retry_success')}) / {N['tts_stats'].get('final_bad')}"),
             ("fit-up units accepted / candidates / rejected", "-", f"{N['tts_stats'].get('fit_up_units')} / {N['tts_stats'].get('fit_up_candidates')} / {N['tts_stats'].get('fit_up_rejected')}"),
             ("Whisper WER / CTC CER / Whisper CER (unit mean)", f"{f(eo.get('wer_whisper_mean'),3)} / {f(eo.get('cer_ctc_mean'),3)} / {f(eo.get('cer_whisper_mean'),3)}", f"{f(en.get('wer_whisper_mean'),3)} / {f(en.get('cer_ctc_mean'),3)} / {f(en.get('cer_whisper_mean'),3)}"),
             ("bad units (CTC CER >= 0.3) / interjection units / final word missing", f"{eo.get('units_bad(cer_ctc>=0.3)')} / {eo.get('interjection_units')} / {eo.get('final_word_missing_units')}", f"{en.get('units_bad(cer_ctc>=0.3)')} / {en.get('interjection_units')} / {en.get('final_word_missing_units')}"),
             ("repeated words (ASR hypotheses)", f((O["repeated_words"] or {}).get("count")), f((N["repeated_words"] or {}).get("count"))), ("lost numbers (units)", f(len(O["lost_numbers"])), f(len(N["lost_numbers"]))),
             ("atempo max / > 1.15 / > 1.20 / mean (groups)", f"{O['atempo']['max']} / {O['atempo']['gt_1_15']} / {O['atempo']['gt_1_20']} / {O['atempo']['mean']} ({O['atempo']['groups']})", f"{N['atempo']['max']} / {N['atempo']['gt_1_15']} / {N['atempo']['gt_1_20']} / {N['atempo']['mean']} ({N['atempo']['groups']})"),
             ("atempo histogram <1 / 1.0 / <=1.05 / <=1.10 / <=1.15 / <=1.20 / >1.20", " / ".join(str(O["atempo"]["histogram"][k]) for _, _, k in BINS), " / ".join(str(N["atempo"]["histogram"][k]) for _, _, k in BINS)),
             ("cut groups (> 50 ms) / cut s total / max per span / overflow groups / slot-fallback units", f"{O['cut_groups']} / {O['cut_seconds']} / {O['cut_max_s']} / {O['overflow_groups']} / {O['units_slot_fallback']}", f"{N['cut_groups']} / {N['cut_seconds']} / {N['cut_max_s']} / {N['overflow_groups']} / {N['units_slot_fallback']}")]
    md = [f"# Audio candidate comparison — {pathlib.Path(old_m['input']).name}: OLD current vs NEW ({time.strftime('%Y-%m-%d')})", "", f"**Verdict: {verdict}** (hard criteria: {', '.join(k for k in hard)}; target original-speech-without-dub < {a.target_without_dub_s:g} s)", "",
          "| metric | OLD | NEW |", "|---|---:|---:|"] + [f"| {k} | {o} | {n} |" for k, o, n in rows] + ["", "Acceptance: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in acc.items()), ""]
    if N["lost_numbers"]:
        md += ["Lost numbers (NEW): " + json.dumps(N["lost_numbers"], ensure_ascii=False), ""]
    if (N["repeated_words"] or {}).get("count"):
        md += ["Repeated words (NEW): " + json.dumps(N["repeated_words"]["units"], ensure_ascii=False), ""]
    md += ["## NEW units", "", "| id | slot | source | final text (fitted*) | spans | tts s | tts/src-speech | fit-up | QA bad | placed |", "|---:|---|---|---|---:|---:|---:|---|---:|---|"]
    for r in N["units_table"]:
        md.append(f"| {r['id']} | {r['slot'][0]:.2f}-{r['slot'][1]:.2f} | {(r['source'] or '').replace('|', '/')} | {(r['final'] or '').replace('|', '/')}{'*' if r['fitted'] else ''} | {r['parts']} | {r['tts_s']} | {r['ratio_src']} | {' '.join(r['fit_up']) or '-'} | {r['qa_bad']} | {' '.join(f'{p[0]:.1f}-{p[1]:.1f}' for p in r['placed'])} |")
    md += ["", "## OLD units", "", "| id | slot | source | final text | parts | tts s | QA bad | placed |", "|---:|---|---|---|---:|---:|---:|---|"]
    for r in O["units_table"]:
        md.append(f"| {r['id']} | {r['slot'][0]:.2f}-{r['slot'][1]:.2f} | {(r['source'] or '').replace('|', '/')} | {(r['final'] or '').replace('|', '/')} | {r['parts']} | {r['tts_s']} | {r['qa_bad']} | {' '.join(f'{p[0]:.1f}-{p[1]:.1f}' for p in r['placed'])} |")
    pathlib.Path(a.md).write_text("\n".join(md) + "\n"); print("\n".join(md[:len(rows) + 8])); return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
