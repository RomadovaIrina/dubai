#!/usr/bin/env python3
"""Offline A/B of the phrase-cut / placement policy on a FINISHED phrase-mode run: same TTS generations (tts_u<id>_raw.wav + CTC word spans
from the manifest), same burst plans (rebuilt with run_clean_pipeline_05.burst_plan from the manifest's units / ASR / VAD), new cut policy ->
parts -> span merge -> align_units_burst -> build_dubbed_track (+ background mix) in a new work dir. Seconds on the CPU: no Qwen, no TTS, no
LatentSync. With the reference policy the run's own placements are reproduced (checked: --check). The new dubbed track can then go through
candidate_video_stage.py (--dubbed-dir) for the video.

    python scripts/quality/recut_offline.py --manifest RUN/final/03_final.manifest.json --out /workspace/dub/runs/recut/03_x --cut acoustic [--long-pause-func 0.8]
"""
from __future__ import annotations
import argparse, copy, json, pathlib, shutil, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality"): sys.path.insert(0, str(HERE.parent / d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cut", choices=["timing", "acoustic"], default="acoustic")
    ap.add_argument("--long-pause-func", type=float, default=0.0, help="acoustic: penalty for a cut after a function word / inside a fixed phrase, scaled by the source pause (0 = off)")
    ap.add_argument("--long-pause-lo", type=float, default=0.3); ap.add_argument("--long-pause-hi", type=float, default=0.8)
    ap.add_argument("--fill-slowdown", type=float, default=None); ap.add_argument("--background", choices=["on", "off"], default="on")
    ap.add_argument("--check", action="store_true", help="compare the placements with the run's own (reference policy must reproduce them)")
    a = ap.parse_args()
    import run_clean_pipeline_05 as rp, tts_burst as tb
    from e1_burst_align import PlaceOpts, burst_fit_ratios
    from tts_align import cut_parts
    from tts_qa import norm
    man = json.loads(pathlib.Path(a.manifest).read_text()); W = pathlib.Path(man["work_dir"]); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dur = man["source"]["duration_s"]; al = man["alignment"]
    units = copy.deepcopy(man["units"])
    base_units = [{k: u[k] for k in ("asr_segment", "start", "end", "text", "words", "id", "speaker")} for u in units]
    plans = rp.burst_plan(base_units, man["asr"]["segments"], man["speech_intervals"], dur, al.get("min_burst_s", 0.0), al.get("spill", True))
    opts = PlaceOpts(al["atempo_cap"], al["hard_cap"], True, 0.06, a.fill_slowdown if a.fill_slowdown is not None else al["fill_slowdown"], spill=al.get("spill", True),
                     max_lead=0.4)   # runner default --burst-max-lead; opts.min_burst_s = al.get("min_burst_s", 0.0); opts.fallback_frac = al.get("fallback_frac", 1.01)
    prev_end_of = {u["id"]: plans[i - 1]["slot"][1] for i, u in enumerate(units) if i > 0}
    cut_cfg = dict(tb.ACOUSTIC_CUT)
    if a.long_pause_func > 0: cut_cfg.update(w_func_long=a.long_pause_func, long_lo=a.long_pause_lo, long_hi=a.long_pause_hi)
    log = []
    for u, p in zip(units, plans):
        y, sr = sf.read(u["tts_raw_file"], dtype="float32"); spans = u["tts"]["word_spans"]
        text = u.get("translation_fitted") or u["translation"]; raw_words = text.split(); n = len(p["bursts"])
        if n == 1 or not spans:
            parts = [text]
        else:
            if a.cut == "acoustic": counts = tb.distribute_words_syntax(spans, raw_words, p["bursts"], norm, y=y, sr=sr, **cut_cfg)
            else: counts = tb.distribute_words(spans, raw_words, p["bursts"], norm)
            if counts is not None and 0 in counts:
                for j in range(len(counts) - 2, -1, -1):
                    if counts[j] == 0:
                        p["bursts"][j:j + 2] = [(p["bursts"][j][0], p["bursts"][j + 1][1])]; p["groups"][j:j + 2] = [p["groups"][j] + " " + p["groups"][j + 1]]
                        p["group_counts"][j:j + 2] = [p["group_counts"][j] + p["group_counts"][j + 1]]; counts[j:j + 2] = [counts[j + 1]]
            parts = []; k0 = 0; acc = 0; idx = 0
            for c in counts:
                wh = []
                while idx < len(raw_words) and acc < k0 + c:
                    acc += len(norm(raw_words[idx])); wh.append(raw_words[idx]); idx += 1
                parts.append(" ".join(wh)); k0 += c
        counts = [len(norm(x)) for x in parts]
        cp = cut_parts(y, sr, spans, counts, onset_s=tb.energy_onset(y, sr)); pieces = [(s_, e_) for _, s_, e_, _ in cp]
        chunks_of = lambda pcs: [y[int(s_ * sr):int(e_ * sr)] for s_, e_ in pcs]
        need = burst_fit_ratios(chunks_of(pieces), p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end_of.get(u["id"]))
        while len(pieces) > 1:
            over = [(k, r) for k, r in enumerate(need) if r is not None and r > opts.atempo_cap + 1e-3]
            if not over: break
            k, _ = max(over, key=lambda kr: kr[1]); nb = [j for j in (k - 1, k + 1) if 0 <= j < len(pieces)]
            j = min(nb, key=lambda j_: need[j_] if need[j_] is not None else 0.0); a_, b_ = sorted((k, j))
            pieces[a_:b_ + 1] = [(pieces[a_][0], pieces[b_][1])]; parts[a_:b_ + 1] = [parts[a_] + " " + parts[b_]]
            p["bursts"][a_:b_ + 1] = [(p["bursts"][a_][0], p["bursts"][b_][1])]; p["groups"][a_:b_ + 1] = [p["groups"][a_] + " " + p["groups"][b_]]; p["group_counts"][a_:b_ + 1] = [p["group_counts"][a_] + p["group_counts"][b_]]
            need = burst_fit_ratios(chunks_of(pieces), p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end_of.get(u["id"]))
        u["parts"] = parts; u["tts_parts"] = []
        for k, ((s_, e_), ptxt) in enumerate(zip(pieces, parts)):
            seg = y[int(s_ * sr):int(e_ * sr)]
            if a.cut == "acoustic": seg = tb.edge_fade(seg, sr, 5.0, 5.0)
            f = out / f"tts_u{u['id']:02d}_p{k}.wav"; sf.write(str(f), seg, sr, subtype="PCM_16")
            u["tts_parts"].append({"file": str(f), "seconds": round(len(seg) / sr, 3), "text": ptxt, "clip_span": [round(s_, 3), round(e_, 3)], "speech_span": [0.0, round(len(seg) / sr, 3)]})
        log.append({"unit": u["id"], "parts": parts})
    man2 = copy.deepcopy(man); man2["alignment"] = rp.align_units_burst(units, plans, out, opts)
    man2["dubbed_track"] = rp.build_dubbed_track(units, dur, out); man2["units"] = units; man2["work_dir"] = str(out)
    for f in ("audio16k.wav", "master_25fps.mp4", "bg_44k.wav", "vocals_44k.wav"):
        if (W / f).exists() and not (out / f).exists(): (out / f).symlink_to(W / f)
    if a.background == "on" and man.get("background", {}).get("enabled"):
        from background_bed import mix_background
        sp = [(v["start"], v["end"]) for v in man["speech_intervals"]]; dv = [(v["start"], v["end"]) for v in rp.run_vad(out / "dubbed_16k.wav")]
        man2["background"] = {"enabled": True, "separation": {**man["background"]["separation"], "bg": str(W / "bg_44k.wav"), "vocals": str(W / "vocals_44k.wav")},
                              "mix": mix_background({"bg": str(W / "bg_44k.wav"), "vocals": str(W / "vocals_44k.wav")}, out / "dubbed_24k.wav", sp, dv, dur, out)}
    man2["output"] = str(out / "no_video.mp4"); man2["recut"] = {"cut": a.cut, "cfg": cut_cfg, "source_manifest": a.manifest, "parts": log}
    (out / "recut.manifest.json").write_text(json.dumps(man2, indent=1, ensure_ascii=False))
    if a.check:
        ref = [(pl["placed"], pl.get("atempo")) for u in man["units"] for pl in u["alignment"]["placements"] if pl.get("placed")]
        new = [(pl["placed"], pl.get("atempo")) for u in units for pl in u["alignment"]["placements"] if pl.get("placed")]
        same = len(ref) == len(new) and all(abs(x[0][0] - y_[0][0]) < 0.002 and abs(x[0][1] - y_[0][1]) < 0.002 for x, y_ in zip(ref, new))
        print(f"check: {len(ref)} ref placements vs {len(new)} new -> {'IDENTICAL' if same else 'DIFFERENT'}")
    diff = [(l["unit"], l["parts"], [t["text"] for t in mu.get("tts_parts", [])]) for l, mu in zip(log, man["units"]) if l["parts"] != [t["text"] for t in mu.get("tts_parts", [])]]
    for uid, new_p, old_p in diff: print(f"  u{uid:02d} OLD {old_p}\n      NEW {new_p}")
    print(f"-> {out / 'recut.manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
