#!/usr/bin/env python3
"""Foreign-sound localisation for one burst/phrase-mode run ("посторонние звуки", 2026-10-13 feedback): find candidate events on the final
timeline and trace every one through the audio stages to the FIRST stage where it appears.

Stages (work dir of run_clean_pipeline_05.py): original (audio16k.wav) | raw TTS (tts_u<id>_raw.wav, the final QA/fit generation) | cut part
(tts_u<id>_p<k>.wav) | placed + atempo unit track (aligned_u<id>.wav) | assembled dub voice (dubbed_24k.wav) | final mix (final_mix_44k.wav, dub +
background bed) | muxed MP4 audio (decoded AAC).

Event detectors (deterministic, no listening):
  click        sample discontinuity: |x[t]-x[t-1]| > 8x its local (+-10 ms) median and > 0.03 FS, in the dub voice and in the final mix
  hot_edge     a placed part starts / ends while the voice is loud: RMS of its first / last 10 ms within 12 dB of the part's p90 (+-100 ms profile kept)
  extra_voice  energy in the raw TTS OUTSIDE the CTC word spans (> 80 ms, within 20 dB of the speech level) that lies inside an exported part:
               breath / garble / hallucinated syllable carried into the dub
  overlap      two placed parts (of any units) overlapping in time > 10 ms: two dub voices at once
  bed_voice    Silero speech in the background bed (= final mix - dub voice) at >= -30 dB rel. dub voice: residual source speech / separation
               artifacts heard in the dub pauses or under the dub
  bed_step     the bed level changes by > 8 dB within 100 ms where the dub is silent (ducking release / room-fill seams)
Outputs <out>/<id>_<tag>.audio_stages.json and spectrogram sheets (rows = stages, +-0.6 s) for the worst events.

    python scripts/quality/audio_stages.py --manifest RUN/final/03_final.manifest.json --out RUN/audio_stages --tag reviewed
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, subprocess, sys, tempfile
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))
SR = 24000



def load(p, sr=SR) -> np.ndarray:
    p = str(p); info = None
    try: info = sf.info(p)
    except Exception: pass
    if info is None or info.samplerate != sr or info.channels != 1:
        t = pathlib.Path(tempfile.mkdtemp()) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le", str(t)], check=True)
        y, _ = sf.read(str(t), dtype="float32"); t.unlink(); t.parent.rmdir(); return y if y.ndim == 1 else y.mean(1)
    y, _ = sf.read(p, dtype="float32"); return y if y.ndim == 1 else y.mean(1)


def rms_db(y, a, b, sr=SR) -> float:
    a, b = max(0, int(a * sr)), min(len(y), int(b * sr))
    if b <= a: return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-9))


def frame_db(y, sr=SR, hop_s=0.01) -> np.ndarray:
    h = int(hop_s * sr); n = len(y) // h
    return 20 * np.log10(np.sqrt((y[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)


def clicks(y, sr=SR, k=10.0, floor=0.01, min_sep_s=0.02) -> list:
    """Impulsive clicks: 1 ms peak of the > 5 kHz band more than k x its +-25 ms median AND isolated (the +-3 ms neighbours stay < k/3 x median),
    peak > floor. Plosives / fricatives are excluded by the isolation test (their HF energy lasts > 5 ms)."""
    from scipy.signal import butter, sosfilt
    if len(y) < sr // 10: return []
    hp = np.abs(sosfilt(butter(4, 5000, "highpass", fs=sr, output="sos"), y)); f = max(1, sr // 1000); n = len(hp) // f
    pk = hp[:n * f].reshape(n, f).max(1)
    from scipy.ndimage import median_filter
    med = np.maximum(median_filter(pk, size=51, mode="nearest"), 1e-5)
    out = []; last = -1e9
    for i in np.where((pk > k * med) & (pk > floor))[0]:
        lo, hi = max(0, i - 4), min(n, i + 5); nb = np.concatenate([pk[lo:max(lo, i - 1)], pk[min(hi, i + 2):hi]])
        if len(nb) and nb.max() > (k / 3) * med[i]: continue
        t = i / 1000.0
        if t - last >= min_sep_s: out.append({"t": round(t, 3), "peak": round(float(pk[i]), 4), "x_local": round(float(pk[i] / med[i]), 1)}); last = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--sheets", type=int, default=10); ap.add_argument("--video", default=None)
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); W = pathlib.Path(man["work_dir"]); vid = a.video or man["output"]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = f"{pathlib.Path(man['input']).stem}_{a.tag}"
    dur = man["source"]["duration_s"]
    orig = load(W / "audio16k.wav"); dub = load(W / "dubbed_24k.wav")
    mix = load(W / "final_mix_44k.wav") if (W / "final_mix_44k.wav").exists() else None
    mp4 = load(vid) if pathlib.Path(vid).exists() else (mix if mix is not None else dub)   # offline recut: no mp4 yet
    units = man["units"]
    # ---- placed parts on the timeline (part k <-> placement k; slot fallback = one placement for the concatenated parts)
    placed = []
    for u in units:
        pls = u["alignment"]["placements"]; parts = u.get("tts_parts", [])
        raw = u.get("tts_raw_file")
        for k, pl in enumerate(pls):
            if not pl.get("placed"): continue
            part = parts[k] if k < len(parts) and len(pls) == len(parts) else None
            placed.append({"unit": u["id"], "k": k, "t0": pl["placed"][0], "t1": pl["placed"][1], "atempo": pl.get("atempo", 1.0), "text": pl.get("text", ""),
                           "clip_span": part.get("clip_span") if part else None, "part_file": part.get("file") if part else None, "raw_file": raw,
                           "word_spans": u.get("tts", {}).get("word_spans"), "cut_at_slot_end_s": pl.get("cut_at_slot_end_s", 0.0)})
    placed.sort(key=lambda p: p["t0"])
    voice_db = np.percentile(frame_db(dub)[frame_db(dub) > -60], 90) if (frame_db(dub) > -60).any() else -20.0
    events = []
    # overlap
    for p, q in zip(placed, placed[1:]):
        ov = p["t1"] - q["t0"]
        if ov > 0.01:
            events.append({"type": "overlap", "t": round(q["t0"], 3), "seconds": round(ov, 3), "units": [p["unit"], q["unit"]], "texts": [p["text"][-40:], q["text"][:40]]})
    # hot edges + +-100 ms RMS profile at every placed edge
    edges = []
    for p in placed:
        seg_db = frame_db(dub[int(p["t0"] * SR):int(p["t1"] * SR)])
        ref = float(np.percentile(seg_db, 90)) if len(seg_db) else -60
        head = rms_db(dub, p["t0"], p["t0"] + 0.01); tail = rms_db(dub, p["t1"] - 0.01, p["t1"])
        prof = [round(rms_db(dub, p["t1"] + d - 0.005, p["t1"] + d + 0.005), 1) for d in np.arange(-0.1, 0.101, 0.02)]
        e = {"unit": p["unit"], "k": p["k"], "t0": p["t0"], "t1": p["t1"], "head_rel_db": round(head - ref, 1), "tail_rel_db": round(tail - ref, 1), "tail_profile_db_-100..+100ms": prof,
             "text": p["text"], "atempo": p["atempo"], "cut_at_slot_end_s": p["cut_at_slot_end_s"]}
        edges.append(e)
        if e["tail_rel_db"] > -12: events.append({"type": "hot_edge", "edge": "tail", "t": round(p["t1"], 3), "rel_db": e["tail_rel_db"], "unit": p["unit"], "k": p["k"], "text": p["text"][-40:]})
        if e["head_rel_db"] > -12: events.append({"type": "hot_edge", "edge": "head", "t": round(p["t0"], 3), "rel_db": e["head_rel_db"], "unit": p["unit"], "k": p["k"], "text": p["text"][:40]})
    # extra voice in raw TTS carried into a part
    seen_raw = set()
    for p in placed:
        if not p["raw_file"] or not p["word_spans"] or not p["clip_span"] or not pathlib.Path(p["raw_file"]).exists(): continue
        y = load(p["raw_file"]); fdb = frame_db(y); ref = float(np.percentile(fdb[fdb > -60], 90)) if (fdb > -60).any() else -20
        cov = np.zeros(len(fdb), bool)
        for w in p["word_spans"]: cov[max(0, int((w["start"] - 0.04) / 0.01)):int((w["end"] + 0.10) / 0.01) + 1] = True
        loud = (fdb > ref - 20) & ~cov
        c0, c1 = int(p["clip_span"][0] / 0.01), int(p["clip_span"][1] / 0.01)
        i = c0
        while i < min(c1, len(loud)):
            if loud[i]:
                j = i
                while j < min(c1, len(loud)) and loud[j]: j += 1
                if j - i >= 8:
                    rt0, rt1 = i * 0.01, j * 0.01; tt = p["t0"] + (rt0 - p["clip_span"][0]) / max(p["atempo"], 1e-3)
                    events.append({"type": "extra_voice", "t": round(tt, 3), "seconds": round((rt1 - rt0) / max(p["atempo"], 1e-3), 3), "rel_db": round(float(fdb[i:j].max() - ref), 1),
                                   "unit": p["unit"], "k": p["k"], "raw_span": [round(rt0, 2), round(rt1, 2)], "text": p["text"][:50]})
                i = j
            else: i += 1
        if p["unit"] not in seen_raw:
            seen_raw.add(p["unit"])
    # clicks: dub voice / mix / mp4
    clk = {"dub_voice": clicks(dub), "final_mix": clicks(mix) if mix is not None else [], "mp4": clicks(mp4)}
    edge_ts = np.array(sorted([p["t0"] for p in placed] + [p["t1"] for p in placed])) if placed else np.zeros(0)
    raw_clicks = {}
    for c in clk["dub_voice"]:
        d = float(np.min(np.abs(edge_ts - c["t"]))) if len(edge_ts) else 9.9
        e = {"type": "click", "stage": "dub_voice", "t": c["t"], "x_local": c["x_local"], "peak": c["peak"], "nearest_part_edge_ms": round(1000 * d)}
        p = next((p for p in placed if p["t0"] - 0.005 <= c["t"] <= p["t1"] + 0.005), None)
        if p is not None and p["raw_file"] and p["clip_span"]:
            if p["raw_file"] not in raw_clicks: raw_clicks[p["raw_file"]] = np.array([x["t"] for x in clicks(load(p["raw_file"]))])
            rt = p["clip_span"][0] + (c["t"] - p["t0"]) * p["atempo"]; rc = raw_clicks[p["raw_file"]]
            e["in_raw_tts"] = bool(len(rc) and np.min(np.abs(rc - rt)) < 0.006 * max(p["atempo"], 1.0)); e["atempo"] = p["atempo"]
        e["origin"] = "raw_tts" if e.get("in_raw_tts") else ("cut_edge" if d < 0.015 else ("atempo" if p is not None and abs(p["atempo"] - 1) > 0.01 else "unexplained"))
        events.append(e)
    dub_clk = np.array([c["t"] for c in clk["dub_voice"]])
    for c in clk["final_mix"]:
        if len(dub_clk) and np.min(np.abs(dub_clk - c["t"])) < 0.02: continue
        events.append({"type": "click", "stage": "final_mix_only", "t": c["t"], "x_local": c["x_local"], "peak": c["peak"]})
    # background bed = mix - dub voice (the mix normalisation factor is in the manifest)
    bg = man.get("background", {}); bed_info = None
    if mix is not None and bg.get("enabled"):
        # bed exactly as background_bed.mix_background builds it: bg (drums+bass+other) x gain x duck(dub VAD) (room fill ignored: reported in the manifest)
        from background_bed import _envelope, _mask
        import run_clean_pipeline_05 as rp
        mx = bg["mix"]; dvad = [(v["start"], v["end"]) for v in rp.run_vad(W / "dubbed_16k.wav")]
        if pathlib.Path(bg["separation"].get("bg", "")).exists() and not (mx.get("residue_duck") or {}).get("enabled"):
            bgy = load(bg["separation"]["bg"]); n_ = len(bgy)
            env = _envelope(_mask(dvad, n_, SR), SR, mx["attack_s"], mx["release_s"])
            bed = bgy * (10 ** (mx["bg_gain_db"] / 20)) * (10 ** (mx["duck_db"] * env / 20)) * mx.get("peak_norm", 1.0)
            if mx.get("src_duck_db"):
                so = _mask([(v["start"], v["end"]) for v in man["speech_intervals"]], n_, SR) & ~_mask(dvad, n_, SR)
                bed = bed * (10 ** (mx["src_duck_db"] * _envelope(so, SR, mx["attack_s"], mx["release_s"]) * (1.0 - env) / 20))
        else:   # r2: the bed AS HEARD = final mix - peak_norm x dub voice (stems not kept; covers residue duck / room fill / any later mix option)
            # at 44.1 kHz with the mixer's own ffmpeg conversion of the dub voice (a 24 kHz subtraction leaves the dub in at about -11 dB)
            td = pathlib.Path(tempfile.mkdtemp())
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(W / "dubbed_24k.wav"), "-ar", "44100", "-ac", "1", "-c:a", "pcm_f32le", str(td / "d44.wav")], check=True)
            m44, _ = sf.read(str(W / "final_mix_44k.wav"), dtype="float32"); d44, _ = sf.read(str(td / "d44.wav"), dtype="float32")
            n44 = min(len(m44), len(d44)); b44 = m44[:n44].mean(1) - mx.get("peak_norm", 1.0) * d44[:n44]
            sf.write(str(td / "b44.wav"), b44, 44100, subtype="FLOAT"); bed = load(td / "b44.wav"); n_ = len(bed); shutil.rmtree(td)
        b16 = out / f"{stem}_bed16k.wav"; sf.write(str(b16), bed, SR); y16 = load(b16, 16000); sf.write(str(b16), y16, 16000)
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        vad = get_speech_timestamps(torch.from_numpy(y16), load_silero_vad(), sampling_rate=16000, return_seconds=True)
        dub_mask = frame_db(dub) > voice_db - 35
        bdb = frame_db(bed); n_ = min(len(bdb), len(dub_mask))
        bed_info = {"bed_rms_db": round(rms_db(bed, 0, dur), 1), "dub_voice_p90_db": round(float(voice_db), 1), "bed_speech_intervals": len(vad),
                    "bed_speech_seconds": round(sum(v["end"] - v["start"] for v in vad), 2)}
        for v in vad:
            lv = rms_db(bed, v["start"], v["end"]); i0, i1 = int(v["start"] / 0.01), int(v["end"] / 0.01)
            in_dub_pause = float(1 - dub_mask[i0:min(i1, n_)].mean()) if i1 > i0 else 0.0
            if lv - voice_db > -30:
                events.append({"type": "bed_voice", "t": round(v["start"], 3), "seconds": round(v["end"] - v["start"], 2), "rel_db": round(lv - voice_db, 1), "frac_in_dub_pause": round(in_dub_pause, 2),
                               "orig_rel_db": round(rms_db(orig, v["start"], v["end"], 16000) - voice_db, 1)})
        # bed level steps in dub pauses
        sm = np.convolve(bdb[:n_], np.ones(5) / 5, mode="same")
        for i in range(10, n_ - 10):
            if not dub_mask[i] and abs(sm[i + 5] - sm[i - 5]) > 8 and sm[max(i - 5, 0):i + 6].max() > voice_db - 40:
                if not events or events[-1].get("type") != "bed_step" or i * 0.01 - events[-1]["t"] > 0.3:
                    events.append({"type": "bed_step", "t": round(i * 0.01, 2), "step_db": round(float(sm[i + 5] - sm[i - 5]), 1), "bed_db": round(float(sm[i]), 1)})
        b16.unlink(missing_ok=True)
        import soundfile as _sf
        bp = out / f"{stem}_bed.wav"; _sf.write(str(bp), bed.astype(np.float32), SR)   # listening copy (not committed)
        bed_info["bed_file"] = str(bp)
    # ---- stage trace per event: RMS (dB) in +-100 ms around t at every stage (raw / part mapped through the placement)
    def part_at(t):
        for p in placed:
            if p["t0"] - 0.02 <= t <= p["t1"] + 0.02: return p
        return None
    for e in events:
        t = e["t"]; tr = {"original": round(rms_db(orig, t - 0.1, t + 0.1, 16000), 1), "dub_voice": round(rms_db(dub, t - 0.1, t + 0.1), 1),
                          "final_mix": round(rms_db(mix, t - 0.1, t + 0.1), 1) if mix is not None else None, "mp4": round(rms_db(mp4, t - 0.1, t + 0.1), 1)}
        p = part_at(t)
        if p is not None and p["clip_span"]:
            rt = p["clip_span"][0] + (t - p["t0"]) * p["atempo"]; tr["raw_tts_t"] = round(rt, 3)
            if p["raw_file"] and pathlib.Path(p["raw_file"]).exists(): tr["raw_tts"] = round(rms_db(load(p["raw_file"]), rt - 0.1 * p["atempo"], rt + 0.1 * p["atempo"]), 1)
            if p["part_file"] and pathlib.Path(p["part_file"]).exists(): tr["part"] = round(rms_db(load(p["part_file"]), rt - p["clip_span"][0] - 0.1, rt - p["clip_span"][0] + 0.1), 1)
            tr["unit"] = p["unit"]; tr["part_k"] = p["k"]; tr["atempo"] = p["atempo"]
        e["trace"] = tr
        # first stage verdict (by detector type)
        if e["type"] in ("bed_voice", "bed_step"): e["first_stage"] = "final_mix (background bed)"
        elif e["type"] == "extra_voice": e["first_stage"] = "raw_tts (carried into the part by the cut)"
        elif e["type"] == "overlap": e["first_stage"] = "placement (two parts at once)"
        elif e["type"] == "hot_edge":
            e["first_stage"] = "cut (raw TTS still voiced at the cut)" if tr.get("raw_tts", -120) > voice_db - 25 else "placement/atempo"
        elif e["type"] == "click": e["first_stage"] = {"raw_tts": "raw_tts", "cut_edge": "cut (part edge)", "atempo": "atempo", "unexplained": "dub_voice (?)"}.get(e.get("origin"), "final_mix only (bed / mix)")
    # ---- spectrogram sheets of the worst events (rows = stages)
    def score(e):
        return {"bed_voice": 3 + e.get("rel_db", -30) / 10, "extra_voice": 3 + e.get("rel_db", -20) / 10, "overlap": 4, "click": 2 + e.get("x_local", 0) / 20,
                "hot_edge": 1 + e.get("rel_db", -12) / 6, "bed_step": 1}[e["type"]]
    events.sort(key=lambda e: e["t"])
    worst = sorted(events, key=lambda e: -score(e))[:a.sheets]; sheets = []
    if worst:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for e in worst:
            t = e["t"]; rows = [("original", orig, 16000), ("dub voice (assembled)", dub, SR)] + ([("final mix", mix, SR)] if mix is not None else []) + [("mp4", mp4, SR)]
            p = part_at(t)
            if p is not None and p["raw_file"] and p["clip_span"] and pathlib.Path(p["raw_file"]).exists():
                rows.insert(1, (f"raw TTS u{p['unit']} @ {e['trace'].get('raw_tts_t')}s (atempo {p['atempo']})", load(p["raw_file"]), SR, e["trace"].get("raw_tts_t")))
            fig, axs = plt.subplots(len(rows), 1, figsize=(9, 1.6 * len(rows)), sharex=False)
            for ax, r in zip(axs, rows):
                name, y, sr = r[:3]; tc = r[3] if len(r) > 3 else t
                a0, a1 = max(0, int((tc - 0.6) * sr)), min(len(y), int((tc + 0.6) * sr)); seg = y[a0:a1]
                if len(seg) > 256: ax.specgram(seg, NFFT=512, Fs=sr, noverlap=384, cmap="magma", vmin=-120)
                ax.axvline(tc - a0 / sr, color="cyan", lw=0.8); ax.set_ylabel(name, fontsize=7, rotation=0, ha="right"); ax.set_ylim(0, 8000); ax.tick_params(labelsize=6)
            fig.suptitle(f"{stem} {e['type']} t={t}s " + json.dumps({k: v for k, v in e.items() if k in ('rel_db', 'seconds', 'x_local', 'text', 'units', 'edge')}, ensure_ascii=False)[:150], fontsize=7)
            fig.tight_layout(); pth = out / f"{stem}_ev_{e['type']}_{t:07.2f}.png"; fig.savefig(pth, dpi=90); plt.close(fig); sheets.append(str(pth))
    by = {}
    for e in events: by[e["type"]] = by.get(e["type"], 0) + 1
    rep = {"manifest": a.manifest, "duration_s": dur, "dub_voice_p90_db": round(float(voice_db), 1), "placed_parts": len(placed), "counts": by,
           "clicks": {k: len(v) for k, v in clk.items()}, "bed": bed_info, "events": events, "edges": edges, "sheets": sheets}
    (out / f"{stem}.audio_stages.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"== {stem}: parts {len(placed)}, events {by}, clicks {rep['clicks']}, bed {bed_info}")
    for e in sorted(events, key=lambda e: -score(e))[:15]:
        print(f"   {e['t']:7.2f}s {e['type']:<11} {e.get('first_stage', '')[:40]:<40} " + json.dumps({k: v for k, v in e.items() if k not in ('trace', 't', 'type', 'first_stage')}, ensure_ascii=False)[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
