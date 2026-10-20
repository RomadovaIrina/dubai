#!/usr/bin/env python3
"""Round 4 audio gate metrics for ONE final mix (an mp4 or an audio file). They run on the mix alone, so a published reference (v4 mp4)
is scored on the same axes as a new run.
The bed of every DabAI mix is a time-varying gain c(t) times the source's Spleeter accompaniment stem bg(t) (duck / residue duck / static gain), so
  c(t)  = least-squares projection of the mix onto bg(t) in 50 ms windows (the dub voice is uncorrelated with bg),
  bed   = c(t) * bg(t)          dub_est = mix - bed   (VAD on dub_est = dub speech intervals; a run's own dubbed_16k.wav is used when given).
Metrics:
  residue_*      original-voice residue: the bed's 300-3400 Hz level where the SOURCE speaks, relative to the dub voice level (dB)
                 under_dub = source speaks and the dub speaks; exposed = the source speaks and the dub is silent (heard as Russian mumbling)
                 + bed_vad_s: seconds where Silero hears speech in the bed AND the bed's speech band is within 25 dB of the dub voice
  pause_bed_*    bed level in joint pauses (source and dub silent) relative to the voice, minus the same ratio in the source (dB; > 0 = the
                 noise comes forward); jump_p95 = p95 of the 400 ms level rise of the mix at dub-pause onsets vs the source at the same moments
  pauses_*       dub vs source pause (gap) distribution inside [first, last speech]: count, median, p90, > 1 s, Wasserstein-1 (s),
                 source-speaks-dub-silent / dub-speaks-source-silent seconds
  loudness       integrated LUFS / true peak / LRA;  duration
    python scripts/quality/r4_audio_metrics.py --mix OUT.mp4 --src SRC.mp4 --sep SEPDIR [--run RUN_WORK_DIR] [--manifest M] [--wer] --json out.json"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE))
SR = 44100; H = int(0.05 * SR)


def load(p, sr=SR, ch=1):
    with tempfile.TemporaryDirectory() as td:
        w = pathlib.Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), "-vn", "-ac", str(ch), "-ar", str(sr), "-c:a", "pcm_f32le", str(w)], check=True)
        y, _ = sf.read(str(w), dtype="float32")
    return y


def separation(src, sep: pathlib.Path):
    """cached Spleeter stems of the source (same separator / call as the pipeline -> the identical bg the mixes were built from)."""
    if not (sep / "bg_44k.wav").exists():
        import background_bed as bb
        sep.mkdir(parents=True, exist_ok=True); bb.separate_background(pathlib.Path(src), sep)
    return sep / "bg_44k.wav", sep / "vocals_44k.wav"


def vad(y, sr):
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    import scipy.signal as ss
    y16 = ss.resample_poly(y, 160, sr // 100).astype(np.float32) if sr != 16000 else y
    ts = get_speech_timestamps(torch.from_numpy(y16), load_silero_vad(), sampling_rate=16000, return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def mask(ivs, n, hop):
    m = np.zeros(n, bool)
    for s, e in ivs: m[max(0, int(s / hop)):min(n, int(np.ceil(e / hop)))] = True
    return m


def frames(y, h=H):
    n = len(y) // h; return y[: n * h].reshape(n, h)


def band(y, lo=300, hi=3400):
    import scipy.signal as ss
    return ss.sosfiltfilt(ss.butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y).astype(np.float32)


def db(p): return float(10 * np.log10(np.mean(p) + 1e-12)) if np.size(p) else None


def gaps(ivs, lo, hi):
    iv = [(max(s, lo), min(e, hi)) for s, e in ivs if e > lo and s < hi]
    return [b[0] - a[1] for a, b in zip(iv, iv[1:]) if b[0] - a[1] > 0.05]


def pause_stats(g):
    g = np.asarray(g)
    return {"count": int(len(g)), "median": round(float(np.median(g)), 3) if len(g) else None, "p90": round(float(np.percentile(g, 90)), 3) if len(g) else None,
            "gt_1s": int((g > 1.0).sum()), "total_s": round(float(g.sum()), 2)}


def score(mix_p, src_p, sep: pathlib.Path, run: pathlib.Path | None = None, manifest=None, do_wer=False, ne_from=None, series=None):
    bg_p, voc_p = separation(src_p, sep)
    mix = load(mix_p); src = load(src_p); bg = sf.read(str(bg_p), dtype="float32")[0].mean(1); voc = sf.read(str(voc_p), dtype="float32")[0].mean(1)
    n = min(len(mix), len(bg)); m_, b_ = frames(mix[:n]), frames(bg[:n]); nf = len(m_)
    c = (m_ * b_).sum(1) / ((b_ * b_).sum(1) + 1e-9)
    c = np.clip(np.convolve(c, np.ones(3) / 3, mode="same"), 0, None)       # 150 ms smoothing: the duck envelopes are >= 40 ms ramps
    bed = (b_ * c[:, None]).reshape(-1); dub_est = mix[: len(bed)] - bed
    dub_wav = (run / "dubbed_16k.wav") if run and (run / "dubbed_16k.wav").exists() else None
    if dub_wav is not None:
        d16, _ = sf.read(str(dub_wav), dtype="float32"); dub_iv = vad(d16, 16000); dub_src = "run dubbed_16k.wav"
    else:
        dub_iv = vad(dub_est, SR); dub_src = "VAD on mix - c*bg"
    src_iv = [(s["start"], s["end"]) for s in json.loads(pathlib.Path(manifest).read_text())["speech_intervals"]] if manifest else vad(voc, SR)
    S, D = mask(src_iv, nf, H / SR), mask(dub_iv, nf, H / SR)
    p_bed_sb = (frames(band(bed)) ** 2).mean(1); p_mix = (m_ ** 2).mean(1); p_src = (frames(src[: nf * H]) ** 2).mean(1)
    voice = db(p_mix[D])                                                                  # dub voice (+ ducked bed) level
    if series: np.savez_compressed(series, S=S, D=D, p_bed_sb=p_bed_sb, p_mix=p_mix, p_src=p_src, voice=voice or 0.0, hop=H / SR)   # 50 ms frames -> r4_review.py
    r = {"dub_intervals_from": dub_src, "duration_s": round(len(mix) / SR, 3), "source_duration_s": round(len(src) / SR, 3),
         "dub_voice_db": round(voice, 2) if voice is not None else None}
    for k, sel in (("under_dub", S & D), ("exposed", S & ~D)):
        v = db(p_bed_sb[sel]); r[f"residue_{k}_db"] = round(v - voice, 2) if v is not None and voice is not None else None
        r[f"residue_{k}_s"] = round(float(sel.sum() * H / SR), 2)
    try:   # Silero on the reconstructed bed: speech the listener can hear (bed speech band within 25 dB of the dub voice)
        biv = vad(bed, SR); B = mask(biv, nf, H / SR); loud = 10 * np.log10(p_bed_sb + 1e-12) > (voice or 0) - 25
        r["bed_vad_s"] = round(float((B & S & loud).sum() * H / SR), 2); r["bed_vad_exposed_s"] = round(float((B & S & ~D & loud).sum() * H / SR), 2)
    except Exception as e: r["bed_vad_error"] = str(e)[:200]
    P = ~S & ~D; Ssp = S
    if P.any() and voice is not None and Ssp.any():
        r["pause_bed_rel_db"] = round((db(p_mix[P]) - voice) - (db(p_src[P]) - db(p_src[Ssp])), 2)
        L = lambda p: 10 * np.log10(np.convolve(p, np.ones(8) / 8, mode="same") + 1e-12)   # 400 ms level
        Lm, Ls = L(p_mix), L(p_src); on = np.flatnonzero(np.diff(D.astype(np.int8)) == -1) + 1; jumps = []
        for i in on:
            if i - 8 >= 0 and i + 8 < nf and not S[i:i + 8].any(): jumps.append((Lm[i + 8] - Lm[i - 8]) - (Ls[i + 8] - Ls[i - 8]))
        r["pause_jump_p95_db"] = round(float(np.percentile(jumps, 95)), 2) if jumps else None; r["pause_jump_n"] = len(jumps)
        pm = 10 * np.log10(p_mix[P] + 1e-12) - voice; ps_ = 10 * np.log10(p_src[P] + 1e-12) - db(p_src[Ssp])
        r["pause_bed_p95_rel_db"] = round(float(np.percentile(pm, 95) - np.percentile(ps_, 95)), 2)
    lo, hi = min(src_iv[0][0], dub_iv[0][0]) if src_iv and dub_iv else 0, max(src_iv[-1][1], dub_iv[-1][1]) if src_iv and dub_iv else 0
    gs, gd = gaps(src_iv, lo, hi), gaps(dub_iv, lo, hi)
    from scipy.stats import wasserstein_distance
    r["pauses_source"], r["pauses_dub"] = pause_stats(gs), pause_stats(gd)
    r["pauses_w1_s"] = round(float(wasserstein_distance(gs, gd)), 3) if gs and gd else None
    r["src_speaks_dub_silent_s"] = round(float((S & ~D).sum() * H / SR), 2); r["dub_speaks_src_silent_s"] = round(float((D & ~S).sum() * H / SR), 2)
    import r3_scorecard as sc
    r["loudness"] = sc.loudness(mix_p)
    if run is not None and (run / "dubbed_24k.wav").exists():
        import r3_hardcut as hc
        ev = hc.events(run / "dubbed_24k.wav"); r["hard_cuts"] = sum(1 for e1, e2 in zip(ev, ev[1:]) if e1[0] == "OFF" and e2[0] == "ON" and e2[1] - e1[1] <= 0.6)
    if run is not None and manifest and (run / "dubbed_24k.wav").exists():
        import r3_joins as rj
        mp = run / pathlib.Path(manifest).name
        if mp.exists():
            r["joins"] = {k: v for k, v in rj.analyse(mp).items() if k not in ("rows", "manifest")}
            import r4_pause_logic as pl   # r4 v2: pauses inside an English phrase (Whisper words + script punctuation)
            pl_ = pl.analyse(run, json.loads(mp.read_text())); r["pause_logic"] = {k: v for k, v in pl_.items() if k != "rows"}; r["pause_rows"] = pl_["rows"]
            import r4_unit_continuity as uc
            r["units"] = {k: v for k, v in uc.analyse(mp).items() if k != "rows"}
    if do_wer and manifest:
        r["asr"] = {k: v for k, v in sc.back_transcribe(mix_p, json.loads(pathlib.Path(manifest).read_text())).items()}
        import ne_guard   # named entities misheard in the final mix (entity list of this run, or --ne-from another run of the same source)
        units = json.loads(pathlib.Path(ne_from or manifest).read_text())["units"]
        if any(u.get("ne") for u in units): r["ne"] = ne_guard.misheard(units, r["asr"]["hyp"])
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mix", required=True); ap.add_argument("--src", required=True); ap.add_argument("--sep", required=True)
    ap.add_argument("--run", default=None); ap.add_argument("--manifest", default=None); ap.add_argument("--wer", action="store_true"); ap.add_argument("--json", default=None)
    ap.add_argument("--series", default=None, help="save the 50 ms frame series (npz) for r4_review.py"); ap.add_argument("--ne-from", default=None, help="take the named-entity list from this manifest (score an older version on the same entities)"); a = ap.parse_args()
    r = score(a.mix, a.src, pathlib.Path(a.sep), pathlib.Path(a.run) if a.run else None, a.manifest, a.wer, a.ne_from, a.series)
    s = json.dumps(r, indent=1, ensure_ascii=False)
    if a.json: pathlib.Path(a.json).write_text(s)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict) or k == "loudness"}))


if __name__ == "__main__":
    main()
