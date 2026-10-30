#!/usr/bin/env python3
"""Round 3 D: remix the final mix of an audio run WITHOUT re-running TTS (user: "the background breathes / pumps").
The dub voice and its timing are untouched (lip renders stay valid); only the bed mix and the loudness stage change:
  * duck -2 dB, attack 0.4 s, release 0.8 s (was -4 dB / 0.25 / 0.6); residue duck -6 dB (was -15);
  * loudness: ONE static gain to the target (-16 LUFS) + a true-peak guard (limiter on a 4x oversampled signal at --tp), never ffmpeg loudnorm's
    dynamic mode (on 02 / 03 / 05 it acted as an AGC: bed gain jitter p95 7.8 dB on 03 vs 1.8 dB with the static gain).
Writes <out>/<id>/ = symlinks of the run's work dir + new final_mix_44k.wav / final_mix_aac.m4a + <id>_r4b.manifest.json, and checks it.
    python scripts/quality/r3_remix.py --src /workspace/dub/artifacts/r3d/audio-r4 --out /workspace/dub/artifacts/r3d/audio-r4b 03 04 ..."""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys
import soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "r7"))


def measure(f):
    e = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(f), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    t = e[e.rfind("Summary:"):]; I = re.search(r"I:\s+(-?[\d.]+) LUFS", t); P = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", t)
    return {"lufs": float(I.group(1)) if I else None, "true_peak_dbtp": float(P.group(1)) if P and P.group(1) != "-inf" else None}


def static_loudness(wav: pathlib.Path, aac: pathlib.Path, target: float, tp: float) -> dict:
    before = measure(wav); tmp = wav.with_name("_static.wav"); gain = target - before["lufs"]; lim = 10 ** (tp / 20); hist = []
    for _ in range(3):   # the limiter can take a little loudness off: correct the static gain (still ONE constant gain per pass)
        af = f"volume={gain:.3f}dB,aresample=176400,alimiter=limit={lim:.5f}:attack=1:release=60:level=disabled:latency=1,aresample=44100"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-af", af, "-c:a", "pcm_f32le", str(tmp)], check=True)
        m = measure(tmp); hist.append({"gain_db": round(gain, 3), **m})
        if m["lufs"] is None or abs(m["lufs"] - target) <= 0.3: break
        gain += target - m["lufs"]
    tmp.replace(wav)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
    for _ in range(3):   # r4: the AAC encode can overshoot the limiter (v4repro 03: -0.9 dBTP) -> lower the whole wav by the overshoot and re-encode
        tp_aac = measure(aac)["true_peak_dbtp"]
        if tp_aac is None or tp_aac <= -1.1: break
        y, sr = sf.read(str(wav), dtype="float32"); sf.write(str(wav), y * 10 ** ((-1.2 - tp_aac) / 20), sr, subtype="FLOAT"); hist.append({"aac_tp_trim_db": round(-1.2 - tp_aac, 2)})
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
    return {"mode": "static gain + oversampled limiter", "target_lufs": target, "limiter_dbfs": tp, "before": before, "passes": hist, "after_wav": measure(wav), "after_aac": measure(aac)}


def unit_level(man: dict, src: pathlib.Path, out24: pathlib.Path, out16: pathlib.Path, amount: float = 0.7, cap_db: float = 4.0) -> dict:
    """r6 --unit-level on: sentence-to-sentence loudness of the dub voice (Chatterbox takes of different seeds / cfg differ by up to 6 dB; r6v2 03 u5
    -20.6 dB next to u4 -14.8 dB). Each unit's speech level (RMS of its frames within 30 dB of the loudest) moves `amount` of the way to the video's
    median unit level, capped at +-cap_db; the gain changes only in the silence between units (20 ms ramps). Writes the levelled 24 kHz / 16 kHz dub."""
    import numpy as np, soundfile as sf
    y, sr = sf.read(str(src), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); h = int(0.02 * sr); spans = []
    for u in man["units"]:
        pl = [x["placed"] for x in (u.get("alignment") or {}).get("placements", []) if x.get("placed")]
        if pl: spans.append((u["id"], min(a for a, _ in pl), max(b for _, b in pl)))
    lv = {}
    for uid, a, b in spans:
        seg = y[int(a * sr):int(b * sr)]; n = len(seg) // h
        if n < 5: continue
        f = 20 * np.log10(np.sqrt((seg[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9); act = f[f > f.max() - 30]
        lv[uid] = float(10 * np.log10(np.mean(10 ** (act / 10))))
    if not lv: return {"applied": False}
    med = float(np.median(list(lv.values()))); g = np.ones(len(y), dtype=np.float32); rows = []
    for uid, a, b in spans:
        if uid not in lv: continue
        gdb = float(np.clip(amount * (med - lv[uid]), -cap_db, cap_db)); rows.append({"unit": uid, "level_db": round(lv[uid], 2), "gain_db": round(gdb, 2)})
        i0, i1 = max(0, int((a - 0.03) * sr)), min(len(y), int((b + 0.05) * sr)); gl = 10 ** (gdb / 20); r = min(int(0.02 * sr), max(1, (i1 - i0) // 4))
        env = np.full(i1 - i0, gl, dtype=np.float32); env[:r] = np.linspace(1, gl, r); env[-r:] = np.linspace(gl, 1, r); g[i0:i1] = env
    z = y * g; pk = float(np.max(np.abs(z))) if len(z) else 0.0
    if pk > 0.98: z *= 0.98 / pk
    sf.write(str(out24), z, sr, subtype="PCM_16")
    import librosa
    sf.write(str(out16), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    return {"applied": True, "median_db": round(med, 2), "amount": amount, "cap_db": cap_db, "rows": rows}


def cap_gap_bed(bg: pathlib.Path, out: pathlib.Path, speech: list, dub_speech: list, over_db: float = 10.0, scope: str = "gaps") -> dict:
    """r6 --gap-bed-cap-db: where the SOURCE speaks and the DUB is silent (sentence gaps, the speaker's own pauses kept by r6), the separated bed can
    carry leaked syllables / clicks of the original voice (r6v7 02 12.1 s: a 100 ms burst 39 dB above the bed's median, +2 dB over the dub voice, in
    a 0.3 s sentence gap). The bed's 10 ms level there is limited to its median over those frames + over_db (smooth gain: 3-frame min + 5-frame mean);
    everything under the dub and in true pauses is untouched."""
    import numpy as np, soundfile as sf
    y, sr = sf.read(str(bg), dtype="float32"); mono = y.mean(1) if y.ndim > 1 else y; h = int(0.01 * sr); n = len(mono) // h
    L = 20 * np.log10(np.sqrt((mono[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9); t = (np.arange(n) + 0.5) * 0.01
    def inside(ivs, pad=0.0):
        m = np.zeros(n, bool)
        for a, b in ivs: m[max(0, int((a - pad) / 0.01)):min(n, int((b + pad) / 0.01) + 1)] = True
        return m
    # scope "speech" (r6 v2): every source-speech frame (r6_audio_v1 02 12.1 s: the leaked syllable overlaps the dub's tail at a 0.3 s sentence gap,
    # so the gap mask missed it); the cap is relative to the bed's median over those frames, so only bursts > over_db above it are touched
    # scope "edges": the gap mask widened by 150 ms into the dub (a leak at a dub edge), long under-dub stretches untouched (scope "speech" reshaped
    # the bed on 05: pause-onset jump 4.7 -> 6.3 dB)
    if scope == "edges":
        nd = ~inside(dub_speech); M = inside(speech) & (np.convolve(nd.astype(float), np.ones(31), mode="same") > 0)
    else:
        M = inside(speech) & (~inside(dub_speech, 0.05) if scope == "gaps" else True)
    if M.sum() < 20: return {"applied": False}
    ref = float(np.median(L[M])); cap = ref + over_db; g = np.zeros(n)
    g[M] = np.minimum(0.0, cap - L[M])
    g = np.minimum.reduce([np.roll(g, k) for k in (-1, 0, 1)]); g = np.convolve(g, np.ones(5) / 5, mode="same")
    gs = 10 ** (np.interp(np.arange(len(mono)) / sr, t, g) / 20).astype(np.float32)
    z = y * (gs[:, None] if y.ndim > 1 else gs); sf.write(str(out), z, sr, subtype="PCM_16")
    return {"applied": True, "ref_db": round(ref, 1), "cap_db": round(cap, 1), "frames_capped": int((g < -0.5).sum()), "max_cut_db": round(float(g.min()), 1)}


def dub_gate(src24: pathlib.Path, out24: pathlib.Path, out16: pathlib.Path, below_db: float = 20.0, min_s: float = 0.6, ramp_s: float = 0.02, floor_db: float = -30.0,
             voiced: float = 0.35, deep_db: float = 30.0) -> dict:
    """r7 v2: the dub voice's noisy pauses (Chatterbox: unvoiced crackly noise at -33..-45 dBFS, 04 3.9-5.7 s, which r7_audio_v1's gap denoise
    exposed by lowering the bed under it) -> floor_db. 10-ms frames > below_db under the voice's p95 AND unvoiced (normalised autocorrelation
    peak at 70-400 Hz in a 40-ms window < voiced), or > deep_db under it, in runs >= min_s (a quiet voiced word stays; a fricative, and a
    0.25-0.5 s breath inside a phrase (muting it opened an audible mid-phrase hole: r4 pauses-inside-phrase 0 -> 4 on 04), are shorter);
    the gain moves with sin^2 (equal-power) ramp_s ramps INSIDE the run, so word edges keep full level."""
    import numpy as np
    from scipy.signal import resample_poly
    y, sr = sf.read(str(src24), dtype="float32"); mono = y if y.ndim == 1 else y.mean(1); h = sr // 100; n = len(mono) // h
    L = 10 * np.log10((mono[:n * h].reshape(n, h) ** 2).mean(1) + 1e-12); ref = float(np.percentile(L[L > L.max() - 50], 95))
    W = 4 * h; pad = np.concatenate([np.zeros(W), mono, np.zeros(W)]); idx = np.arange(n) * h + W - (W - h) // 2
    fr = np.stack([pad[i:i + W] for i in idx]); fr = fr - fr.mean(1, keepdims=True)
    F = np.fft.rfft(fr, 2 * W, axis=1); ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :W]; ac = ac / (ac[:, :1] + 1e-12)
    v = ac[:, int(sr / 400):int(sr / 70)].max(1)
    low = (((L < ref - below_db) & (v < voiced)) | (L < ref - deep_db)) & (L >= -90)
    raw = []; i = 0
    while i < n:
        if not low[i]: i += 1; continue
        j = i
        while j < n and low[j]: j += 1
        raw.append([i, j]); i = j
    mer = []   # merge runs split by a < 100 ms quiet hole first (one noisy pause = one run; the hole would pop up as a blip), then the length rule
    for a0, b0 in raw:
        if mer and a0 - mer[-1][1] < 10 and (L[mer[-1][1]:a0] < ref - below_db).all(): mer[-1][1] = b0
        else: mer.append([a0, b0])
    g = np.ones(n); runs = []
    for i, j in mer:
        dig = (i > 0 and L[i - 1] < -90) or (j < n and L[j] < -90)   # a word's decay into the take's digital silence (or the rise out of it) stays
        if (j - i) / 100 >= min_s and not dig: g[i:j] = 0.0; runs.append([round(i / 100, 2), round(j / 100, 2)])
    gs = np.repeat(g, h); gs = np.concatenate([gs, np.full(len(mono) - len(gs), gs[-1] if len(gs) else 1.0)])
    k = int(ramp_s * sr)
    # erode the gated region by one ramp on each side, then a raised-cosine (sin^2) transition entirely inside the original run
    gm = (gs < 0.5).astype(float); er = np.convolve(gm, np.ones(k), "same") >= k - 0.5   # eroded mask
    lin = np.clip(np.convolve(er.astype(float), np.ones(k) / k, "same"), 0, 1)       # 0..1 over ramp_s, inside the run
    fl = 10 ** (floor_db / 20); gain = 1 - (1 - fl) * np.sin(np.pi / 2 * lin) ** 2
    z = (y * (gain if y.ndim == 1 else gain[:, None])).astype("float32")
    sf.write(str(out24), z, sr, subtype="FLOAT"); sf.write(str(out16), resample_poly(z if z.ndim == 1 else z.mean(1), 2, 3).astype("float32") if sr == 24000 else z, 16000, subtype="PCM_16")
    return {"below_db": below_db, "min_s": min_s, "ramp_s": ramp_s, "floor_db": floor_db, "voiced": voiced, "deep_db": deep_db, "ref_db": round(ref, 1), "gated_s": round(float((gain < 0.5).sum() / sr), 2), "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("ids", nargs="+")
    ap.add_argument("--duck-db", type=float, default=-2.0); ap.add_argument("--attack", type=float, default=0.4); ap.add_argument("--release", type=float, default=0.8)
    ap.add_argument("--residue-db", type=float, default=-6.0); ap.add_argument("--src-duck-db", type=float, default=0.0); ap.add_argument("--target", type=float, default=-16.0); ap.add_argument("--tp", type=float, default=-1.5)
    ap.add_argument("--unit-level", choices=["on", "off"], default="off", help="r6: sentence-to-sentence loudness of the dub voice (unit_level())")
    ap.add_argument("--gap-bed-cap-scope", choices=["gaps", "edges", "speech"], default="gaps", help="r6: frames the bed cap applies to (cap_gap_bed())")
    ap.add_argument("--gap-bed-cap-db", type=float, default=0.0, help="r6: > 0 = bed level cap (median + this) where the source speaks and the dub is silent (cap_gap_bed())")
    ap.add_argument("--gap-clean", choices=["on", "off"], default="off", help="r7: residual-voice / burst detector + room tone + denoise of the bed in dub gaps (scripts/r7/gap_clean.py)")
    ap.add_argument("--dub-gate", choices=["on", "off"], default="off", help="r7 v2: dub voice residue in its own pauses (unvoiced runs >= --dub-gate-min s, > --dub-gate-db under the voice p95) -> -30 dB, sin^2 20-ms ramps (dub_gate())")
    ap.add_argument("--dub-gate-db", type=float, default=20.0); ap.add_argument("--dub-gate-min", type=float, default=0.6)
    ap.add_argument("--gap-clean-voice-p", type=float, default=0.4); ap.add_argument("--gap-clean-burst-db", type=float, default=6.0); ap.add_argument("--gap-clean-nr", type=float, default=0.6); ap.add_argument("--gap-clean-nr-ns", type=float, default=0.0)
    ap.add_argument("--gap-clean-floor", type=float, default=0.0, help="r8 v4: > 0 = a cleaned dub gap never sits more than this many dB under the local ambience (filled with the source's own ambience, >= 120-ms ramps)")
    ap.add_argument("--sep", choices=["spleeter", "hdemucs", "htdemucs", "htdemucs_ft"], default="spleeter", help="r8-sep: separator of the bed. spleeter = the run's own "
                    "bg_44k.wav (MIT, unchanged); the others are NC (MUSDB18-HQ weights, user-accepted deviation, DUB_ALLOW_NC_BED=1): stems from the cache "
                    "$DUB_SEP_STEMS/stems_<sep>[_res]/<id>/ (scripts/r8/sep_stems.py; separated here when missing)")
    ap.add_argument("--sep-bg", choices=["stems", "residual"], default="stems", help="r8-sep, --sep != spleeter: bed = sum of the non-vocal stems | source - vocals stem")
    ap.add_argument("--room-fill", choices=["on", "off"], default="on", help="r8-sep: mix_background's room-tone top-up under the source speech (an HDemucs workaround)")
    ap.add_argument("--bed-pauses", choices=["sep", "source"], default="sep", help="r8-sep: source = in the source pauses (speech_intervals +-0.3 s off) the bed IS the "
                    "source (bed + vocals stems), separated bed only under / around the speech, 60-ms equal-power joins (htdemucs files part of 02's room noise "
                    "under vocals: its pause bed was 7 dB low)")
    ap.add_argument("--bed-noise-back", choices=["on", "off"], default="off", help="r8-sep: the stationary noise of the vocals stem (noisereduce, profile = the "
                    "vocals stem in the source pauses) moved back to the bed: htdemucs takes 02's room noise out WITH the voice (bed under speech -65 vs -52 dB)")
    a = ap.parse_args()
    import background_bed as bb, run_clean_pipeline_05 as rp, r3_bedswing  # noqa: F401
    stems_root = pathlib.Path(os.environ.get("DUB_SEP_STEMS", "/workspace/dub/r8a"))
    for id_ in a.ids:
        W = (pathlib.Path(a.src) / id_).resolve(); O = pathlib.Path(a.out) / id_; O.mkdir(parents=True, exist_ok=True)
        for f in W.iterdir():
            if f.name.startswith("final_mix") or f.name.endswith(".manifest.json"): continue
            (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f)
        man = json.loads(next(W.glob(f"{id_}_*.manifest.json")).read_text())
        sep = {"bg": str(W / "bg_44k.wav"), "vocals": str(W / "vocals_44k.wav")}
        if a.sep != "spleeter":   # r8-sep: the bed (and the vocals stem the gains / room tone read) of another separator; the dub, timing and flags unchanged
            C = stems_root / f"stems_{a.sep}{'_res' if a.sep_bg == 'residual' else ''}" / id_
            if not (C / "bg_44k.wav").exists():
                os.environ["DUB_SEP_BG"] = a.sep_bg; bb.SEP_BG = a.sep_bg; C.mkdir(parents=True, exist_ok=True)
                src = next(pathlib.Path(os.environ.get("DUB_INPUTS", "/workspace/dub/inputs/test_videos_raw")).glob(f"{id_}.*"))
                (C / "sep.json").write_text(json.dumps(bb.separate_background(src, C, separator=a.sep), indent=1))
            sep = {"bg": str(C / "bg_44k.wav"), "vocals": str(C / "vocals_44k.wav")}
            for f in ("bg_44k.wav", "vocals_44k.wav"): (O / f).unlink(missing_ok=True); (O / f).symlink_to(C / f)   # mixdrop / metrics read the run dir's raw bed
            man["bed_separator"] = {"sep": a.sep, "bg": a.sep_bg, "stems": str(C), "licence": "NC weights (MUSDB18-HQ), user-accepted deviation 29.10"}
        D = W
        if a.unit_level == "on":   # the levelled dub replaces the symlinked one in the remix dir (metrics, handoff and the mix all read it)
            for f in ("dubbed_24k.wav", "dubbed_16k.wav"): (O / f).unlink(missing_ok=True)
            man["unit_level"] = unit_level(man, W / "dubbed_24k.wav", O / "dubbed_24k.wav", O / "dubbed_16k.wav"); D = O
        if a.dub_gate == "on":   # before the dub VAD: the gated dub is what the mix, the metrics and the handoff read
            g24, g16 = O / "dubbed_24k.wav", O / "dubbed_16k.wav"; src24 = (D / "dubbed_24k.wav").resolve()
            if D == O: src24 = src24.rename(O / "dubbed_24k_ungated.wav")   # unit-levelled dub already in O: keep it as the gate's input
            g24.unlink(missing_ok=True); g16.unlink(missing_ok=True)
            man["dub_gate"] = dub_gate(src24, g24, g16, a.dub_gate_db, a.dub_gate_min); D = O
            print(f"   {id_} dub_gate: {man['dub_gate']['gated_s']} s gated in {len(man['dub_gate']['runs'])} runs", flush=True)
        speech = [(s["start"], s["end"]) for s in man["speech_intervals"]]; dub_speech = [(s["start"], s["end"]) for s in rp.run_vad(D / "dubbed_16k.wav")]
        if a.bed_noise_back == "on":   # r8-sep: the vocals stem's stationary noise (profile = that stem in the source pauses) goes back to the bed
            import numpy as np, noisereduce as nr
            bgy, bsr = sf.read(sep["bg"], dtype="float32"); vy, _ = sf.read(sep["vocals"], dtype="float32"); m_ = min(len(bgy), len(vy)); bgy, vy = bgy[:m_], vy[:m_]
            Pm = ~bb._mask([(s0 - 0.3, e0 + 0.3) for s0, e0 in speech], m_, bsr); prof = vy[Pm][: 30 * bsr]
            if len(prof) >= bsr:
                vn = np.stack([vy[:, c] - nr.reduce_noise(y=vy[:, c], sr=bsr, stationary=True, y_noise=prof[:, c], prop_decrease=1.0, n_jobs=1)[:m_] for c in range(vy.shape[1])], 1)
                # only the STATIONARY part goes back: 10-ms frames above the component's own pause level (median) + 3 dB are scaled down to it (the
                # gate's mask lags at voice onsets: 02 69.89 s a consonant onset passed into the bed), 5-ms smoothed gains
                h = int(0.01 * bsr); nf_ = len(vn) // h; r_ = np.sqrt((vn[:nf_ * h].mean(1).reshape(nf_, h) ** 2).mean(1)) + 1e-9
                lim = np.median(r_[Pm[:nf_ * h:h]]) * 10 ** (3 / 20) if Pm[:nf_ * h:h].any() else np.median(r_)
                from scipy.ndimage import minimum_filter1d
                gn = np.convolve(minimum_filter1d(np.minimum(1.0, lim / r_), 3), np.ones(3) / 3, "same")   # min over +-10 ms first: no onset escapes the ramp
                vn = vn * np.interp(np.arange(m_), (np.arange(nf_) + 0.5) * h, gn).astype(np.float32)[:, None]
                sf.write(str(O / "bg_nb_44k.wav"), (bgy + vn).astype(np.float32), bsr, subtype="FLOAT"); sf.write(str(O / "vocals_nb_44k.wav"), (vy - vn).astype(np.float32), bsr, subtype="FLOAT")
                sep = {**sep, "bg": str(O / "bg_nb_44k.wav"), "vocals": str(O / "vocals_nb_44k.wav")}
                man["bed_noise_back"] = {"noise_db": round(float(bb._db(vn.mean(1))), 2), "vocals_db": round(float(bb._db(vy.mean(1))), 2), "profile_s": round(len(prof) / bsr, 1)}
        if a.bed_pauses == "source":   # r8-sep: true pauses keep the original sound itself (nothing to separate there: no voice)
            import numpy as np
            bgy, bsr = sf.read(sep["bg"], dtype="float32"); vy, _ = sf.read(sep["vocals"], dtype="float32"); m_ = min(len(bgy), len(vy)); bgy, vy = bgy[:m_], vy[:m_]
            # weight 1 only 0.3 s clear of the speech, 150-ms linear ramps INSIDE the pause (bed and source are correlated: linear); 60 ms left a step
            # on 02 (separated bed 30 dB under the source room) that the click detector caught at 28.47 s
            w = (~bb._mask([(s0 - 0.3 - 0.075, e0 + 0.3 + 0.075) for s0, e0 in speech], m_, bsr)).astype(np.float32); k = int(0.15 * bsr)
            w = np.clip(np.convolve(w, np.ones(k, np.float32) / k, "same"), 0, 1)[:, None]
            # bed + w * vocals = the source in pauses; the vocals stem keeps the rest, so bed + vocals stays the source for room tone / gains
            sf.write(str(O / "bg_hyb_44k.wav"), (bgy + w * vy).astype(np.float32), bsr, subtype="FLOAT"); sf.write(str(O / "vocals_hyb_44k.wav"), ((1 - w) * vy).astype(np.float32), bsr, subtype="FLOAT")
            sep = {**sep, "bg": str(O / "bg_hyb_44k.wav"), "vocals": str(O / "vocals_hyb_44k.wav")}
            man["bed_pauses"] = {"mode": "source", "pause_s": round(float((w[:, 0] > 0.5).sum() / bsr), 2)}
        if a.gap_bed_cap_db > 0:
            man["gap_bed_cap"] = cap_gap_bed(pathlib.Path(sep["bg"]), O / "bg_capped_44k.wav", speech, dub_speech, a.gap_bed_cap_db, a.gap_bed_cap_scope)
            if man["gap_bed_cap"].get("applied"): sep = {**sep, "bg": str(O / "bg_capped_44k.wav")}
        if a.gap_clean == "on":   # r7: after the r6 cap, before the mix (room fill / ducks / gains unchanged)
            import gap_clean
            bgy, bsr = sf.read(sep["bg"], dtype="float32"); vy, _ = sf.read(sep["vocals"], dtype="float32"); m_ = min(len(bgy), len(vy))
            z, man["gap_clean"] = gap_clean.clean(bgy, bsr, speech, dub_speech, a.gap_clean_voice_p, a.gap_clean_burst_db, a.gap_clean_nr, full=bgy[:m_] + vy[:m_], nr_ns=a.gap_clean_nr_ns, floor_db=a.gap_clean_floor)
            sf.write(str(O / "bg_clean_44k.wav"), z, bsr, subtype="FLOAT"); sep = {**sep, "bg": str(O / "bg_clean_44k.wav")}
            print(f"   {id_} gap_clean: { {k: v for k, v in man['gap_clean'].items() if k != 'runs'} }", flush=True)
        mix = bb.mix_background(sep, D / "dubbed_24k.wav", speech, dub_speech, man["source"]["duration_s"], O, duck_db=a.duck_db, attack_s=a.attack,
                                release_s=a.release, residue_duck_db=a.residue_db, src_duck_db=a.src_duck_db, room_fill=a.room_fill == "on")
        loud = static_loudness(O / "final_mix_44k.wav", O / "final_mix_aac.m4a", a.target, a.tp)
        man["background"] = {**man.get("background", {}), "mix": mix, "loudness": loud, "remix": "r3 D r3_remix.py (bed duck -2 dB / 0.4 / 0.8 s, residue -6 dB, static loudness)"}
        man["dubbed_track"] = {**man["dubbed_track"], "aac": str(O / "final_mix_aac.m4a")}
        (O / f"{id_}_r4b.manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
        sw = subprocess.run([sys.executable, str(HERE / "r3_bedswing.py"), "--mix", str(O / "final_mix_44k.wav"), "--dub", str(D / "dubbed_24k.wav"), "--bg", sep["bg"]],
                            capture_output=True, text=True).stdout.strip()
        print(f"== {id_} r4b: {loud['after_aac']} bed {sw}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
