#!/usr/bin/env python3
"""r7 --gap-clean: no junk in the bed while the dub is silent (user on r6 m5: "farts" = separation artefacts, and leaked fragments of the original voice
in dub silences). Runs on the separated bed BEFORE mix_background (which still does room fill, ducks and gains exactly as in r6). In every dub gap
(dub VAD silent, 60 ms into the dub edges):
  1. detector, 20 ms frames:
       voice = Silero on the bed stem itself >= voice_p (the separator left the original voice in the accompaniment) -- anywhere in a gap;
       burst = the bed > its local ambience + burst_db, where the SOURCE speaks (the only place a separator leaves "farts": the bed there is the
               accompaniment with the voice cut out). Local ambience = 30th percentile of the bed level in non-speech frames within +-3 s.
  2. detected frames (+-40 ms) are REPLACED by room tone of this video (background_bed.room_tone: the steady noise floor of the source's
     non-speech frames, shuffled with equal-power joins), level-matched to the local ambience; 30 ms crossfades.
  3. the rest of the gap where the source speaks is denoised (noisereduce, MIT, stationary, noise profile = that room tone, prop_decrease nr_prop):
     the musical noise of the separator goes, steady ambience stays (mix_background's room fill tops it up to the floor).
Under the dub and in true pauses away from detected voice the bed is untouched (r6_audio_v1 behaviour). Thresholds are relative to the video's
own bed: no per-video settings. Room tone = a floor <= -42 dBFS or a steady one (steadiness() <= 3 dB). Without it (music / variable outdoor sound) only leaked VOICE is treated, lowered to the local ambience."""
from __future__ import annotations
import numpy as np

FR = 0.02


def bed_voice_prob(bg: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Silero speech probability of the bed stem, 32 ms frames (same model as bed_speech_runs)."""
    import torch, scipy.signal as ss
    from silero_vad import load_silero_vad
    y16 = ss.resample_poly(bg.mean(1) if bg.ndim > 1 else bg, 160, 441).astype(np.float32); m = load_silero_vad(); w = 512; p = []
    with torch.no_grad():
        for i in range(0, len(y16) - w, w):
            p.append(float(m(torch.from_numpy(y16[i:i + w].copy()), 16000)))
    return np.array(p), w / 16000


def _mask_frames(ivs, n, pad=0.0):
    m = np.zeros(n, bool)
    for a, b in ivs: m[max(0, int((a - pad) / FR)):min(n, int(np.ceil((b + pad) / FR)))] = True
    return m


def _dilate(m, k):
    return np.convolve(m.astype(float), np.ones(2 * k + 1), "same") > 0 if k > 0 else m


def _ramp(m, sr, n_out, xf_s=0.03):
    """frame mask -> sample weight 0..1 with linear xf_s crossfades"""
    k = max(1, int(round(xf_s / FR))); w = np.convolve(m.astype(float), np.ones(2 * k + 1) / (2 * k + 1), "same")
    return np.clip(np.interp(np.arange(n_out) / sr, (np.arange(len(m)) + 0.5) * FR, w), 0, 1).astype(np.float32)


def steadiness(src, sr, source_speech, frame_s=0.2, margin_s=0.3):
    """p70 - p30 (dB) of the source level in non-speech 200 ms frames: <= 3 dB = a steady ambience (room / hum / even street noise) usable as a room
    tone even when it is loud (03 -40 dBFS 2.0 dB, 04 -36 dBFS 1.6 dB); music / variable outdoor sound is 7-9 dB (01, 05)"""
    import background_bed as bb
    h = int(frame_s * sr); r = 20 * np.log10(bb._frame_rms(src.mean(1) if src.ndim > 1 else src, h) + 1e-9); m = np.zeros(len(r), bool)
    for a, b in source_speech: m[max(0, int((a - margin_s) / frame_s)):int(np.ceil((b + margin_s) / frame_s))] = True
    c = r[~m]
    return round(float(np.percentile(c, 70) - np.percentile(c, 30)), 2) if len(c) >= 5 else None


def detect(bg, sr, source_speech, dub_speech, voice_p=0.4, burst_db=6.0, edge_s=0.06, bursts=True):
    import background_bed as bb
    mono = bg.mean(1) if bg.ndim > 1 else bg; h = int(FR * sr); n = len(mono) // h
    L = 20 * np.log10(bb._frame_rms(mono, h) + 1e-9)[:n]
    S = _mask_frames(source_speech, n); D = _mask_frames(dub_speech, n, edge_s); G = ~D
    amb = ~_mask_frames(source_speech, n, 0.3)
    ref = np.full(n, np.percentile(L[amb], 30) if amb.sum() > 50 else np.percentile(L, 20))
    k = int(3.0 / FR); ai = np.flatnonzero(amb)
    if len(ai) > 50:   # local ambience: 30th percentile of the non-speech bed within +-3 s (falls back to the video's value)
        cs = np.searchsorted
        for i in range(0, n, 25):
            sel = L[ai[cs(ai, i - k):cs(ai, i + k)]]
            if len(sel) >= 25: ref[i:i + 25] = np.percentile(sel, 30)
    p, ph = bed_voice_prob(bg, sr); pv = np.interp((np.arange(n) + 0.5) * FR, (np.arange(len(p)) + 0.5) * ph, p) if len(p) else np.zeros(n)
    voice = G & (pv >= voice_p); burst = G & S & (L > ref + burst_db) & bursts
    junk = _dilate(voice | burst, 2) & G
    return {"L": L, "ref": ref, "S": S, "G": G, "voice": voice, "burst": burst, "junk": junk, "n": n}


def clean(bg: np.ndarray, sr: int, source_speech: list, dub_speech: list, voice_p: float = 0.4, burst_db: float = 6.0, nr_prop: float = 0.6,
          xf_s: float = 0.03, seed: int = 0, full: np.ndarray | None = None, nr_ns: float = 0.0) -> tuple[np.ndarray, dict]:
    """full = bed + vocals stems (the whole source): the room tone is taken from it, as mix_background's room fill does"""
    import background_bed as bb, noisereduce as nr
    N = len(bg); st = bg if bg.ndim > 1 else bg[:, None]
    src = st if full is None else full[:N]; tone, floor = bb.room_tone(src, source_speech, sr, N, seed=seed, max_floor_db=0.0)
    spread = steadiness(src, sr, source_speech)
    if tone is not None and 20 * np.log10(floor + 1e-9) > -42.0 and (spread is None or spread > 3.0): tone = None   # loud AND unsteady: music / traffic, not a room tone
    # bursts only over a steady room tone: over music / loud ambience (no room tone) a loud frame is the music itself, only leaked voice is treated
    d = detect(bg, sr, source_speech, dub_speech, voice_p, burst_db, bursts=tone is not None); n = d["n"]
    rep = {"voice_s": round(float(d["voice"].sum() * FR), 2), "burst_s": round(float(d["burst"].sum() * FR), 2), "replaced_s": round(float(d["junk"].sum() * FR), 2),
           "gap_s": round(float(d["G"].sum() * FR), 2), "gap_src_speaks_s": round(float((d["G"] & d["S"]).sum() * FR), 2), "voice_p": voice_p, "burst_db": burst_db,
           "nr_prop": nr_prop, "room_tone": tone is not None,
           "floor_db": None if floor is None else round(20 * np.log10(floor + 1e-9), 1), "ambience_spread_db": spread}
    out = st.copy()
    gs = d["G"] & d["S"] & ~d["junk"]
    if nr_prop > 0 and gs.any() and tone is not None:   # 3. denoise the gap where the source speaks (separator musical noise), noise profile = room tone
        prof = tone[: min(N, 10 * sr)]
        den = np.stack([nr.reduce_noise(y=st[:, c], sr=sr, stationary=True, y_noise=prof[:, c], prop_decrease=nr_prop, n_jobs=1) for c in range(st.shape[1])], 1)
        wd = _ramp(gs, sr, N, xf_s)[:, None]; out = out * (1 - wd) + den[:N] * wd
        rep["denoised_s"] = round(float(gs.sum() * FR), 2)
    elif nr_ns > 0 and gs.any():   # no room tone (music / variable outdoor): non-stationary noisereduce (its own running noise estimate) in those gaps only
        den = np.stack([nr.reduce_noise(y=st[:, c], sr=sr, stationary=False, prop_decrease=nr_ns, n_jobs=1) for c in range(st.shape[1])], 1)
        wd = _ramp(gs, sr, N, xf_s)[:, None]; out = out * (1 - wd) + den[:N] * wd
        rep["denoised_nonstationary_s"] = round(float(gs.sum() * FR), 2); rep["nr_ns"] = nr_ns
    if d["junk"].any():   # 2. replace detected frames by room tone at the local ambience level
        wj = _ramp(d["junk"], sr, N, xf_s)[:, None]
        if tone is not None:
            gdb = np.clip(d["ref"] - 20 * np.log10(floor + 1e-9), -6, 6)
            g = 10 ** (np.interp(np.arange(N) / sr, (np.arange(n) + 0.5) * FR, gdb) / 20).astype(np.float32)[:, None]
            out = out * (1 - wj) + tone * g * wj
            rep["tone_floor_db"] = round(20 * np.log10(floor + 1e-9), 1)
        else:              # no room tone: lower the detected frames to the local ambience
            cut = np.minimum(0.0, d["ref"] - d["L"]); g = 10 ** (np.interp(np.arange(N) / sr, (np.arange(n) + 0.5) * FR, cut) / 20).astype(np.float32)[:, None]
            out = out * (1 - wj) + out * g * wj
    runs = []; m = d["junk"]; dd = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)):
        runs.append([round(a * FR, 2), round(b * FR, 2), "voice" if d["voice"][a:b].any() else "burst"])
    rep["runs"] = runs[:200]
    return (out if bg.ndim > 1 else out[:, 0]).astype(np.float32), rep
