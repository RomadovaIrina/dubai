"""Background bed for the final dubbed track (2026-10-12, human feedback: "the background goes dead silent, creepy"; "the voice breaks off
and then there is nothing"). Before this module the muxed audio was the dubbed voice ONLY: every non-speech moment of the video (room tone,
ambience, music, steps, water ...) was digital silence, so each pause between dubbed phrases sounded like a drop-out.

    separate_background()  Hybrid Demucs (torchaudio HDEMUCS_HIGH_MUSDB_PLUS, no new dependency) splits the source audio into
                           vocals + accompaniment; the background bed = drums + bass + other (44.1 kHz stereo). ~1-2 s GPU per video minute.
    mix_background()       final = dubbed voice + background bed. The bed keeps its ORIGINAL balance against the voice (bed gain = dubbed voice
                           level - original vocal level, both measured inside their own speech), is ducked softly while the dub speaks
                           (masks residual source-voice leakage, keeps the dub in front) and is never gated, so pauses keep the real ambience.
                           Room-tone fill: under the source speech HDemucs removes the room noise together with the voice (02: bed -52 dB vs the
                           -45 dB noise floor of the pauses), so the bed is topped up there with the video's OWN room tone (steady noise-floor
                           frames from its non-speech parts, crossfaded) up to that floor.

Only the MUXED audio changes: LatentSync, the speech gate, the TTS QA and the dub-only metrics keep using the clean dubbed voice track.
"""
from __future__ import annotations

import pathlib
import subprocess
import time

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEMUCS_WEIGHTS = ROOT / "models" / "demucs" / "hdemucs_high_trained.pt"   # scripts/download_models.sh; else torchaudio downloads it to the torch hub cache
SEP_SR = 44100
SOURCES = ["drums", "bass", "other", "vocals"]


def ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", *args], check=True)


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9)) if x.size else -180.0


def _mask(intervals: list, n: int, sr: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in intervals:
        m[max(0, int(s * sr)):min(n, int(e * sr))] = True
    return m


def _level_in(y: np.ndarray, intervals: list, sr: int) -> float:
    """RMS level (dBFS) of a mono signal inside the given intervals (whole signal when they are empty)."""
    m = _mask(intervals, len(y), sr)
    return _db(y[m] if m.any() else y)


def load_demucs(device: str = "cuda"):
    import torch
    import torchaudio
    if DEMUCS_WEIGHTS.exists():
        model = torchaudio.models.hdemucs_high(sources=SOURCES)
        model.load_state_dict(torch.load(str(DEMUCS_WEIGHTS), map_location="cpu"))
    else:
        model = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model()
    return model.to(device).eval()


def _separate(model, mix, device: str, seg_s: float = 10.0, overlap: float = 0.1):
    """Chunked HDemucs inference with linear cross-fades (torchaudio tutorial recipe); returns (4, C, L) on the CPU."""
    import torch
    from torchaudio.transforms import Fade
    C, L = mix.shape; ref = mix.mean(0); mu, sd = ref.mean(), ref.std().clamp_min(1e-6); mix = (mix - mu) / sd
    chunk = int(SEP_SR * seg_s * (1 + overlap)); ov = int(overlap * SEP_SR); start, end = 0, chunk
    fade = Fade(fade_in_len=0, fade_out_len=ov, fade_shape="linear"); out = torch.zeros(len(SOURCES), C, L)
    while start < L - ov:
        with torch.no_grad():
            o = model(mix[:, start:end].to(device)[None])[0].float().cpu()
        out[:, :, start:end] += fade(o)
        if start == 0:
            fade.fade_in_len = ov; start += chunk - ov
        else:
            start += chunk
        end += chunk
        if end >= L:
            fade.fade_out_len = 0
    return out * sd + mu


def separate_background(src: pathlib.Path, work: pathlib.Path, device: str = "cuda") -> dict:
    """Source audio -> bg_44k.wav (drums+bass+other, stereo) and vocals_44k.wav (stereo). Returns paths + timings + levels."""
    import torch
    t0 = time.perf_counter()
    wav = work / "source_44k.wav"; ff(["-i", str(src), "-vn", "-ac", "2", "-ar", str(SEP_SR), "-c:a", "pcm_f32le", str(wav)])
    y, sr = sf.read(str(wav), dtype="float32"); assert sr == SEP_SR
    t1 = time.perf_counter(); model = load_demucs(device); t_load = time.perf_counter() - t1
    t2 = time.perf_counter(); s = _separate(model, torch.from_numpy(np.ascontiguousarray(y.T)), device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_sep = time.perf_counter() - t2
    bg = (s[0] + s[1] + s[2]).numpy().T.astype(np.float32); voc = s[3].numpy().T.astype(np.float32)
    del model, s
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    p_bg, p_voc = work / "bg_44k.wav", work / "vocals_44k.wav"
    sf.write(str(p_bg), bg, SEP_SR, subtype="FLOAT"); sf.write(str(p_voc), voc, SEP_SR, subtype="FLOAT"); wav.unlink()
    return {"model": "torchaudio HDEMUCS_HIGH_MUSDB_PLUS (bg = drums+bass+other)", "weights": str(DEMUCS_WEIGHTS) if DEMUCS_WEIGHTS.exists() else "torch hub cache",
            "bg": str(p_bg), "vocals": str(p_voc), "seconds": round(len(y) / SEP_SR, 3), "load_s": round(t_load, 3), "separate_s": round(t_sep, 3),
            "total_s": round(time.perf_counter() - t0, 3), "source_db": round(_db(y.mean(1)), 2), "vocals_db": round(_db(voc.mean(1)), 2), "bg_db": round(_db(bg.mean(1)), 2)}


def _envelope(mask: np.ndarray, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    """0..1 ducking envelope from a boolean speech mask: linear ramps of attack_s before each speech onset and release_s after each offset."""
    env = mask.astype(np.float32)
    if not mask.any():
        return env
    a, r = max(1, int(attack_s * sr)), max(1, int(release_s * sr))
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])); on = np.flatnonzero(d == 1); off = np.flatnonzero(d == -1)
    for s in on:      # rise towards the onset (the duck is fully in place when the dub starts)
        lo = max(0, s - a); env[lo:s] = np.maximum(env[lo:s], np.linspace(0.0, 1.0, s - lo, endpoint=False, dtype=np.float32))
    for e in off:     # slow release after the offset
        hi = min(len(env), e + r); env[e:hi] = np.maximum(env[e:hi], np.linspace(1.0, 0.0, hi - e, endpoint=False, dtype=np.float32))
    return env


def _frame_rms(y: np.ndarray, h: int) -> np.ndarray:
    n = len(y) // h
    return np.sqrt(np.mean(y[: n * h].reshape(n, h) ** 2, axis=1) + 1e-12) if n else np.zeros(0)


def room_tone(bed: np.ndarray, source_speech: list, sr: int, n_out: int, frame_s: float = 0.2, margin_s: float = 0.3, xfade_s: float = 0.03, seed: int = 0,
              max_floor_db: float = -42.0):
    """Room tone of THIS video: the source audio (bed + vocals stem) in the frames outside source speech (+margin) whose level is the steady
    noise floor (30th percentile of those frames = the floor; frames <= floor + 3 dB: no door slams / steps / breaths), shuffled and joined with
    equal-power crossfades to n_out samples. Returns (tone[n_out, C], floor_rms); (None, None) with < 1 s of material, (None, floor) when the
    floor is louder than max_floor_db (music / loud ambience, not a room tone)."""
    h = int(frame_s * sr); mono = bed.mean(1); r = _frame_rms(mono, h); nf = len(r)
    speech = np.zeros(nf, dtype=bool)
    for s, e in source_speech:
        speech[max(0, int((s - margin_s) / frame_s)):min(nf, int(np.ceil((e + margin_s) / frame_s)))] = True
    cand = np.flatnonzero(~speech & (r > 1e-6))
    if len(cand) * frame_s < 1.0:
        return None, None
    floor = float(np.percentile(r[cand], 30)); keep = cand[r[cand] <= floor * 10 ** (3 / 20)]
    if 20 * np.log10(floor) > max_floor_db:   # not a room tone but music / loud ambience: shuffled frames would garble it, and HDemucs keeps such a bed anyway
        return None, floor
    if len(keep) * frame_s < 1.0:
        return None, None
    rng = np.random.default_rng(seed); x = int(xfade_s * sr); fade = np.sqrt(np.linspace(0.0, 1.0, x, dtype=np.float32))[:, None]
    out = np.zeros((n_out + h, bed.shape[1]), dtype=np.float32); pos = 0
    while pos < n_out:
        k = int(rng.choice(keep)); seg = bed[k * h:(k + 1) * h + x].copy()
        if len(seg) < h + x:
            continue
        if pos:
            seg[:x] *= fade; out[pos - x:pos] *= fade[::-1]; out[pos - x:pos] += seg[:x]; out[pos:pos + h] = seg[x:x + h]
        else:
            out[:h] = seg[:h]
        pos += h
    return out[:n_out], floor


def mix_background(sep: dict, dubbed24k: pathlib.Path, source_speech: list, dub_speech: list, duration: float, work: pathlib.Path,
                   bg_offset_db: float = 0.0, duck_db: float = -6.0, max_bg_gain_db: float = 12.0, attack_s: float = 0.08, release_s: float = 0.35,
                   room_fill: bool = True, room_floor_db: float = -42.0) -> dict:
    """final_mix_44k.wav (stereo) + final_mix_aac.m4a = dub voice (unchanged level, both channels) + background bed.
    Bed gain = (dub voice level in dub speech) - (original vocal level in source speech) + bg_offset_db, clamped to +-max_bg_gain_db, i.e. the
    original voice/background balance is reproduced; the bed is ducked by duck_db while the dub speaks (soft attack / release).
    room_floor_db (2026-10-14): the room-tone fill is used for noise floors up to this level. -42 (previous) skipped 03 (-40 dBFS room) and 04
    (-36 dBFS outdoor ambience) although HDemucs takes 5-6 dB of that ambience out under the source speech; with the thought pause policy the
    dub is silent there, and with duck_db = 0 nothing masks it: "улица задыхается, пока он говорит". Only the missing energy is added, so a bed
    already at the floor (05 street, 01 music) gets ~nothing."""
    t0 = time.perf_counter()
    d44 = work / "dubbed_44k.wav"; ff(["-i", str(dubbed24k), "-ar", str(SEP_SR), "-ac", "1", "-c:a", "pcm_f32le", str(d44)])
    dub, _ = sf.read(str(d44), dtype="float32"); d44.unlink()
    bg, _ = sf.read(sep["bg"], dtype="float32"); voc, _ = sf.read(sep["vocals"], dtype="float32")
    n = int(round(duration * SEP_SR)); dub = np.pad(dub, (0, max(0, n - len(dub))))[:n]
    bg = np.pad(bg, ((0, max(0, n - len(bg))), (0, 0)))[:n]; voc = np.pad(voc, ((0, max(0, n - len(voc))), (0, 0)))[:n]
    fill = {"enabled": room_fill, "applied": False}
    if room_fill:   # under the source speech HDemucs removes the room noise with the voice: top the bed up to the video's own noise floor there
        tone, floor = room_tone(bg + voc, source_speech, SEP_SR, n, max_floor_db=room_floor_db)   # the full source: HDemucs files the steady noise floor under "vocals" too
        if tone is None:
            fill["skipped"] = "no steady non-speech material" if floor is None else f"floor {20 * np.log10(floor + 1e-12):.1f} dBFS > {room_floor_db} (music / loud ambience)"
        else:
            h = int(0.05 * SEP_SR); r = _frame_rms(bg.mean(1), h); r = np.convolve(r, np.ones(4) / 4, mode="same")   # 200 ms smoothed bed level
            w = np.sqrt(np.clip(floor ** 2 - r ** 2, 0.0, None)) / floor                                            # missing energy, 0..1
            w = np.interp(np.arange(n), (np.arange(len(r)) + 0.5) * h, w).astype(np.float32)
            bg = bg + tone * w[:, None]
            fill = {"enabled": True, "applied": True, "floor_db": round(20 * np.log10(floor + 1e-12), 2), "mean_fill": round(float(w.mean()), 3),
                    "seconds_filled_gt_50pct": round(float((w > 0.5).sum() / SEP_SR), 2)}
    voice_db = _level_in(dub, dub_speech, SEP_SR); orig_voice_db = _level_in(voc.mean(1), source_speech, SEP_SR)
    gain_db = float(np.clip(voice_db - orig_voice_db, -max_bg_gain_db, max_bg_gain_db)) + bg_offset_db
    env = _envelope(_mask(dub_speech, n, SEP_SR), SEP_SR, attack_s, release_s)
    g = (10 ** (gain_db / 20)) * (10 ** (duck_db * env / 20))
    bed = bg * g[:, None]; mix = bed + dub[:, None]
    peak = float(np.abs(mix).max()) if n else 0.0; norm = 0.98 / peak if peak > 0.98 else 1.0; mix *= norm
    out = work / "final_mix_44k.wav"; sf.write(str(out), mix, SEP_SR, subtype="PCM_16")
    aac = work / "final_mix_aac.m4a"; ff(["-i", str(out), "-c:a", "aac", "-b:a", "192k", str(aac)])
    pause = ~_mask(dub_speech, n, SEP_SR)
    return {"wav": str(out), "aac": str(aac), "room_fill": fill, "bg_gain_db": round(gain_db, 2), "duck_db": duck_db, "attack_s": attack_s, "release_s": release_s, "bg_offset_db": bg_offset_db,
            "dub_voice_db": round(voice_db, 2), "orig_vocals_db": round(orig_voice_db, 2), "peak_norm": round(norm, 4),
            "bed_db_in_dub_pauses": round(_db(bed.mean(1)[pause]), 2) if pause.any() else None,
            "mix_db_in_dub_pauses": round(_db(mix.mean(1)[pause]), 2) if pause.any() else None,
            "seconds": round(n / SEP_SR, 3), "mix_s": round(time.perf_counter() - t0, 3)}
