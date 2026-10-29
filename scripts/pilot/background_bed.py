"""Background bed for the final dubbed track (2026-10-12, human feedback: "the background goes dead silent, creepy"; "the voice breaks off
and then there is nothing"). Before this module the muxed audio was the dubbed voice ONLY: every non-speech moment of the video (room tone,
ambience, music, steps, water ...) was digital silence, so each pause between dubbed phrases sounded like a drop-out.

    separate_background()  splits the source audio into vocals + accompaniment (44.1 kHz stereo); the accompaniment is the bed.
                           separator="spleeter" (default since 2026-10-17, Agent C r2): Deezer Spleeter 2stems, MIT code, weights = release
                           asset of the MIT repo (reports/quality/audio_timing/LICENSES.md), run in its own TensorFlow venv on the CPU
                           (scripts/setup_spleeter.sh; ~8 s per 90 s video). 03 A/B (bed_p1.md): no detectable voice residue in dub pauses,
                           ambience level continuous under the source speech (-0.1 dB vs HDEMUCS -3.0 dB).
                           separator="hdemucs": torchaudio HDEMUCS_HIGH_MUSDB_PLUS (bg = drums + bass + other), weights WITHOUT a licence:
                           non-commercial reference only, refused unless DUB_ALLOW_NC_BED=1.
    mix_background()       final = dubbed voice + background bed. The bed keeps its ORIGINAL balance against the voice (bed gain = dubbed voice
                           level - original vocal level, both measured inside their own speech), is ducked softly while the dub speaks
                           (masks residual source-voice leakage, keeps the dub in front) and is never gated, so pauses keep the real ambience.
                           Room-tone fill: under the source speech HDemucs removes the room noise together with the voice (02: bed -52 dB vs the
                           -45 dB noise floor of the pauses), so the bed is topped up there with the video's OWN room tone (steady noise-floor
                           frames from its non-speech parts, crossfaded) up to that floor.

Only the MUXED audio changes: LatentSync, the speech gate, the TTS QA and the dub-only metrics keep using the clean dubbed voice track.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import time

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEMUCS_WEIGHTS = ROOT / "models" / "demucs" / "hdemucs_high_trained.pt"   # scripts/download_models.sh; else torchaudio downloads it to the torch hub cache
SEP_SR = 44100
SOURCES = ["drums", "bass", "other", "vocals"]
SPLEETER_PY = pathlib.Path(os.environ.get("DUB_SPLEETER_PY", "/workspace/dub/venv-spleeter/bin/python"))   # scripts/setup_spleeter.sh
SPLEETER_MODELS = ROOT / "models" / "spleeter" / "pretrained_models"                                            # 2stems/ (sha256-checked release asset)
SPLEETER_TORCH_W = ROOT / "models" / "spleeter" / "2stems_tf_vars.npz"   # the same weights as npz (spleeter_torch.py convert, setup_spleeter.sh)
# r3 2026-10-19: torch port on the GPU (SNR vs TF 68 / 82 dB on 04 0-20 s, spleeter_torch.py verify) = default; tf = the TF 2.9 CPU venv
SPLEETER_BACKEND = os.environ.get("DUB_SPLEETER_BACKEND", "torch")
# r8-sep (29.10, user accepts the licence risk): NC separators = torchaudio HDEMUCS_HIGH_MUSDB_PLUS, Demucs v4 htdemucs / htdemucs_ft (MIT code,
# weights trained on MUSDB18-HQ = NC data). Their weights live OUTSIDE the scanned roots (clean_face_stack.banned_files: repo models/, third_party/,
# ~/.cache/torch/hub) so the licence guard of every other run on this server stays green; still refused unless DUB_ALLOW_NC_BED=1.
NC_SEPARATORS = ("hdemucs", "htdemucs", "htdemucs_ft")
NC_TORCH_HOME = pathlib.Path(os.environ.get("DUB_NC_TORCH_HOME", "/workspace/dub/quarantine_nc/torch"))
# bed of a 4-stem separator: stems = drums + bass + other (the r1 behaviour); residual = source - vocals (keeps whatever the model filed under no stem,
# e.g. the room noise HDemucs dropped under the voice: the reason for the room-tone fill)
SEP_BG = os.environ.get("DUB_SEP_BG", "stems")
DEMUCS_SHIFTS = int(os.environ.get("DUB_DEMUCS_SHIFTS", "1"))
SPLEETER_CODE = r"""
import os, sys, time, numpy as np, soundfile as sf, tensorflow as tf
n = int(os.environ.get("DUB_THREADS", "8")); tf.config.threading.set_intra_op_parallelism_threads(n); tf.config.threading.set_inter_op_parallelism_threads(1)
from spleeter.separator import Separator
y, sr = sf.read(sys.argv[1], dtype="float32"); assert sr == 44100
t0 = time.perf_counter(); sep = Separator("spleeter:2stems", multiprocess=False); s = sep.separate(y); t1 = time.perf_counter()
sf.write(sys.argv[2], s["accompaniment"], sr, subtype="FLOAT"); sf.write(sys.argv[3], s["vocals"], sr, subtype="FLOAT"); print("SEPARATE_S", round(t1 - t0, 3))
"""


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


def separate_spleeter_torch(wav: pathlib.Path, work: pathlib.Path, t0: float, device: str) -> dict:
    """Spleeter 2stems, PyTorch port (spleeter_torch.py) on `device`, model cached per process: bg = accompaniment, vocals = vocals."""
    import model_cache
    from spleeter_torch import Spleeter2Stems
    if not SPLEETER_TORCH_W.exists():
        raise RuntimeError(f"{SPLEETER_TORCH_W} missing: run scripts/setup_spleeter.sh")
    t1 = time.perf_counter(); sp = model_cache.get(("spleeter_torch", device), lambda: Spleeter2Stems(SPLEETER_TORCH_W, device)); load_s = time.perf_counter() - t1
    y, sr = sf.read(str(wav), dtype="float32"); assert sr == SEP_SR
    t2 = time.perf_counter(); out = sp.separate(y); sep_s = time.perf_counter() - t2
    bg, voc = out["accompaniment"], out["vocals"]; p_bg, p_voc = work / "bg_44k.wav", work / "vocals_44k.wav"
    sf.write(str(p_bg), bg, sr, subtype="FLOAT"); sf.write(str(p_voc), voc, sr, subtype="FLOAT"); wav.unlink()
    total = time.perf_counter() - t0
    return {"model": "Spleeter 2stems v1.4.0 (bg = accompaniment; MIT code, weights = MIT repo release asset) - PyTorch port spleeter_torch.py", "backend": f"torch/{device}",
            "weights": str(SPLEETER_TORCH_W), "bg": str(p_bg), "vocals": str(p_voc), "seconds": round(len(y) / SEP_SR, 3), "load_s": round(load_s, 3), "separate_s": round(sep_s, 3),
            "total_s": round(total, 3), "source_db": round(_db(y.mean(1)), 2), "vocals_db": round(_db(voc.mean(1)), 2), "bg_db": round(_db(bg.mean(1)), 2)}


def separate_spleeter(wav: pathlib.Path, work: pathlib.Path, t0: float) -> dict:
    """Spleeter 2stems in its own venv (CPU, TensorFlow): bg = accompaniment, vocals = vocals."""
    if not SPLEETER_PY.exists() or not (SPLEETER_MODELS / "2stems" / "model.index").exists():
        raise RuntimeError(f"Spleeter not set up ({SPLEETER_PY}, {SPLEETER_MODELS}/2stems): run scripts/setup_spleeter.sh")
    p_bg, p_voc = work / "bg_44k.wav", work / "vocals_44k.wav"
    env = {**os.environ, "MODEL_PATH": str(SPLEETER_MODELS), "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2"}
    for k in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"): env.pop(k, None)
    r = subprocess.run([str(SPLEETER_PY), "-I", "-c", SPLEETER_CODE, str(wav), str(p_bg), str(p_voc)], env=env, cwd=str(work), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Spleeter failed (rc {r.returncode}): {r.stderr[-800:]}")
    sep_s = float(next((ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("SEPARATE_S")), "nan"))
    y, _ = sf.read(str(wav), dtype="float32"); bg, _ = sf.read(str(p_bg), dtype="float32"); voc, _ = sf.read(str(p_voc), dtype="float32"); wav.unlink()
    total = time.perf_counter() - t0
    return {"model": "Spleeter 2stems v1.4.0 (bg = accompaniment; MIT code, weights = MIT repo release asset)", "weights": str(SPLEETER_MODELS / "2stems"),
            "bg": str(p_bg), "vocals": str(p_voc), "seconds": round(len(y) / SEP_SR, 3), "load_s": round(max(0.0, total - sep_s), 3), "separate_s": round(sep_s, 3),
            "total_s": round(total, 3), "source_db": round(_db(y.mean(1)), 2), "vocals_db": round(_db(voc.mean(1)), 2), "bg_db": round(_db(bg.mean(1)), 2)}


def separate_background(src: pathlib.Path, work: pathlib.Path, device: str = "cuda", separator: str = "spleeter") -> dict:
    """Source audio -> bg_44k.wav (accompaniment, stereo) and vocals_44k.wav (stereo). Returns paths + timings + levels."""
    t0 = time.perf_counter()
    wav = work / "source_44k.wav"; ff(["-i", str(src), "-vn", "-ac", "2", "-ar", str(SEP_SR), "-c:a", "pcm_f32le", str(wav)])
    if separator == "spleeter":
        if SPLEETER_BACKEND == "torch":
            from pilot_common import dev
            return separate_spleeter_torch(wav, work, t0, dev())
        return separate_spleeter(wav, work, t0)
    if separator not in NC_SEPARATORS:
        raise ValueError(f"unknown separator {separator}")
    if os.environ.get("DUB_ALLOW_NC_BED") != "1":
        raise RuntimeError(f"{separator}: weights trained on MUSDB18-HQ (non-commercial): set DUB_ALLOW_NC_BED=1 (r8-sep: deliberate NC deviation, user-approved)")
    import torch
    torch.hub.set_dir(str(NC_TORCH_HOME / "hub"))   # never the default ~/.cache/torch/hub (scanned by the licence guard)
    y, sr = sf.read(str(wav), dtype="float32"); assert sr == SEP_SR
    import model_cache
    mix = torch.from_numpy(np.ascontiguousarray(y.T))
    if separator == "hdemucs":
        t1 = time.perf_counter(); model = model_cache.get(("demucs", device), lambda: load_demucs(device)); t_load = time.perf_counter() - t1
        t2 = time.perf_counter(); s = _separate(model, mix, device); names = SOURCES
        what = "torchaudio HDEMUCS_HIGH_MUSDB_PLUS"; weights = str(DEMUCS_WEIGHTS) if DEMUCS_WEIGHTS.exists() else str(NC_TORCH_HOME / "hub")
    else:   # Demucs v4 (pip demucs 4.0.1, MIT code): htdemucs = 1 model, htdemucs_ft = bag of 4 per-source fine-tuned models
        t1 = time.perf_counter(); model = model_cache.get(("demucs_v4", separator, device), lambda: load_demucs_v4(separator, device)); t_load = time.perf_counter() - t1
        t2 = time.perf_counter(); s = _separate_v4(model, mix, device); names = list(model.sources)
        what = f"Demucs v4 {separator} (shifts {DEMUCS_SHIFTS}, overlap 0.25)"; weights = str(NC_TORCH_HOME / "hub" / "checkpoints")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_sep = time.perf_counter() - t2
    vi = names.index("vocals"); voc = s[vi].numpy().T.astype(np.float32)
    if SEP_BG == "residual":
        bg = (y - voc).astype(np.float32)
    else:
        bg = sum(s[i] for i in range(len(names)) if i != vi).numpy().T.astype(np.float32)
    del model, s
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    p_bg, p_voc = work / "bg_44k.wav", work / "vocals_44k.wav"
    sf.write(str(p_bg), bg, SEP_SR, subtype="FLOAT"); sf.write(str(p_voc), voc, SEP_SR, subtype="FLOAT"); wav.unlink()
    return {"model": f"{what} (bg = {'source - vocals' if SEP_BG == 'residual' else ' + '.join(n for n in names if n != 'vocals')}; NC weights)", "weights": weights,
            "separator": separator, "bg_mode": SEP_BG,
            "bg": str(p_bg), "vocals": str(p_voc), "seconds": round(len(y) / SEP_SR, 3), "load_s": round(t_load, 3), "separate_s": round(t_sep, 3),
            "total_s": round(time.perf_counter() - t0, 3), "source_db": round(_db(y.mean(1)), 2), "vocals_db": round(_db(voc.mean(1)), 2), "bg_db": round(_db(bg.mean(1)), 2)}


def load_demucs_v4(name: str, device: str = "cuda"):
    """Demucs v4 pretrained (htdemucs / htdemucs_ft); the checkpoints download once into NC_TORCH_HOME/hub/checkpoints (~80 MB per model)."""
    import torch
    torch.hub.set_dir(str(NC_TORCH_HOME / "hub"))
    from demucs.pretrained import get_model
    return get_model(name).to(device).eval()


def _separate_v4(model, mix, device: str):
    """demucs.apply.apply_model as the demucs CLI runs it (split into the model's segments, overlap 0.25, DEMUCS_SHIFTS random shifts), on the
    source normalised by its own mean / std; returns (S, C, L) on the CPU in model.sources order."""
    import torch
    from demucs.apply import apply_model
    ref = mix.mean(0); mu, sd = ref.mean(), ref.std().clamp_min(1e-6)
    with torch.no_grad():
        out = apply_model(model, ((mix - mu) / sd)[None], device=device, shifts=DEMUCS_SHIFTS, split=True, overlap=0.25, progress=False)[0]
    return (out.float().cpu() * sd + mu)


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


def bed_speech_runs(bg: np.ndarray, sr: int, thr: float = 0.5, min_s: float = 0.06, pad_s: float = 0.05) -> list:
    """Where the separated bed itself sounds like speech (Silero on the bed stem, 32 ms frames): the separator left part of the original voice in
    the accompaniment (r2 2026-10-17, Spleeter: 02 39.0-39.4 s a vocal sound p 0.99, 03 59.3 s a consonant onset). Returns padded (start, end) s."""
    import torch
    from silero_vad import load_silero_vad
    import scipy.signal as ss
    y16 = ss.resample_poly(bg.mean(1) if bg.ndim > 1 else bg, 160, 441).astype(np.float32); m = load_silero_vad(); w = 512; hop = w / 16000; p = []
    with torch.no_grad():
        for i in range(0, len(y16) - w, w):
            p.append(float(m(torch.from_numpy(y16[i:i + w].copy()), 16000)))
    on = np.array(p) > thr; out = []; i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]: j += 1
            if (j - i) * hop >= min_s: out.append((max(0.0, i * hop - pad_s), j * hop + pad_s))
            i = j
        else:
            i += 1
    return out


def mix_background(sep: dict, dubbed24k: pathlib.Path, source_speech: list, dub_speech: list, duration: float, work: pathlib.Path,
                   bg_offset_db: float = 0.0, duck_db: float = -6.0, max_bg_gain_db: float = 12.0, attack_s: float = 0.08, release_s: float = 0.35,
                   room_fill: bool = True, src_duck_db: float = 0.0, residue_duck_db: float = 0.0) -> dict:
    """final_mix_44k.wav (stereo) + final_mix_aac.m4a = dub voice (unchanged level, both channels) + background bed.
    Bed gain = (dub voice level in dub speech) - (original vocal level in source speech) + bg_offset_db, clamped to +-max_bg_gain_db, i.e. the
    original voice/background balance is reproduced; the bed is ducked by duck_db while the dub speaks (soft attack / release).
    src_duck_db < 0 (2026-10-13, "посторонние звуки"): the bed is ALSO lowered where the SOURCE speaks and the dub is silent: there HDemucs
    leaves a residue of the original voice in "other" (03 88.6 s: the source's harmonics in a dub pause, -19 dB rel. the dub voice), heard as
    foreign mumbling between dub phrases. Same soft envelope; elsewhere the bed is unchanged."""
    t0 = time.perf_counter()
    d44 = work / "dubbed_44k.wav"; ff(["-i", str(dubbed24k), "-ar", str(SEP_SR), "-ac", "1", "-c:a", "pcm_f32le", str(d44)])
    dub, _ = sf.read(str(d44), dtype="float32"); d44.unlink()
    bg, _ = sf.read(sep["bg"], dtype="float32"); voc, _ = sf.read(sep["vocals"], dtype="float32")
    n = int(round(duration * SEP_SR)); dub = np.pad(dub, (0, max(0, n - len(dub))))[:n]
    bg = np.pad(bg, ((0, max(0, n - len(bg))), (0, 0)))[:n]; voc = np.pad(voc, ((0, max(0, n - len(voc))), (0, 0)))[:n]
    fill = {"enabled": room_fill, "applied": False}
    if room_fill:   # under the source speech HDemucs removes the room noise with the voice: top the bed up to the video's own noise floor there
        tone, floor = room_tone(bg + voc, source_speech, SEP_SR, n)   # the full source: HDemucs files the steady noise floor under "vocals" too
        if tone is None:
            fill["skipped"] = "no steady non-speech material" if floor is None else f"floor {20 * np.log10(floor + 1e-12):.1f} dBFS > -42 (music / loud ambience)"
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
    if src_duck_db:
        src_only = _mask(source_speech, n, SEP_SR) & ~_mask(dub_speech, n, SEP_SR)
        env_s = _envelope(src_only, SEP_SR, attack_s, release_s) * (1.0 - env)          # never on top of the dub duck
        g = g * (10 ** (src_duck_db * env_s / 20))
    residue = {"enabled": bool(residue_duck_db), "runs": [], "seconds": 0.0}
    if residue_duck_db:   # r2: duck the bed only where IT sounds like speech, the source speaks and the dub is silent (voice leak of the separator)
        runs = [(a_, b_) for a_, b_ in bed_speech_runs(bg, SEP_SR)]
        rm = _mask(runs, n, SEP_SR) & _mask(source_speech, n, SEP_SR) & ~_mask(dub_speech, n, SEP_SR)
        if rm.any():
            env_r = _envelope(rm, SEP_SR, 0.04, 0.12) * (1.0 - env)
            g = g * (10 ** (residue_duck_db * env_r / 20))
        d_ = np.diff(np.concatenate([[0], rm.astype(np.int8), [0]])); on_, off_ = np.flatnonzero(d_ == 1), np.flatnonzero(d_ == -1)
        residue = {"enabled": True, "duck_db": residue_duck_db, "runs": [[round(a_ / SEP_SR, 2), round(b_ / SEP_SR, 2)] for a_, b_ in zip(on_, off_)],
                   "seconds": round(float(rm.sum() / SEP_SR), 2)}
    bed = bg * g[:, None]; mix = bed + dub[:, None]
    peak = float(np.abs(mix).max()) if n else 0.0; norm = 0.98 / peak if peak > 0.98 else 1.0; mix *= norm
    out = work / "final_mix_44k.wav"; sf.write(str(out), mix, SEP_SR, subtype="PCM_16")
    aac = work / "final_mix_aac.m4a"; ff(["-i", str(out), "-c:a", "aac", "-b:a", "192k", str(aac)])
    pause = ~_mask(dub_speech, n, SEP_SR)
    return {"wav": str(out), "aac": str(aac), "room_fill": fill, "residue_duck": residue, "bg_gain_db": round(gain_db, 2), "duck_db": duck_db, "src_duck_db": src_duck_db, "attack_s": attack_s, "release_s": release_s, "bg_offset_db": bg_offset_db,
            "dub_voice_db": round(voice_db, 2), "orig_vocals_db": round(orig_voice_db, 2), "peak_norm": round(norm, 4),
            "bed_db_in_dub_pauses": round(_db(bed.mean(1)[pause]), 2) if pause.any() else None,
            "mix_db_in_dub_pauses": round(_db(mix.mean(1)[pause]), 2) if pause.any() else None,
            "seconds": round(n / SEP_SR, 3), "mix_s": round(time.perf_counter() - t0, 3)}
