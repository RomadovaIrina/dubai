#!/usr/bin/env python3
"""r8-sep: the RAW separated bed (before gap_clean / cap / room fill) of several separators on the same video, against the r8v4 dub timing.
20-ms frames; S = source speech (r8v4 manifest speech_intervals), P = source pauses (S +-0.3 s off), G = dub gaps of r8v4 (dub VAD silent, 60 ms
off the dub edges: where the bed is heard alone), GS = G & S (the separator removed the voice there: leaks and "farts" live here).
  leak    vad_S_s      Silero p >= 0.5 on the bed in S (s)          vad_GS_s   p >= 0.4 (gap_clean's threshold) in GS (s)
          vband_S_db   voice-band (24 mel 250-4000 Hz) excess of the bed in S over the bed's own per-band median in P (dB, positive part)
          words_GS     faster-whisper large-v3 (ru) on the bed alone in every GS run >= 0.3 s (+-0.1 s): words p >= 0.5, no_speech <= 0.6
  junk    burst_S_s    bed > local ambience (30th pct of the bed in P within +-3 s) + 6 dB in S        clicks / clicks_S   scripts/r7/clicks.py on the bed
  natural lvl_S_db     median bed level in S - median SOURCE level in P (the ambience should carry on under the voice: 0 = continuous)
          flat_S       mean spectral flatness (100 Hz-8 kHz) of the bed in S;  flat_P_src = the same for the source in P (the room itself)
          lsd_S_db     RMS dB distance of the mean 64-mel log spectrum of the bed in S vs the source in P (lower = the bed under speech sounds like the room)
          lvl_P_db / lsd_P_db  the same for the bed in P (a separator should leave pauses alone)
    python scripts/r8/bed_metrics.py --var spleeter=/workspace/dub/r8a/stems_spleeter --var htdemucs=... --ref-tag r8v4c --out X.json 01 02 ..."""
from __future__ import annotations
import os
import argparse, json, pathlib, subprocess, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "r7"))
A = pathlib.Path("/workspace/dub/artifacts/r4a"); RAW = pathlib.Path(os.environ.get("DUB_INPUTS", "/workspace/dub/inputs/test_videos_raw")); FR = 0.02; SR = 44100
HALLU = ("продолжение следует", "субтитр", "редактор", "корректор", "спасибо за просмотр", "подписывайтесь")


def fmask(ivs, n, pad=0.0):
    m = np.zeros(n, bool)
    for a, b in ivs: m[max(0, int((a - pad) / FR)):min(n, int(np.ceil((b + pad) / FR)))] = True
    return m


def runs(m, min_len=1):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def source44(id_):
    p = pathlib.Path(f"/tmp/r8sep_src_{id_}.wav")
    if not p.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(next(RAW.glob(f"{id_}.*"))), "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(p)], check=True)
    return sf.read(str(p), dtype="float32")[0]


def spec(mono, n):
    import librosa
    h = int(FR * SR)
    P = np.abs(librosa.stft(mono, n_fft=2048, hop_length=h, center=False)) ** 2; P = np.pad(P, ((0, 0), (0, max(0, n - P.shape[1]))), mode="edge")[:, :n]
    f = librosa.fft_frequencies(sr=SR, n_fft=2048); band = (f >= 100) & (f <= 8000); Pb = P[band] + 1e-12
    flat = np.exp(np.log(Pb).mean(0)) / Pb.mean(0)
    mel = librosa.feature.melspectrogram(S=P, sr=SR, n_mels=64, fmin=60, fmax=12000)
    vb = librosa.feature.melspectrogram(S=P, sr=SR, n_mels=24, fmin=250, fmax=4000)
    return flat, 10 * np.log10(mel + 1e-12), 10 * np.log10(vb + 1e-12)


def lev(mono, n):
    h = int(FR * SR); m = min(n, len(mono) // h); L = np.full(n, -120.0); L[:m] = 20 * np.log10(np.sqrt((mono[:m * h].reshape(m, h) ** 2).mean(1)) + 1e-9); return L


def analyse(id_, bed, src, S, P, G, wm):
    import gap_clean as gc, clicks
    n = len(S); mono = bed.mean(1); sm = src.mean(1)
    Lb, Ls = lev(mono, n), lev(sm, n)
    flat_b, mel_b, vb_b = spec(mono, n); flat_s, mel_s, _ = spec(sm, n)
    p, ph = gc.bed_voice_prob(bed, SR); pv = np.interp((np.arange(n) + 0.5) * FR, (np.arange(len(p)) + 0.5) * ph, p)
    GS = G & S
    ref = np.full(n, np.percentile(Lb[P], 30)); pi = np.flatnonzero(P); k = int(3 / FR)
    for i in range(0, n, 25):
        sel = Lb[pi[np.searchsorted(pi, i - k):np.searchsorted(pi, i + k)]]
        if len(sel) >= 25: ref[i:i + 25] = np.percentile(sel, 30)
    vbx = np.clip(vb_b - np.median(vb_b[:, P], 1, keepdims=True), 0, None).mean(0)
    room = mel_s[:, P].mean(1)
    lsd = lambda M, m: float(np.sqrt(np.mean((M[:, m].mean(1) - room) ** 2))) if m.any() else None
    ck = clicks.detect(bed, SR); ckS = [c for c in ck if S[min(n - 1, int(c["t"] / FR))]]
    r = {"vad_S_s": round(float((S & (pv >= 0.5)).sum() * FR), 2), "vad_GS_s": round(float((GS & (pv >= 0.4)).sum() * FR), 2),
         "vband_S_db": round(float(vbx[S].mean()), 2), "vband_GS_db": round(float(vbx[GS].mean()), 2) if GS.any() else None,
         "burst_S_s": round(float((S & (Lb > ref + 6)).sum() * FR), 2), "burst_GS_s": round(float((GS & (Lb > ref + 6)).sum() * FR), 2),
         "clicks": len(ck), "clicks_S": len(ckS),
         "lvl_S_db": round(float(np.median(Lb[S]) - np.median(Ls[P])), 2), "lvl_P_db": round(float(np.median(Lb[P]) - np.median(Ls[P])), 2),
         "flat_S": round(float(flat_b[S].mean()), 4), "flat_P_src": round(float(flat_s[P].mean()), 4), "flat_P": round(float(flat_b[P].mean()), 4),
         "lsd_S_db": round(lsd(mel_b, S), 2), "lsd_P_db": round(lsd(mel_b, P), 2), "S_s": round(float(S.sum() * FR), 1), "GS_s": round(float(GS.sum() * FR), 2)}
    if wm is not None:
        import librosa
        y16 = librosa.resample(mono, orig_sr=SR, target_sr=16000); ws_all = []; rows = []
        for a, b in runs(GS, int(0.3 / FR)):
            t0, t1 = max(0.0, a * FR - 0.1), b * FR + 0.1; seg = y16[int(t0 * 16000):int(t1 * 16000)]
            ss, _ = wm.transcribe(seg, language="ru", word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
            ws = [w.word.strip() for q in ss if q.no_speech_prob <= 0.6 and not any(h in q.text.lower() for h in HALLU) for w in (q.words or []) if w.probability >= 0.5]
            if ws: rows.append({"t": [round(t0, 2), round(t1, 2)], "words": " ".join(ws)}); ws_all += ws
        r["words_GS"] = len(ws_all); r["words_GS_rows"] = rows
    r["leak_runs_GS"] = [[round(a * FR, 2), round(b * FR, 2)] for a, b in runs(GS & (pv >= 0.4), 3)][:60]
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--var", action="append", required=True, help="name=stem_dir (stem_dir/<id>/bg_44k.wav)")
    ap.add_argument("--ref-tag", default="r8v4c"); ap.add_argument("--out", required=True); ap.add_argument("--asr", default="on"); ap.add_argument("ids", nargs="+"); a = ap.parse_args()
    import run_clean_pipeline_05 as rp
    wm = None
    if a.asr == "on":
        import torch  # noqa: F401  (libcublas before ctranslate2)
        from faster_whisper import WhisperModel
        wm = WhisperModel(str(HERE.parents[1] / "models" / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
    out = pathlib.Path(a.out); R = json.loads(out.read_text()) if out.exists() else {}
    for id_ in a.ids:
        run = A / a.ref_tag / id_; man = json.loads(next(run.glob(f"{id_}_*.manifest.json")).read_text())
        src = source44(id_); n = len(src) // int(FR * SR)
        sp = [(s["start"], s["end"]) for s in man["speech_intervals"]]; S = fmask(sp, n); P = ~fmask(sp, n, 0.3)
        dub = [(s["start"], s["end"]) for s in rp.run_vad(run / "dubbed_16k.wav")]; G = ~fmask(dub, n, 0.06)
        R.setdefault(id_, {})
        for v in a.var:
            name, d = v.split("=", 1); bed = sf.read(str(pathlib.Path(d) / id_ / "bg_44k.wav"), dtype="float32")[0]
            bed = np.pad(bed, ((0, max(0, len(src) - len(bed))), (0, 0)))[:len(src)]
            sj = pathlib.Path(d) / id_ / "sep.json"; r = analyse(id_, bed, src, S, P, G, wm)
            if sj.exists(): j = json.loads(sj.read_text()); r["s_per_video_min"] = j.get("s_per_video_min"); r["model"] = j.get("model")
            R[id_][name] = r
            print(id_, name, {k: v_ for k, v_ in r.items() if k not in ("leak_runs_GS", "words_GS_rows", "model")}, flush=True)
            out.write_text(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
