#!/usr/bin/env python3
"""r7 gap metric + listen clips: what the listener hears while the DUB is silent (user on r6 m5: "farts" and original-voice fragments in dub gaps).
Gaps = dub-VAD silent runs >= 0.15 s (60 ms off the dub edges) of a remix dir <tag>/<id>/. On the FINAL mix, 20 ms frames:
  gap_voice_s   Silero speech p >= 0.5 on the mix in gaps (the dub is silent there, so any speech is a leak of the original voice)
  gap_burst_s   mix level > local ambience + 6 dB (ambience = 30th percentile of the mix in joint pauses within +-3 s): bursts / farts
  gap_junk_db   p95 of (mix level - local ambience) over gap frames
  gap_vband_db  mean voice-band excess (24 mel bands 250-4000 Hz, dB above the per-band median of the joint pauses, positive part) over gap frames:
                quiet separation warble ("farts") that the level rules miss;  gap_vband_frac = share of gap frames with excess > 6 dB
  *_src_speaks  the same over the gap frames where the SOURCE speaks: where the separator removed the voice (leaks / farts live there; in joint
                pauses the bed is the original's own sound)
  gap_words     faster-whisper (source language) on all gaps joined: words with p >= 0.5 (leaked original speech; known silence hallucinations dropped)
Listen clips (--clips DIR): <id>_<tag>_gaps.m4a = every gap with 0.3 s context, 0.4 s silence + 30 ms tick between; with --prev also
<id>_gaps_AB.m4a = per gap PREV then NEW (two ticks before NEW), and <id>_gaps.md (timecodes, per-gap values PREV / NEW).
    python scripts/r7/gap_metrics.py --tag NEW [--prev PREV] [--clips DIR] [--asr on] 01 02 ..."""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot"))
A = pathlib.Path("/workspace/dub/artifacts/r4a"); FR = 0.02
HALLU = ("продолжение следует", "субтитр", "редактор", "корректор", "спасибо за просмотр", "подписывайтесь")


def gaps_of(run: pathlib.Path, min_s=0.15, edge=0.06):
    import run_clean_pipeline_05 as rp
    dub = [(s["start"], s["end"]) for s in rp.run_vad(run / "dubbed_16k.wav")]; dur = sf.info(str(run / "final_mix_44k.wav")).duration
    out = []; t = 0.0
    for a, b in dub + [(dur, dur)]:
        if a - t - 2 * edge >= min_s: out.append((round(t + (edge if t > 0 else 0), 3), round(a - (edge if a < dur else 0), 3)))
        t = max(t, b)
    return out, dub


def frame_db(y, sr):
    h = int(FR * sr); n = len(y) // h
    return 20 * np.log10(np.sqrt((y[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)


def analyse(run: pathlib.Path, man: dict, asr: bool):
    import gap_clean as gc
    y, sr = sf.read(str(run / "final_mix_44k.wav"), dtype="float32"); mono = y.mean(1) if y.ndim > 1 else y
    gaps, dub = gaps_of(run); L = frame_db(mono, sr); n = len(L)
    S = gc._mask_frames([(s["start"], s["end"]) for s in man["speech_intervals"]], n); G = gc._mask_frames(gaps, n)
    joint = G & ~gc._mask_frames([(s["start"], s["end"]) for s in man["speech_intervals"]], n, 0.3)
    ref = np.full(n, np.percentile(L[joint], 30) if joint.sum() > 25 else np.percentile(L[G], 30) if G.any() else -90.0)
    ji = np.flatnonzero(joint); k = int(3 / FR)
    for i in range(0, n, 25):
        sel = L[ji[np.searchsorted(ji, i - k):np.searchsorted(ji, i + k)]]
        if len(sel) >= 25: ref[i:i + 25] = np.percentile(sel, 30)
    p, ph = gc.bed_voice_prob(y, sr); pv = np.interp((np.arange(n) + 0.5) * FR, (np.arange(len(p)) + 0.5) * ph, p)
    V = G & (pv >= 0.5); B = G & (L > ref + 6); ex = (L - ref)[G]
    import librosa
    M = librosa.feature.melspectrogram(y=mono, sr=sr, n_fft=2048, hop_length=int(FR * sr), n_mels=24, fmin=250, fmax=4000, center=False)
    Mb = 10 * np.log10(M + 1e-12)[:, :n]; Mb = np.pad(Mb, ((0, 0), (0, n - Mb.shape[1])), mode="edge")
    rb = np.median(Mb[:, joint], 1, keepdims=True) if joint.sum() > 25 else np.median(Mb, 1, keepdims=True)
    VX = np.clip(Mb - rb, 0, None).mean(0)
    rows = []
    for a, b in gaps:
        i0, i1 = int(a / FR), min(n, int(np.ceil(b / FR)))
        rows.append({"t": [a, b], "src_speaks": bool(S[i0:i1].any()), "voice_s": round(float(V[i0:i1].sum() * FR), 2), "burst_s": round(float(B[i0:i1].sum() * FR), 2),
                     "max_over_amb_db": round(float((L - ref)[i0:i1].max()), 1) if i1 > i0 else None, "vband_db": round(float(VX[i0:i1].mean()), 2) if i1 > i0 else None})
    r = {"gaps": len(gaps), "gap_s": round(float(G.sum() * FR), 2), "gap_voice_s": round(float(V.sum() * FR), 2), "gap_burst_s": round(float(B.sum() * FR), 2),
         "gap_junk_db_p95": round(float(np.percentile(ex, 95)), 2) if len(ex) else None,
         "gap_vband_db": round(float(VX[G].mean()), 2) if G.any() else None, "gap_vband_frac": round(float((VX[G] > 6).mean()), 3) if G.any() else None,
         "gap_voice_s_src_speaks": round(float((V & S).sum() * FR), 2), "gap_burst_s_src_speaks": round(float((B & S).sum() * FR), 2),
         "gap_vband_db_src_speaks": round(float(VX[G & S].mean()), 2) if (G & S).any() else None, "rows": rows}
    if asr and gaps:
        import run_clean_pipeline_05 as rp, librosa, tempfile
        y16 = librosa.resample(mono, orig_sr=sr, target_sr=16000); sil = np.zeros(8000, np.float32); parts = []
        for a, b in gaps: parts += [y16[int(a * 16000):int(b * 16000)], sil]
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "g.wav"; sf.write(str(f), np.concatenate(parts), 16000); segs, _ = rp.run_asr(f, man.get("source", {}).get("language") or "ru")
        words = [w for s in segs if not any(h in s["text"].lower() for h in HALLU) for w in s["words"] if w["p"] >= 0.5]
        r["gap_words"] = len(words); r["gap_words_text"] = " ".join(w["word"].strip() for w in words)[:400]
    return r


def tick(sr, n=1):
    t = np.arange(int(0.03 * sr)) / sr; b = (0.2 * np.sin(2 * np.pi * 1500 * t) * np.hanning(len(t))).astype(np.float32); s = np.zeros(int(0.06 * sr), np.float32)
    return np.concatenate([np.concatenate([b, s]) for _ in range(n)])


def clip(y, sr, a, b, ctx=0.3):
    return y[max(0, int((a - ctx) * sr)):int((b + ctx) * sr)]


def write_m4a(y, sr, out):
    w = out.with_suffix(".wav"); sf.write(str(w), y, sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(w), "-c:a", "aac", "-b:a", "128k", str(out)], check=True); w.unlink()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--prev"); ap.add_argument("--clips"); ap.add_argument("--asr", default="on")
    ap.add_argument("ids", nargs="+"); a = ap.parse_args()
    for id_ in a.ids:
        run = A / a.tag / id_; man = json.loads((run / f"{id_}_r4b.manifest.json").read_text())
        r = analyse(run, man, a.asr == "on"); (A / a.tag / f"{id_}.gaps.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(id_, a.tag, {k: v for k, v in r.items() if k != "rows"}, flush=True)
        if not a.clips: continue
        C = pathlib.Path(a.clips); C.mkdir(parents=True, exist_ok=True); y, sr = sf.read(str(run / "final_mix_44k.wav"), dtype="float32"); pause = np.zeros((int(0.4 * sr),) + y.shape[1:], np.float32)
        tk = lambda k: np.repeat(tick(sr, k)[:, None], y.shape[1], 1) if y.ndim > 1 else tick(sr, k)
        write_m4a(np.concatenate([x for g in r["rows"] for x in (clip(y, sr, *g["t"]), pause, tk(1))] or [pause]), sr, C / f"{id_}_{a.tag}_gaps.m4a")
        if a.prev:
            pr = A / a.prev / id_; yp, _ = sf.read(str(pr / "final_mix_44k.wav"), dtype="float32"); P = json.loads((A / a.prev / f"{id_}.gaps.json").read_text()) if (A / a.prev / f"{id_}.gaps.json").exists() else None
            write_m4a(np.concatenate([x for g in r["rows"] for x in (tk(1), clip(yp, sr, *g["t"]), pause, tk(2), clip(y, sr, *g["t"]), pause, pause)] or [pause]), sr, C / f"{id_}_gaps_AB.m4a")
            md = [f"# {id_} dub gaps: PREV `{a.prev}` vs NEW `{a.tag}`", "", "AB clip `" + f"{id_}_gaps_AB.m4a" + "`: one tick = PREV, two ticks = NEW, gaps in time order (0.3 s context).", ""]
            def pv_(g):   # PREV value over the same span (PREV frames)
                if not P: return ""
                return next((q.get("vband_db") for q in P["rows"] if q["t"][0] < g["t"][1] and q["t"][1] > g["t"][0]), "")
            if P: md += [f"PREV: voice {P['gap_voice_s']} s, bursts {P['gap_burst_s']} s, voice-band {P.get('gap_vband_db')} dB ({P.get('gap_vband_frac')}), junk p95 {P['gap_junk_db_p95']} dB, words {P.get('gap_words')} ({P.get('gap_words_text', '')[:120]})"]
            md += [f"NEW:  voice {r['gap_voice_s']} s, bursts {r['gap_burst_s']} s, voice-band {r.get('gap_vband_db')} dB ({r.get('gap_vband_frac')}), junk p95 {r['gap_junk_db_p95']} dB, words {r.get('gap_words')} ({r.get('gap_words_text', '')[:120]})", "",
                   "| # | gap | src speaks | NEW voice s | NEW burst s | NEW max over amb dB | NEW voice-band excess dB | PREV voice-band excess dB |", "|---:|---|---|---:|---:|---:|---:|---:|"]
            md += [f"| {i + 1} | {int(g['t'][0] // 60)}:{g['t'][0] % 60:05.2f}-{g['t'][1]:.2f} | {'yes' if g['src_speaks'] else ''} | {g['voice_s']} | {g['burst_s']} | {g['max_over_amb_db']} | {g.get('vband_db')} | {pv_(g)} |" for i, g in enumerate(r["rows"])]
            (C / f"{id_}_gaps.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
