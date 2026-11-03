#!/usr/bin/env python3
"""r13 spec 6.2 test: the production ASR call (run_clean_pipeline_05.run_vad + run_asr) on a stress file built from a real video:
    [speech A] + [digital silence] + [room tone -55 dBFS] + [music, no voice] + [speech B] + [music] + [silence]
Speech = 20 s pieces of the source (16 kHz mono); music = the htdemucs / Spleeter bed of a video (no vocals), else a synthetic chord bed.
For every inserted non-speech stretch: the ASR text that lands there with the filter OFF (= hallucination candidates) and with it ON
(the production default); for the speech stretches: the transcript must be the same with the filter on and off (nothing real dropped).
    python scripts/r13/hallu_test.py --src 03.mp4 [--music bed.wav] --out DIR"""
import argparse, json, pathlib, subprocess, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import run_clean_pipeline_05 as P  # noqa: E402

SR = 16000


def load(p, t0=0.0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), *(["-t", str(dur)] if dur else []), "-i", str(p), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).copy()


def chords(dur, seed=0):
    """a synthetic music bed (no voice): chord pads + a kick, -20 dBFS"""
    r = np.random.default_rng(seed); t = np.arange(int(dur * SR)) / SR; y = np.zeros_like(t)
    roots = [220.0, 174.6, 196.0, 261.6]
    for i, f0 in enumerate(roots * int(dur // 8 + 1)):
        m = (t >= i * 2) & (t < i * 2 + 2)
        for k in (1, 1.25, 1.5, 2): y[m] += np.sin(2 * np.pi * f0 * k * t[m]) * 0.15
    kick = np.exp(-np.arange(int(0.15 * SR)) / (0.03 * SR)) * np.sin(2 * np.pi * 60 * np.arange(int(0.15 * SR)) / SR)
    for b in np.arange(0, dur, 0.5): i = int(b * SR); y[i:i + len(kick)] += kick[:len(y) - i] * 0.6
    y += r.normal(0, 0.003, len(y)); return (y / np.abs(y).max() * 0.1).astype(np.float32)


def text_in(segs, a, b):
    return " | ".join(s["text"] for s in segs if min(s["end"], b) - max(s["start"], a) > 0.5 * (s["end"] - s["start"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--music"); ap.add_argument("--out", required=True)
    ap.add_argument("--speech-a", type=float, default=0.0); ap.add_argument("--speech-b", type=float, default=25.0); ap.add_argument("--lang", default=None)
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rt = np.random.default_rng(1).normal(0, 10 ** (-55 / 20), 15 * SR).astype(np.float32)
    mus = load(a.music, 0, 60) if a.music else chords(30); m1, m2 = mus[:15 * SR], mus[15 * SR:30 * SR] if len(mus) >= 30 * SR else mus[:15 * SR]
    m1 = m1 / (np.abs(m1).max() + 1e-9) * 0.3; m2 = m2 / (np.abs(m2).max() + 1e-9) * 0.3
    parts = [("speech_A", load(a.src, a.speech_a, 20)), ("silence", np.zeros(15 * SR, np.float32)), ("room_tone_-55dB", rt), ("music_1", m1),
             ("speech_B", load(a.src, a.speech_b, 20)), ("music_2", m2), ("silence_tail", np.zeros(10 * SR, np.float32))]
    y, spans, t = [], [], 0.0
    for n, x in parts: y.append(x); spans.append((n, t, t + len(x) / SR)); t += len(x) / SR
    wav = out / "stress_16k.wav"; sf.write(wav, np.concatenate(y), SR, subtype="PCM_16")
    speech = P.run_vad(wav); res = {"file": str(wav), "spans": spans, "music": a.music or "synthetic chords", "vad_speech_s": round(sum(v["end"] - v["start"] for v in speech), 2)}
    for mode in ("off", "on"):
        P.ASR_OPTS.update(hallu_filter=mode, segments="off")
        segs, meta = P.run_asr(wav, a.lang, speech)
        res[mode] = {"segments": [{k: s[k] for k in ("start", "end", "text", "no_speech_prob", "avg_logprob", "compression_ratio")} for s in segs],
                     "by_span": {n: text_in(segs, s0, s1) for n, s0, s1 in spans}, "filter": meta.get("hallu_filter")}
    res["verdict"] = {n: {"filter_off": res["off"]["by_span"][n], "filter_on": res["on"]["by_span"][n]} for n, *_ in spans}
    nonsp = [n for n, *_ in spans if not n.startswith("speech")]
    res["summary"] = {"hallucinated_spans_filter_off": [n for n in nonsp if res["off"]["by_span"][n]], "hallucinated_spans_filter_on": [n for n in nonsp if res["on"]["by_span"][n]],
                      "speech_text_identical": all(res["off"]["by_span"][n] == res["on"]["by_span"][n] for n, *_ in spans if n.startswith("speech")),
                      "dropped": len((res["on"]["filter"] or {}).get("dropped", [])), "flagged_kept": len((res["on"]["filter"] or {}).get("flagged_kept", []))}
    (out / "hallu_test.json").write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(json.dumps(res["summary"], ensure_ascii=False))
    for n, v in res["verdict"].items(): print(f"  {n:16s} OFF: {v['filter_off'][:90]!r}\n  {'':16s} ON : {v['filter_on'][:90]!r}")


if __name__ == "__main__":
    main()
