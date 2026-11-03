#!/usr/bin/env python3
"""r13 spec 6.2 regression on real videos: the production ASR call with the hallucination filter OFF vs ON (whole file, as e2e runs it).
Every dropped / trimmed / flagged segment is listed with its scores and a 16 kHz listen clip (<out>/<id>_drop<k>.wav), so a human can confirm
nothing real was dropped. Also the 5.2 VAD pieces (--asr-segments vad) per video: cut points, pause length / level, transcript diff vs whole-file.
    python scripts/r13/hallu_real.py --out DIR video1.mp4 [video2 ...]"""
import argparse, difflib, json, pathlib, subprocess, sys
import soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import run_clean_pipeline_05 as P  # noqa: E402


def words(segs): return [w.lower().strip(".,!?…") for s in segs for w in s["text"].split()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--segments", choices=["on", "off"], default="on")
    ap.add_argument("videos", nargs="+"); a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = {}
    for v in a.videos:
        vid = pathlib.Path(v).stem; wav = out / f"{vid}_16k.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        speech = P.run_vad(wav); r = {"duration_s": round(sf.info(str(wav)).duration, 2), "speech_s": round(sum(x["end"] - x["start"] for x in speech), 2)}
        P.ASR_OPTS.update(hallu_filter="off", segments="off"); off, _ = P.run_asr(wav, None, speech)
        P.ASR_OPTS.update(hallu_filter="on", segments="off"); on, meta = P.run_asr(wav, None, speech); rep = meta["hallu_filter"]
        y, sr = sf.read(str(wav))
        for k, d in enumerate(rep["dropped"] + rep["trimmed"]):
            sf.write(out / f"{vid}_drop{k}.wav", y[max(0, int((d["start"] - 1) * sr)):int((d["end"] + 1) * sr)], sr)
        r.update(segments_off=len(off), segments_on=len(on), words_off=len(words(off)), words_on=len(words(on)),
                 dropped=rep["dropped"], trimmed=rep["trimmed"], flagged_kept=rep["flagged_kept"])
        if a.segments == "on":
            P.ASR_OPTS.update(hallu_filter="on", segments="vad"); sv, m2 = P.run_asr(wav, None, speech)
            x, z = words(on), words(sv); sm = difflib.SequenceMatcher(None, x, z, autojunk=False)
            r["vad_segments"] = {"pieces": m2["chunks"]["pieces"], "cuts": m2["chunks"]["cuts"],
                                 "word_changes_vs_whole": sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"), "words": len(x),
                                 "diff": [f"{' '.join(x[i1:i2])} -> {' '.join(z[j1:j2])}" for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"][:20]}
        rows[vid] = r
        print(f"== {vid}: {r['duration_s']} s, segs {len(off)} -> {len(on)}, dropped {len(rep['dropped'])} trimmed {len(rep['trimmed'])} flagged {len(rep['flagged_kept'])}"
              + (f"; vad pieces {len(r['vad_segments']['pieces'])} cuts {[c.get('t') for c in r['vad_segments']['cuts']]} word changes {r['vad_segments']['word_changes_vs_whole']}" if 'vad_segments' in r else ""), flush=True)
        for d in rep["dropped"] + rep["trimmed"]: print(f"   drop {d['start']}-{d['end']} {d['why']} nsp={d['no_speech_prob']} lp={d['avg_logprob']} cr={d['compression_ratio']} vad={d['vad_coverage']} {d['text'][:80]!r}")
    (out / "hallu_real.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
