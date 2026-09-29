#!/usr/bin/env python3
"""Do E1's phrase cuts fall inside words? (Phase A acceptance: "no cut words")

E1 splits each TTS clip at Silero pauses and, when there are fewer pauses than original bursts, at energy valleys. This script
gets word timestamps of every TTS clip with the project's faster-whisper (existing model, word_timestamps=True — the same call as
the pipeline's ASR, here on clean synthetic English, so the transcript is near-perfect) and checks every cut position of
e1_alignment.json against the word intervals: a cut strictly inside [word.start + 40 ms, word.end - 40 ms] is a cut word.
    python scripts/quality/e1_cut_words.py --id 04 --e1-dir /tmp/dabai_quality/candidates/e1/work_04 --run-dir /tmp/dabai_quality/baseline
"""
from __future__ import annotations
import argparse, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--e1-dir", required=True); ap.add_argument("--run-dir", default="/tmp/dabai_quality/baseline")
    ap.add_argument("--tol-ms", type=int, default=40); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch  # noqa: F401
    from faster_whisper import WhisperModel
    from pilot_common import MODELS
    man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_baseline.manifest.json").read_text()); rep = json.loads((pathlib.Path(a.e1_dir) / "e1_alignment.json").read_text())
    m = WhisperModel(str(MODELS / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
    tol = a.tol_ms / 1000; rows = []; n_cuts = n_bad = 0
    for u, r in zip(man["units"], rep["units"]):
        segs, _ = m.transcribe(u["tts"]["file"], word_timestamps=True, beam_size=5, language="en", vad_filter=False)
        words = [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or [])]
        # cut points = boundaries between consecutive phrases of the TTS clip (e1_alignment.json phrase_spans); a boundary that is
        # a real pause (gap >= 100 ms) cannot cut a word, so only tight boundaries (valley cuts) are checked against the words
        spans = r.get("phrase_spans") or []; bad = []; tight = 0
        for (s0, e0), (s1, e1_) in zip(spans, spans[1:]):
            n_cuts += 1
            if s1 - e0 >= 0.1:
                continue
            tight += 1; c = (e0 + s1) / 2
            for w, ws, we in words:
                if ws + tol < c < we - tol:
                    bad.append({"cut_s": round(c, 3), "word": w, "word_span": [round(ws, 3), round(we, 3)]}); n_bad += 1; break
        rows.append({"unit": u["id"], "translation": u["translation"], "words": len(words), "phrases": len(spans), "tight_cuts": tight, "cut_words": bad,
                     "word_timestamps": [(w, round(ws, 2), round(we, 2)) for w, ws, we in words]})
    res = {"id": a.id, "tol_ms": a.tol_ms, "cuts": n_cuts, "cut_words": n_bad, "units": rows}
    print(json.dumps({k: v for k, v in res.items() if k != "units"}))
    for r in rows:
        if r["cut_words"]:
            print(f"  u{r['unit']:02d} {len(r['cut_words'])}/{r['tight_cuts']} valley cuts inside words: " + ", ".join(f"'{b['word']}'@{b['cut_s']}" for b in r["cut_words"]))
    out = pathlib.Path(a.out or (pathlib.Path("/tmp/dabai_quality/manifests") / f"{a.id}_e1.cut_words.json")); out.write_text(json.dumps(res, indent=1)); print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
