#!/usr/bin/env python3
"""eval_run.evaluate (per-unit Whisper WER / CTC CER of the dub track, the pipeline's independent-ASR QA) on the CPU for offline timing A/B
(Agent C, server B shares its GPU): faster-whisper large-v3 int8 + wav2vec2-base-960h on the CPU. Absolute values differ slightly from the GPU
fp16 numbers of the baseline table -> compare policies with each other under this script only.

    CUDA_VISIBLE_DEVICES= python scripts/quality/eval_cpu.py --manifest AB/03_x/recut.manifest.json [--manifest ...] --out AB/eval
"""
from __future__ import annotations
import argparse, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "diag", "quality"): sys.path.insert(0, str(HERE.parent / d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import faster_whisper
    _WM = faster_whisper.WhisperModel
    class CpuWM(_WM):   # tts_qa.TtsQA builds WhisperModel(..., device, compute_type="float16"): force int8 on the CPU
        def __init__(self, path, device="cpu", compute_type="int8", **kw):
            import os
            super().__init__(path, device="cpu", compute_type="int8", cpu_threads=int(os.environ.get("DUB_THREADS", "6")), **kw)
    faster_whisper.WhisperModel = CpuWM
    import tts_qa, eval_run
    qa = tts_qa.TtsQA(device="cpu"); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for mp in a.manifest:
        man = json.loads(pathlib.Path(mp).read_text()); W = pathlib.Path(man["work_dir"]); tag = f"{W.parent.parent.name}_{W.name}" if W.name.startswith("work_") else W.name   # runner work dirs: <root>/work/work_<id>
        r = eval_run.evaluate(man, str(W / "dubbed_24k.wav"), label=tag, qa=qa)
        (out / f"{tag}.eval.json").write_text(json.dumps(r, indent=1, ensure_ascii=False)); s = r["summary"]
        print(f"{tag:28s} WER {s['wer_whisper_mean']} CER {s['cer_ctc_mean']} bad {s['units_bad(cer_ctc>=0.3)']} interj {s['interjection_units']} final_miss {s['final_word_missing_units']}", flush=True)
    qa.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
