#!/usr/bin/env python3
"""Qwen on GPU vs CPU check (Agent C r3, 2026-10-19): replay the runner's Qwen stages (sentence segmentation + translation) on the stored
ASR of a finished run and compare them with that run (e.g. audio-r2, Qwen Q8_0 on the CPU). Reports load / segmentation / translation
seconds, peak VRAM of this process (nvidia-smi, sampled) and identical-output counts.

    PYTHONPATH=/workspace/dub/c/pylib-llama-cuda python scripts/quality/qwen_gpu_check.py --manifest RUN/03_r2.manifest.json --gpu-layers -1 --quant q8_0
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, threading, time
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality"): sys.path.insert(0, str(HERE.parent / d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--gpu-layers", type=int, default=-1); ap.add_argument("--quant", default="q8_0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import run_clean_pipeline_05 as rp
    from sentence_units import sentence_units
    rp.QWEN_GPU_LAYERS = a.gpu_layers; rp.QWEN_QUANT = a.quant
    man = json.loads(pathlib.Path(a.manifest).read_text()); segs = man["asr"]["segments"]; dur = man["source"]["duration_s"]
    src = man["source_lang"]; peak = [0]; stop = threading.Event()

    def sample():
        while not stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            for ln in out.splitlines():
                p, m = [x.strip() for x in ln.split(",")]
                if p == str(os.getpid()): peak[0] = max(peak[0], int(m))
            time.sleep(0.5)
    th = threading.Thread(target=sample, daemon=True); th.start()
    times: dict = {}
    llm, name = rp.load_llm(times)
    t0 = time.perf_counter(); units, rep = sentence_units(segs, llm, rp.lang_name(src), dur, 20.0); times["segmentation"] = round(time.perf_counter() - t0, 3)
    ref_units = man["units"]
    # translation on the reference run's own units (after its diarization + tiny merge), so translation is compared on identical input
    tu = [{k: u[k] for k in ("id", "start", "end", "text", "words", "speaker", "asr_segment") if k in u} | ({"sentence": u["sentence"]} if "sentence" in u else {}) for u in ref_units]
    plans = rp.burst_plan(tu, segs, man["speech_intervals"], dur, 0.0, True)
    rp.run_translation(tu, src, "en", times, plans=plans, keep_llm=False, llm=llm, split=False)
    stop.set(); th.join()
    seg_ref = [u["text"] for u in ref_units]; seg_new = [u["text"] for u in units]
    outs = lambda chunks: [((c.get("attempts") or [{}])[-1]).get("out") for c in chunks] if isinstance(chunks, list) else None
    same_seg = outs(rep.get("chunks")) == outs(man["segmentation"].get("chunks")) if outs(man["segmentation"].get("chunks")) else None
    tr_same = sum(x["translation"] == y["translation"] for x, y in zip(tu, ref_units))
    res = {"manifest": a.manifest, "quant": a.quant, "gpu_layers": a.gpu_layers, "model": name, "times_s": times, "peak_vram_mib": peak[0],
           "segmentation": {"sentences_new": rep.get("sentences"), "sentences_ref": man["segmentation"].get("sentences"), "units_new": len(units),
                            "chunk_outputs_identical": same_seg},
           "translation": {"units": len(tu), "identical": tr_same,
                           "diffs": [{"id": x["id"], "ref": y["translation"], "new": x["translation"]} for x, y in zip(tu, ref_units) if x["translation"] != y["translation"]]}}
    print(json.dumps({k: v for k, v in res.items() if k != "translation"} | {"translation_identical": f"{tr_same}/{len(tu)}"}, ensure_ascii=False))
    for d in res["translation"]["diffs"]: print(f"  u{d['id']:02d} CPU: {d['ref']}\n      GPU: {d['new']}")
    if a.out: pathlib.Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
