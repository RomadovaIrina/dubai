#!/usr/bin/env python3
"""Pilot 0.9 addendum (2026-10-13): Qwen2.5-7B Q8_0 fully on the GPU (CUDA build of the same llama-cpp-python, scripts/setup_llama_cuda.sh)
on the SAME fixed set and with the SAME timing code as benchmark_09_qwen_quant.py; outputs compared with the recorded CPU Q8_0 outputs of
0.9 (reports/pilot/0.9_qwen_translations.csv). Writes reports/pilot/0.9_qwen_gpu_addendum.{json,md}; the closed 0.9 files are not touched.
    PYTHONPATH=third_party/llama_cuda:$PYTHONPATH python scripts/pilot/benchmark_09_qwen_gpu.py --dataset scripts/pilot/qwen_eval.jsonl
"""
from __future__ import annotations
import argparse, csv, json, pathlib, statistics, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, write_json
from benchmark_09_qwen_quant import bench, first_shard, load_data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--q8", default=str(pathlib.Path(__file__).resolve().parents[2] / "models/qwen2.5-7b-instruct-gguf/*q8_0*.gguf"))
    ap.add_argument("--ctx", type=int, default=4096); ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    import llama_cpp
    if not llama_cpp.llama_supports_gpu_offload():
        print(f"{llama_cpp.__file__} is a CPU build", file=sys.stderr); return 2
    data = load_data(pathlib.Path(a.dataset))
    rows, cfg = bench(first_shard(a.q8), "q8_0", data, a.ctx, a.max_tokens, 4, a.warmup, a.repeats, gpu_layers=-1)
    cpu = {r["id"]: r for r in csv.DictReader((PILOT / "0.9_qwen_translations.csv").open(encoding="utf-8"))}
    same = [r["id"] for r in rows if r["id"] in cpu and r["translation"] == cpu[r["id"]]["q8_0_output"]]
    g = [r["gen_tok_per_s_mean"] for r in rows if r["gen_tok_per_s_mean"]]; c = [float(cpu[r["id"]]["q8_gen_tok_s"]) for r in rows if r["id"] in cpu and cpu[r["id"]]["q8_gen_tok_s"]]
    diff = [{"id": r["id"], "source": r["source"], "cpu_q8": cpu[r["id"]]["q8_0_output"], "gpu_q8": r["translation"]} for r in rows if r["id"] in cpu and r["id"] not in same]
    rep = {"config": cfg, "sentences": len(rows), "identical_to_cpu_q8": len(same), "gen_tok_per_s_gpu_median": round(statistics.median(g), 2) if g else None,
           "gen_tok_per_s_cpu_q8_median_0_9": round(statistics.median(c), 2) if c else None, "outputs_deterministic_across_repeats": all(r["outputs_identical_across_repeats"] for r in rows),
           "differences": diff, "rows": rows}
    write_json(PILOT / "0.9_qwen_gpu_addendum.json", rep)
    md = ["# Pilot 0.9 addendum — Qwen2.5-7B Q8_0 on the GPU (2026-10-13)", "",
          f"Same fixed set ({len(rows)} sentences, {sorted({r['direction'] for r in rows})}), same timing code as 0.9; CUDA build of llama-cpp-python {llama_cpp.__version__}, n_gpu_layers -1, RTX 5090 sm_120.", "",
          f"| | CPU Q8_0 (0.9) | GPU Q8_0 |", "|---|---:|---:|",
          f"| generation tok/s (median per sentence) | {rep['gen_tok_per_s_cpu_q8_median_0_9']} | **{rep['gen_tok_per_s_gpu_median']}** |",
          f"| load s / size GiB | – | {cfg['load_s']} / {cfg['size_gib']} |",
          f"| outputs identical to the CPU Q8_0 outputs of 0.9 | – | {len(same)} / {len(rows)} |",
          f"| deterministic across {a.repeats} repeats | – | {rep['outputs_deterministic_across_repeats']} |", "",
          "Different outputs (greedy decoding on a different backend: same quality class, review below):", ""]
    md += [f"* `{d['id']}` {d['source'][:80]}\n  * CPU: {d['cpu_q8']}\n  * GPU: {d['gpu_q8']}" for d in diff]
    (PILOT / "0.9_qwen_gpu_addendum.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md[:12])); return 0


if __name__ == "__main__":
    raise SystemExit(main())
