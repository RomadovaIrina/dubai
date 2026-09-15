# DabAI pilot 0.1–0.11 — measurement toolkit

This directory completes the **measurement harness**, not the product pipeline. It is
built around the existing frozen environment, smoke tests and these already-present files:

- `scripts/pilot/pilot_common.py`
- `scripts/pilot/check_flashattention.py`
- `scripts/pilot/measure_vram.py`
- `scripts/pilot/measure_stages.py`
- `scripts/check_*.py|sh`

All compact evidence is written to `reports/pilot/`; generated media/work directories are
kept in `/tmp` where possible.

## Mapping

| Pilot | Script | Formal closure condition |
|---|---|---|
| 0.1 | `validate_01_blackwell.py --run-smoke` | sm_120 + CUDA 12.8 + torch cu128 + real smoke |
| 0.2 | `validate_02_latentsync.py --run-smoke` | real LatentSync inference under immutable baseline |
| 0.3 | `probe_03_flashattention_build.sh` | source build attempted on sm_120; PASS **or reproducible FAIL** closes experiment |
| 0.4 | `run_vram_suite.py` + `resident_vram_probe.py` | measured peaks + residency -> RESIDENT/UNLOAD decision for 32 GiB |
| 0.5 | `benchmark_05_gpu_coefficient.py` | clean real pipeline, CodeFormer disabled, final number on target GPU |
| 0.6 | `benchmark_06_codeformer.py` | CodeFormer + LatentSync s/video-min, final run on Blackwell |
| 0.7 | `benchmark_07_codeformer_syncnet.py` | SyncNet before/after sweep and selected fidelity weight |
| 0.8 | `benchmark_08_syncnet_dataset.py` | exactly five real generated videos measured |
| 0.9 | `benchmark_09_qwen_quant.py` | Q5_K_M vs Q8_0 CPU speed + agreed quality on >=3 languages |
| 0.10 | `benchmark_10_cold_start.py` | repeated start-to-ready timing on target multi-stage + network-NVMe topology |
| 0.11 | existing `measure_stages.py`; dataset wrapper `benchmark_11_s_dataset.py` | VAD/split + concat + subtitle burn measured |

## Start with an environment snapshot

```bash
source scripts/env.sh
python scripts/pilot/env_snapshot.py
```

## 0.1 / 0.2

Final evidence belongs on RTX 5090 / sm_120:

```bash
python scripts/pilot/validate_01_blackwell.py --run-smoke
python scripts/pilot/validate_02_latentsync.py --run-smoke
```

## 0.3 FlashAttention

The probe builds `flash-attn==2.8.3.post1` into `/tmp`, installs it into an isolated target
directory and runs the real CUDA kernel. It does **not** modify `/venv/dabai`.

```bash
MAX_JOBS=4 bash scripts/pilot/probe_03_flashattention_build.sh
```

A reproducible build failure on **sm_120** is a valid experimental result: the task asks
whether it builds. A pass on sm_89 is only a preparation run.

## 0.4 VRAM

First collect isolated real-inference peaks:

```bash
python scripts/pilot/run_vram_suite.py --clear
```

Then load models simultaneously and keep them resident:

```bash
python scripts/pilot/resident_vram_probe.py --profile audio --target-gib 32
python scripts/pilot/resident_vram_probe.py --profile video --target-gib 32
python scripts/pilot/resident_vram_probe.py --profile all   --target-gib 32
```

The 48 GiB 4090 is useful here precisely because it can reveal a 25/30/35 GiB steady
working set without first OOMing at 32 GiB. Final confirmation on the target 5090 is still
recommended for borderline results.

## 0.5 base GPU coefficient

This script intentionally does not invent a second E2E pipeline. Give it the **real** clean
pipeline command, with CodeFormer explicitly disabled:

```bash
python scripts/pilot/benchmark_05_gpu_coefficient.py /path/to/pilot_videos \
  --command-template 'python -m YOUR_PIPELINE --input {input} --output {output} --no-codeformer' \
  --repeat 2
```

It samples the process tree in `nvidia-smi` and computes GPU occupied seconds / source
video seconds. A 4090 run is marked provisional; formal closure requires sm_120.

## 0.6 CodeFormer

```bash
python scripts/pilot/benchmark_06_codeformer.py /path/to/04.mp4 \
  --repeat 1 \
  --latentsync-s-per-min 123.4
```

`--latentsync-s-per-min` must come from the same target configuration. Without it the
CodeFormer measurement is still useful but the combined 300 s/video-min verdict is open.

## 0.7 SyncNet before/after CodeFormer

LatentSync upstream already contains its SyncNet evaluator. This script uses that exact
evaluator and sweeps CodeFormer fidelity weights:

```bash
python scripts/pilot/benchmark_07_codeformer_syncnet.py /path/to/latentsync_output.mp4 \
  --weights 0.3 0.5 0.7 0.9 \
  --max-confidence-drop 0 \
  --max-offset-worsening-frames 0
```

The source specification does not define a tolerance. The defaults are therefore strict;
change them only as an explicit project decision.

## 0.8 five real videos

Point this at the five **generated** real-content videos that should be evaluated:

```bash
python scripts/pilot/benchmark_08_syncnet_dataset.py /path/to/five_outputs --expect 5
```

It reports every score plus mean/median/min/max, but deliberately does not choose the
contract threshold because the source spec does not define how the five values are to be
aggregated.

## 0.9 Qwen Q5_K_M vs Q8_0

Disk is tight, so download quantizations explicitly:

```bash
python scripts/pilot/fetch_09_qwen_quant.py q5_k_m --yes
python scripts/pilot/fetch_09_qwen_quant.py q8_0 --yes
```

Create a JSONL dataset with at least three actual project languages. The bundled
`qwen_eval.example.jsonl` is only an example, not the contractual dataset.

```bash
python scripts/pilot/benchmark_09_qwen_quant.py \
  --dataset scripts/pilot/qwen_eval.jsonl
```

If `sacrebleu` is available and references exist, chrF/BLEU are recorded. Otherwise the
script writes all translations to CSV and leaves quality open for human review rather than
pretending an unspecified metric is authoritative.

## 0.10 cold start

Do **not** close this on a normal already-running Vast box. Run on the target topology:

```bash
python scripts/pilot/benchmark_10_cold_start.py \
  --start-command './infra/start_node.sh' \
  --ready-command './infra/is_ready.sh' \
  --stop-command './infra/stop_node.sh' \
  --repeat 3 \
  --confirm-target-topology
```

`is_ready.sh` must return 0 only when the worker and required weights are actually ready
for a job, not merely when SSH/HTTP is reachable.

## 0.11 S

Single video:

```bash
python scripts/pilot/measure_stages.py /path/to/04.mp4 --repeat 2 --asr
```

All five:

```bash
python scripts/pilot/benchmark_11_s_dataset.py /path/to/five_videos --repeat 2 --asr
```

## Status

```bash
python scripts/pilot/pilot_status.py
# or
bash scripts/pilot/run_pilot.sh status
```

The status script distinguishes **formal CLOSED** from provisional dev-GPU results.
