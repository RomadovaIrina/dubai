# Track B — restoration / temporal stabilization (`experiment/restoration-temporal`) — STATUS

| phase | state | evidence |
|---|---|---|
| 0 baseline reproduction | **done 2026-10-16, accepted by the coordinator** | `env_phase0.md`, `baseline_phase0.md`, `baseline/`, `/workspace/dub/artifacts/baseline/<id>/` |
| 1 prior work, diagnosis plan, shortlist, licenses | **done** (go given; pre-approved to continue) | `PHASE1.md`, `LICENSES.md`, `LICENSES_desk_research.md` |
| 2 cheapest A/B on a 03 window | decomposition done; candidates C1-C3 next | `decomp_03_w1050_1506/`, `metric_mouth_jitter_hr.md`, `/workspace/dub/artifacts/restoration-temporal/` |
| 3 refine | – | – |
| 4 E2E regression | – | – |
| 5 final | – | – |

Branch base: `tts-intelligibility-fix` @ `559d700e0bf14ffa5692d855e5eae10a42eadd11` (verified 2026-10-16, remote not moved).
Baseline = cand5 (`reports/quality/feedback_2026-10-13/README.md`, "new candidate"), CPU Qwen, run with `scripts/pilot/refresh_e2e.sh`
(`baseline/baseline_run.sh` = exactly this, one run root per video):

```bash
DUB_EXTRA_ARGS="--phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive --tts-gap-qa on --gate-merge-gap-s 1.2 \
  --mouth-only-pasteback on --codeformer-scope mouth --mask-smooth 3 --ls-landmark-smooth 2" \
  PILOT_REFRESH_ROOT=/workspace/dub/artifacts/baseline/<id> bash scripts/pilot/refresh_e2e.sh <id>
```

Phase 0 result: 5/5 valid; SyncNet dub-only 7.86 / – / 5.69 / 3.81 / 3.66, AV offset 0; wall 549 / 256 / 881 / 426 / 500 s
(324 / 139 / 585 / 474 / 572 s per video-minute); peak VRAM 22.4-23.5 GB. Table: `baseline_phase0.md`.

Open flags for the coordinator:
* License: CodeFormer (baseline restoration) is NTU S-Lab 1.0 = non-commercial; InsightFace buffalo_l (LatentSync detector + the 106
  landmarks this track's CodeFormer alignment / mask / smoothing use) is non-commercial research-only. See `LICENSES.md`.
* Metric gap: fixed — common metric `scripts/quality/mouth_jitter_hr.py` (25aedba).
* Cross-track: the 16-frame window-boundary jump is created inside LatentSync (independent windows); a generator-side fix belongs to track A.
