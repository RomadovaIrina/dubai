# r3-integration (Agent D, round 3) — changelog

Branch `experiment/r3-integration` from `clean-base` 326e807. Server: RTX 5090, 24 vCPU.

## Merges
| # | merge | conflicts / resolution |
|---|---|---|
| 1 | `r2-lipsync-position` 92de423 (A: fp32 AlignRestore, flow anchors, head-locked mask, 05 coverage) | none |
| 2 | `r2-composite-position` 6341e2e (B: FIX30 source-fps composite) | `anchor_filters.py` = A's superset (blend filter, rigid thread cap); `slide_metric.py` identical (mode); B's `r2_sbs.sh` kept, A's as `r2_sbs_A.sh`; B's dub-B3 pip freeze dropped. Tree = B's r3-face merge 98c8333 apart from env records |
| 3 | `r2-audio` a1fd764 (C: L1 + floor 0.875, Spleeter bed, declick, duck −4 dB / 250 ms) | runner argparse: C's `--background on` (Spleeter 2stems, MIT) + `--audio-policy/--declick/--bed-separator` taken; HDEMUCS stays behind `DUB_ALLOW_NC_BED` |

## Tooling
- `candidate_video_stage.py --dump-ls-raw on`: the same run also writes `<out>_ls_raw.mp4` (raw LatentSync frames on the master,
  lossless) = the `--ls-raw` input of `fps_composite.py` (FIX30), so no second LatentSync pass (B used `decompose_video_stage.py`).
- `master_timing.py`: regenerates B's `<id>_master_timing.json` (master frame → source frame; generator was not committed).
  03 / 04: master lags its clock by 66 ms mean (max 80 ms), matches B's finding.
- `r3_video.sh <arm> <ids>`: video stage on canonical cand5 audio (arms `clean0`, `r2m`).
- `r3_scorecard.py score|table`: per-video JSON + Markdown table (9.4, coverage, SyncNet, v4, slide, joints, feather, WER, LUFS, speed).

## Licences
Nothing new. Spleeter 2stems (MIT code, MIT release weights) set up in `/workspace/dub/venv-spleeter` via `scripts/setup_spleeter.sh`.
