# Round 3, track A (lip-sync, generator side) — changelog

Branch `experiment/r3-lipsync` (from `experiment/r2-lipsync-position` @ 92de423). Server A (RTX 5090, CPU quota ~31 cores shared with C).
No new model weights in this round so far (LatentSync 1.6 Apache-2.0, MediaPipe Apache-2.0, SyncNet / S3FD only inside the eval).

## Infrastructure
- `scripts/nothp`: runs a job with transparent huge pages off (prctl) + 8-thread pools + optional CPU pinning (`DUB_CPUS`).
  This host stalls in THP compaction: 512 MB touch > 100 s without, 0.3 s with; a 4 s window 203 s -> 60 s; ffmpeg too.
- CPU cap (coordinator): GPU jobs on CPUs 0-7, metric jobs on 8-15 (`artifacts/r3/cpupin.sh` keeps it).
- `r3_gpuq.sh`: GPU job queue (one LatentSync job at a time, window batch 1 = ~20 GB VRAM); eval queue on the CPU side.
- `r3_prep.sh` / `r3_run.sh` / `r3_arms.sh`: masters, full-video runs, window arms on top of r2best.
- Audio: all round-3 deliverables use track C **audio-r2** (`handoff/in/C/audio-r2`, Spleeter bed): `artifacts/r3/audio-r2/<id>.manifest.json`
  (C's units / placements, the full-run speech gate) + `work_<id>` (C's dub / mix, canonical source audio). cand5 audio = reference only.

## Measurement
- `sync_perframe.py`: per-frame SyncNet distances (same detector / crop / model as LatentSync's eval) -> per-second conf / offset; lip gap
  (MediaPipe on the output) vs dub log-RMS correlation by lag; dub / source activity per second.
- `r3_eval.py` / `r3_table.py`: SyncNet (mix upstream, dub-only via sync_perframe), per-second stats on dub-speech seconds, v4 (mouth vel / jerk
  / resid / sharp, frozen mouth in dub-silent seconds, LatentSync window-joint jumps), slide, LatentSync s per LS frame, audio column.

## Generator changes (all opt-in flags)
- `--silence-mix LAM` (`r3_lipsync.py`): dub-silent mouth — whisper conditioning moved toward the source speech by LAM where the source talks and
  no dub part sounds (manifest placements or C's `residual.json`), cosine-eased ramps; continuous in time (no snap at a window start).
- `--ref-mode shiftS` (`ls_stable.py`): UNet reference of frame i = the current-frame latents of frame i+S (training-like reference, no copy of
  the frame's own source-language mouth); `fixed` (a calm closed-mouth 16-frame clip from elsewhere) from round 1, first measured now.
- `--dump-raw` (`candidate_video_stage.py`): full-length lossless raw LatentSync layer for track B's source-fps composite.
- Spec 11: `--inject-ls-error` fault injection + the 2 % SUCCESS / PARTIAL_SUCCESS status in `candidate_video_stage.py` (as in run_clean_pipeline_05).
- Spec 5.4 / 5.5: `r3_pose.py`; spec 8.2: `r3_twoface.py` (synthetic two-face clip + untouched-face check).
