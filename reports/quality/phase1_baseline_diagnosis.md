# Quality — Phase 1 baseline measurement + Phase 2 diagnosis (2026-09-22, videos 01–04; 05 pending upload)

Config: exactly `reports/pilot/FINAL_ML_CONFIG.md` canonical command (optimized backend, RetinaFace router stride 3, gate 0.24/0.6,
crossfade 4, DeepCache 5, batch 2, SDPA auto, 20 steps, guidance 1.5, fp16, seed 1247, Qwen Q8_0, CodeFormer OFF). One run per video.
Runner: `scripts/quality/run_baseline.sh`; analysis: `scripts/quality/quality_manifest.py`, `face_metrics.py`, `router_stats.py`.
Media: `/tmp/dabai_quality/{baseline,work,frames}`, copies of outputs in `result_videos/baseline/`. Nothing in the pipeline was changed.
Environment: RTX 5090 sm_120, torch 2.7.1+cu128, `/venv/dabai` (bootstrap 2026-09-22, smoke 10/10).

## Baseline table

| video | source dur / fps / frames | output dur / fps / frames | VALID/SMALL/NO_FACE/NO_SPEECH frames | LS segments (transitions) | LS inference s | peak VRAM MiB | units / VAD intervals | tts/slot ratio min/med/max | units >1.25x / <0.6 | LS frames with SILENT dub | …while original speaks | SyncNet out / original | mouth sharpness out/master | upper-face sharpness | flicker out/master |
|---|---|---|---|---|---:|---:|---|---|---|---|---:|---|---|---|---|
| 01 | 101.667 / 25.0033 / 2542 | 101.68 / 25.0 / 2542 | 261/177/1996/108 | 1 (1 tr.) | 77.9 | 22928 | 29 / 25 | 0.58/0.96/2.08 | 3 / 1 | 45/260 (17%) | 40 | 6.41 / 5.36 | 0.72–0.72 | 0.87–0.87 | 0.93–0.93 |
| 02 | 110.867 / 29.982 / 3324 | 110.92 / 25.0 / 2773 | 0/0/2195/578 | 0 | 0 | 5460 | 12 / 29 | 0.25/0.60/1.35 | 2 / 6 | – | – | N/A (no face) | – | – | – |
| 03 | 90.333 / 30.0 / 2710 | 90.4 / 25.0 / 2260 | 1966/0/0/294 | 8 (16 tr.) | 465.02 | 23073 | 10 / 27 | 0.42/0.67/2.48 | 1 / 4 | 777/1966 (40%) | 433 | 4.59 / 4.89 | 0.51–0.54 | 0.84–0.89 | 0.73–0.86 |
| 04 | 53.9 / 30.0 / 1617 | 53.96 / 25.0 / 1349 | 1097/0/0/252 | 7 (14 tr.) | 263.62 | 23049 | 5 / 15 | 0.40/0.48/0.58 | 0 / 5 | 494/1097 (45%) | 422 | 2.8 / 3.24 | 0.59–0.79 | 0.62–0.76 | 0.92–0.97 |
All 4 outputs: validation PASS (duration Δ ≤ 0.07 s, resolution preserved, frames = 25-fps master, 0 blank/undecodable, dubbed AAC present,
0 LatentSync failures, 0 "Face not detected"). AV offset = 0 frames on every scoreable output. 02 has no face at all (SyncNet N/A, expected).
Baseline reproduces pilot 0.8 (SyncNet 04 2.80 vs 2.76, 03 4.59 vs 4.65, 01 6.41 vs 6.84).

## Diagnosis — suspected bottleneck → evidence → proposed experiment

### C1. TTS is concatenated at the start of a long slot; the rest of the slot is silence while the person on screen keeps talking — DOMINANT
- Evidence: units are whole whisper segments (04: 5 units over 50.6 s of slots, VAD sees 15 speech bursts / 29.5 s). Total TTS (24.5 s)
  ≈ total original speech (29.5 s), but each TTS clip is placed at slot start at natural speed and padded with silence (ratio 0.40–0.58).
  Result: **45 % of LatentSync frames in 04 and 40 % in 03 are driven by silent dubbed audio; 422 (04) and 433 (03) of those frames are
  moments where the original speaker is audibly talking.** LatentSync draws a closed/neutral mouth there while head/body gesture continues →
  "mouth looks fake / face is dead", and the dubbed voice sounds like bursts followed by long gaps. SyncNet of the dub is below the original
  (04 2.80 vs 3.24, 03 4.59 vs 4.89) although the AV offset is 0 — the audio simply is not there for a large part of the lip-synced frames.
  Contact sheet `frames/04_baseline/mouth_seg005.png` f593: output mouth closed, master shows teeth.
- Not a LatentSync problem and not fixable by LatentSync parameters.
- Experiment E1 (Phase 3 §6): distribute the TTS over the original speech bursts instead of the slot start. Candidate implementation as an
  optional path in a wrapper (no change to run_clean_pipeline_05.py defaults): (a) keep translation per sentence; (b) run Silero VAD on the TTS
  clip to split it into phrases; (c) assign phrases to the unit's original VAD bursts in order (cumulative-duration matching); (d) fit each phrase
  to its burst with atempo capped (e.g. ≤1.25) and allow the burst to extend into adjacent silence when the timeline permits.
  Metrics: "LS frames with silent dub while original speaks" → target ≈ 0, SyncNet, atempo distribution, human listen (voice pacing).

### C2 / D. Sentence fragmentation and missing punctuation (conversational videos 02/03/04)
- Evidence: whisper large-v3 returns no punctuation on 03/04 speech; units cut mid-sentence ("…for creating our" | "magical application which…"),
  Qwen mirrors the input (no punctuation, fragments), Chatterbox then synthesises fragments with flat prosody. ASR errors propagate
  ("стебель" → "stem" in 01). No mixed-script / CJK / preamble / over-length problems found (length ratio 0.86–1.09 on 04, guardrail scan).
- Experiment E2 (Phase 3 §7): translation prompt variant asking for a fluent, punctuated spoken-style translation (still "translation only"),
  A/B on 04/03 by manual listening of raw TTS (`work_<id>/tts_u*.wav`) + duration/ratio distribution; unit re-segmentation to sentence
  boundaries is coupled with E1 (bursts), evaluate together only after E1.

### C3. Short-phrase speed-ups on narrated content (01)
- Evidence: one-word sentences get 0.6–1.0 s slots (first..last word timestamps) → "Bathroom." 2.08x, "Balcony." 1.45x, "Prime finishing…"
  1.38x; 3 units >1.25x. Audible rushing; 17 % silent LS frames only.
- Experiment E3 (Phase 3 §6): slot extension into the following silence (gap to next unit) before atempo, atempo cap 1.25–1.3. Cheap, A/B on 01.

### A. Face texture: LatentSync softens the whole pasted face square, not only the mouth — SECONDARY (systematic, upstream behaviour)
- Evidence: mouth-crop Laplacian sharpness output/master 0.51–0.54 (03), 0.59–0.79 (04), 0.72 (01); upper face (eyes/nose, untouched by
  lip-sync) 0.62–0.89 → the affine warp → 512² → inverse warp + blend resamples the entire face crop. Upper-face PSNR 37–40 dB (identity kept,
  texture lost). Temporal flicker ratio 0.73–0.97 (no added flicker). Boundary jumps at every LS↔pass-through transition equal the master's
  (crossfade 4 works; no blink/jump at 30 transitions measured).
- Experiment E4: assembly-side "mouth-only paste-back" — blend the LatentSync frame into the master only inside the lower-face mask (our wrapper,
  upstream code untouched) → expected upper_sharp_ratio → 1.0, mouth unchanged, SyncNet unchanged. A/B on 04 with face_metrics + SyncNet.
- Experiment E5 (Phase 3 §1/§2, requested first): DeepCache 5 / 3 / off, steps 20 / 25 / 30, guidance 1.0 / 1.5 / 2.0, batch 1 vs 2 on one real
  segment (04 seg 1, 1.44–16.36 s, 373 frames) — sharpness / flicker / SyncNet / time / VRAM. Expectation from the pilot data: PSNR 44 dB
  between interval 3 and 5, so gains will be small; the probe settles it.

### B. Routing — NOT a source of artifacts on this dataset
- Evidence (`router_stats.py`, every master frame, conf 0.02): RetinaFace scores are bimodal (≥0.9 or ≤0.3) on all four videos — confidence
  0.5–0.8 would change 0 frames. The only borderline run is 01 @ 89.9–96.9 s: face 60–80 px high, score ≥0.9 → SMALL_FACE (7 s of talking face
  without lip-sync). min_h 60 would route it, but LatentSync's own detector rejects h<80 → only the pilot-0.8 crop+upscale path could serve it.
  Stride 3: 0 verify splits on 03/04, 1 split on 01 at a hard cut (10.4 s). 02: no face anywhere (correct pass-through).
- Experiment E6 (cheap, routing-only, no LatentSync): stride 1/2/3 diff on 01/03/04; gate margin 0.24/0.35/0.45 and merge gap 0.6/0.8/1.0
  measured as transitions count + silent-fraction; crossfade 0/2/4/8 boundary-jump metric on 04. Low expected impact; run after E1/E5.

### Other observations (recorded, not acted on)
- 30-fps sources (02/03/04) become a 25-fps master → every 6th frame dropped in the WHOLE output (1617 → 1349 frames), lip-synced or not:
  slight motion judder vs the original. Inherent to LatentSync's 25-fps pipeline; a fix would be re-timing outside LatentSync (not in scope now).
- Voice reference: 10 s of the speaker's own diarized speech (2 longest turns) — fine on 01–04 (single speaker each); Chatterbox fp16 OK, no fallbacks.
- Runtime on this host is CPU-quota bound (13.6 vCPU): 04 wall 406 s vs 315 s in pilot; not a quality factor.

## Plan (one parameter group per A/B)
1. E5 LatentSync probes on 04 seg 1 (DeepCache → steps → guidance → batch) — cheap, isolates the face-quality ceiling.
2. E1 burst-aligned TTS placement (optional path) — the dominant defect; A/B 04 + 03, then all videos.
3. E3 slot extension / atempo cap — 01.
4. E4 mouth-only paste-back — 04.
5. E6 gate/crossfade/stride — only if E1 leaves visible on/off switching.
6. E2 prompt variant — after E1 (interacts with segmentation).
