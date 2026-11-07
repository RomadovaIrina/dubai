# r16 Q: DUB_FACE=auto: active speaker + beard routing (07.11)

Three CPU scripts:

| script | what | runtime (45 s clip) |
|---|---|---|
| `scripts/quality/r16q_active_speaker.py` | SyncNet (S3FD tracks + syncnet_v2, = sync_perframe) on the SOURCE video + SOURCE audio → per diarization turn: does a visible face speak it? | CPU 24 threads ~3–5 min (`--device cuda` ~20 s) |
| `scripts/quality/r16q_beard.py` | per-face beard / stubble score on 40 sampled frames (MediaPipe) → `face_engine` | ~20 s CPU |
| `scripts/r16q/face_route.py` | merges both → `route.json` | instant |

```
python  scripts/quality/r16q_active_speaker.py --video master_25fps.mp4 --manifest <dub manifest> --out as.json [--device cuda]
/venv/mediapipe/bin/python scripts/quality/r16q_beard.py <source video> --out beard.json
python3 scripts/r16q/face_route.py --active as.json --beard beard.json --out route.json
```

## route.json = the hook for the e2e (T)

- `keep_source_all`: no turn has a visible speaker → skip the face stage (voice-only).
- `keep_source_ranges`:
  - seconds where diarization has speech but no visible face speaks it in the source (an off-screen voice, a silent listener on screen);
  - **the paste must show the SOURCE face there** (no lip-sync; cross-fade ~3 frames);
  - the dub audio plays unchanged;
  - padded by 0.12 s, minus the on-screen turns.
- `face_engine`: `latentsync` when the main on-screen speaker has a beard / stubble, else `keysync`.
  - `DUB_FACE_ENGINE` overrides it;
  - an unmeasured beard (too dark / small / profile) → `keysync`.
- `lipsync_track` + `lipsync_box_center_xy`: the S3FD track that speaks most, to match MediaPipe tracks by centre.

## Active speaker: rules and results

A turn counts as on screen when either:
- SyncNet conf ≥ 2.0, with the minimum within ±3 frames of 0; or
- conf ≥ 1.0 at exactly the whole-clip offset of a track that speaks overall (conf ≥ 2). This covers low-res / occluded faces.

A silent face scores at random offsets with conf < 1.2. Turns shorter than 1.5 s inherit their speaker's result.

A mouth-landmark ↔ speech-envelope correlation was tried first and fails: 0.07 even on SyncNet-synced KeySync output.

| clip | result | keep_source |
|---|---|---|
| **q02_vera** | Vera speaks 0–6.2 s (conf 5.9 / 4.2). **7.1–40.3 s: off-screen narrator** (Vera conf 0.04–1.1 at random offsets). From 40.9 s the man is on screen (conf 7.1). | **31.6 s** |
| **m03** | mother track conf 7.0. The child's 1 s turn overlaps the mother's speech → mapped to the mother; the child's face (conf 0.13) is never lip-synced. | 0 |
| **q07_altapress** | S3FD finds the interviewee (2 shots, conf 2.6 / 2.4) that MediaPipe missed; 4 reporters ~0.0. | 0 |
| q01 / q03 / q04 | single speaker, conf 2.9 / 3.7 / 2.8 | 0 |
| q05_rozhkov | the speaker in 6 track pieces (head turns, cup) conf 2.3–5.3; SPEAKER_02 / 03 (off-screen interviewer?) conf 1.1 / 0.2 | 5.3 s |
| q06_suvorov | 12 track pieces (dark, profile) conf 1.2–3.6 | 2.7 s |
| q08_albats | 480p, cigarette: turns at 1.0–3.9, all at offset −3 → on screen | 0 |

Side finding: the source AV offsets are not 0 for some clips: q08 −3, q03 / q07 −2, m03 +2 frames. These come from the camera / upload.

## Beard: rules and calibration

Beard zone:
- inside the jaw outline, eroded off its edge;
- from the lower lip to 3/4 of the way to the chin (the shadowed chin underside made smooth chins look dark).

Features:
- `speck`: share of pixels darker than their own neighbourhood (L* < blur σ4 − 3: hair strands / stubble dots, not smooth shading), divided by the same share on the cheek + forehead skin;
- `dark`: share of pixels with L* < skin − 18.

Rule:
- beard = (speck ≥ 2.0 and native IOD ≥ 50 px) or dark ≥ 0.5;
- the IOD guard exists because at 480p the upscaled texture is compression noise.

| face | label | speck | dark | IOD | → |
|---|---|---|---|---|---|
| old03 | stubble | 2.89 | 0.00 | 89 | latentsync ✓ |
| q01_amir | beard | 13.6 | 1.00 | 78 | latentsync ✓ |
| q03_priest | beard | 2.50 | 0.39 | 114 | latentsync ✓ |
| q04_farikh | grey beard | 4.00 | 0.18 | 81 | latentsync ✓ |
| q05_rozhkov | beard | 16.6 | 0.66 | 84 | latentsync ✓ |
| q09_vertical | beard | 4.15 | 0.86 | 110 | latentsync ✓ |
| m03 | none | 0.00 | 0.16 | 127 | keysync ✓ |
| m18 | none | 0.72 | 0.22 | 125 | keysync ✓ |
| q02_vera | none (wrinkles) | 1.60 | 0.36 | 174 | keysync ✓ |
| raw04 | none | 0.32 | 0.13 | 109 | keysync ✓ |
| q08_albats | none, 480p | 3.16 | 0.08 | **39** | keysync ✓ (guard) |
| q06 / q07 / raw01 / raw02 | unmeasured (dark / small / no landmarks) | — | — | — | keysync default |

Margins:
- The closest pair is q02 at 1.60 (deep wrinkles) vs q03 at 2.50.
- Stubble (old03, 2.89) sits well above the smooth faces (≤ 0.72).
- No false latentsync among the non-beard faces.
- Caveat: raw05 is only 4 frames at IOD 34 and comes out as latentsync on `dark`. It is not checked by eye.
