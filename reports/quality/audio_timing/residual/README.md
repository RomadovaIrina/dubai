# Residual "speaker talks, dub silent" gaps — list for the video side (Agent C → Agent B), 2026-10-16

One JSON per video and timing policy: `<vid>_<policy>.residual.json` (generator: `scripts/quality/residual_for_video.py`, branch
`experiment/audio-timing`; copies in `reports/quality/audio_timing/residual/`).

| policy | what it is | use it for |
|---|---|---|
| `cand5` | today's dub track (the baseline runs in `/workspace/dub/artifacts/baseline/<id>/`) | the videos you are working on now |
| `L1_0.875` | proposed timing: `--sentence-flow-gap 0.2 --spread pauses --fill-slowdown 0.875` (offline recut of the same TTS) | the expected residual after the timing fix (default proposal) |
| `L1_0.85` | same with floor 0.85 (only if the listening test accepts the slower speech) | upper bound of what timing alone removes |

Per gap: `t0` / `t1` (s), `frames` = master_25fps.mp4 frame indices [f0, f1), `ls_frames` (LatentSync frames inside it, cand5 LS mask),
`class` = `natural` (< 0.35 s: no action needed) or `residual`, `position_in_burst` (head = dub starts late, tail = dub ended while the speaker
continues, inner = between two dub parts, whole = a whole source burst without dub), `source_words` (Whisper words overlapping the gap by
≥ 0.15 s or half the word: Whisper's word times are wide, so this is a hint, not proof), the sentence (source / dub), the dub parts before /
after, the unit's fit-up history, `layer3_could_fill` (a more literal translation could cover it; needs GPU re-synthesis, not done yet).

Summary (residual = gaps ≥ 0.35 s):

| video | cand5: residual n / s / LS frames | L1 + 0.875 | L1 + 0.85 | longest (cand5 → L1 + 0.875) |
|---|---|---|---|---|
| 01 | 8 / 5.3 / 15 (01 has almost no valid face) | 5 / 3.0 / 13 | 3 / 1.9 / 10 | 1.7 → 1.1 s |
| 03 | 12 / 9.4 / 240 | 8 / 5.6 / 143 | 7 / 4.9 / 125 | 2.0 → 1.3 s |
| 04 | 6 / 4.1 / 106 | 5 / 2.6 / 67 | 5 / 2.5 / 64 | 1.0 → 0.6 s |
| 05 | 8 / 6.4 / 164 | 7 / 4.8 / 121 | 7 / 4.7 / 120 | 1.3 → 1.1 s |

Suggested reading for the "silent face" / pass-through options: every residual gap is inside a source speech burst, so the original mouth
is moving there; `tail` gaps are the common case (the English phrase ended, the Russian continues). Gaps of 0.35–0.6 s may be acceptable as
is; the long ones after L1 + 0.875 on 03 (34.6–35.5 s, 66.8–68.1 s) and 05 (8.4–9.5 s, 44.0–44.8 s) are where the video side matters most.
