# r7_audio_v2 (WATCH): crackles, 05, on-screen coverage

v2 is built from the same r6_audio_v1 takes, translation and voice as r7_audio_v1. Qwen is still 14B, not rerun.

```
REUSE=r6v1src R7_PREV=r7v1b REMIX_ARGS="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on" bash scripts/r7/r7_audio.sh r7v2
FORCE_WATCH=1 bash scripts/r7/r7_publish.sh v2 r7v2 r7v1b
```

- Audio stage: 174 s for the five videos, i.e. the r7 part. With TTS from scratch it's about 120 s/video-min.
- Outbox: `<id>_r7_audio_v2_WATCH_AB_prev-then-new.mp4` (r7v1, then r7v2). For the user, 04 and 05 matter most.
- Handoff: `handoff/out/F/audio-r7-v2/` (61 MB, READY).

## 1. Crackles (04)

**Cause:** the crackle came from the dub voice, not the bed. Chatterbox left a noisy, crackly pause in the take: 04 at 3.87–5.76 s, between "…with the glasses," and "let's see", at −33 to −45 dBFS and unvoiced. In r6 the bed covered it. r7_audio_v1's gap denoise lowered the bed there by 7 dB (−36.7 → −44.1 dB) and exposed it.

**Fix (generic):** `r3_remix.py --dub-gate on`. Runs ≥ 0.6 s of the dub that are more than 20 dB under the voice's p95 and unvoiced, or more than 30 dB under it, go to −30 dB.
- **Ramps:** 20 ms sin² (equal-power), placed inside the run, so word edges keep full level.
- **What it never touches:**
  - a word's decay into the take's digital silence;
  - short breaths inside a phrase (muting those opened mid-phrase holes, and the r4 count of pauses inside a phrase went 0 → 4 on 04);
  - any frame within 20 dB of the voice level (checked: 0 frames on all five).
- **Gated per video:** 04 at 3.87–5.76, 02 at 100.13–100.73, 03 at 33.16–33.82; nothing on 01 or 05.

Also new in v2:
- lip-placement block joins use 20 ms equal-power fades instead of 5 ms linear;
- room-tone splices in gap clean use an equal-power crossfade instead of a linear one.

**Detector:** `scripts/r7/clicks.py`. It compares a 2 kHz high-pass 1-ms peak and a sample step against the median of the surrounding ±25 ms. A transient counts at ≥ 12 dB above that and ≥ −75 dBFS.

Transients in the final mix (most are the dub's own consonants and are present in every version):

| video | r6v1 (upload) | r6v1rx (local bed, no r7) | r7_audio_v1 | r7_audio_v2 |
|---|---:|---:|---:|---:|
| 01 | 248 | 250 | 251 | 251 |
| 02 | 231 | 239 | 237 | 237 |
| 03 | 245 | 241 | 241 | 235 |
| 04 | 37 | 37 | **58** | **32** |
| 05 | 52 | 52 | 52 | 46 |

On 01 and 02, the excess over r6v1 is already in r6v1rx, which uses the local Spleeter bed with no r7 step at all, so r7 adds nothing there.

**Fixed crackle timestamps on 04** (present in r7v1, gone in r7v2):
- 4.506, 4.792–5.268 (22 transients in the noisy pause);
- 21.968, 21.992, 22.496, 22.539, 22.556;
- 41.111, 41.329, 43.732.

## 2. 05

See [05_notes.md](05_notes.md). The main problem is pace: the English is 29–57 % as long as the drawn-out Russian bursts, so the dub ends early while the face still talks. The other problems are translation and pronunciation slips ("shall we?" padding, "taking pictures" for "снимаю", "superfluous" unclear) and an outdoor bed with no room tone.

The cheap generic fix is wider lip limits. Translation and pronunciation need new takes, so they're left for the user to decide.

## 3. Speaker on screen talks, dub silent

Lip limits are now max shift 2.0 s (was 1.0), growth of a gap between units 1.5 s (was 0.8), and growth inside a unit 0.5 s (was 0.35). Blocks are still only moved, never stretched, and splits only happen at quiet English boundaries.

| video | speaker on screen talks, dub silent (s), v1 → v2 | source speaks, dub silent (s) | pause W1 (s) | longest gap beyond the source pause (s) |
|---|---|---|---|---|
| 01 | 0 → 0 | 6.5 → 6.5 | 0.261 → 0.261 | 0.7 → 0.7 |
| 02 | 0 → 0 | 14.8 → 14.9 | 0.269 → 0.266 | 2.5 → 2.5 |
| 03 | 0 → 0 | 7.6 → 5.7 | 0.242 → 0.306 | 1.45 → 1.15 |
| 04 | 6.4 → **5.25** | 9.45 → 9.2 | 0.835 → **0.692** | 4.05 → 4.25 |
| 05 | 7.7 → **4.3** | 10.8 → 8.65 | 0.777 → **0.667** | 3.05 → **2.6** |

05 at 24–25 s: the dub "Well, I just want to see how," still ends about 2 s before "…посмотреть, как" does. Chatterbox reads that phrase with no quiet point at its commas, so it can't be split without cutting through the voice. Closing that gap needs a longer take.

## Gate vs r7_audio_v1 (WATCH, 4 rows)

Gap junk is unchanged: leaked voice in the gaps is ≤ 0.06 s per video, there are no new bursts, voice-band excess is at or below v1, and 0 words are heard. Inside-phrase pauses do not increase.

| video | failing row | v1 → v2 | why |
|---|---|---|---|
| 03 | speech-band residue under the dub | −26.4 → −24.3 dB | Same bed; the dub now overlaps more of the source speech, so the measured frames shift. |
| 04 | speech-band residue in dub silences | −36.2 → −34.6 dB | Same effect; the gap metrics show no new leaked voice or bursts. |
| 04 | level jump at dub-pause onsets | 10.2 → 16.3 dB | Known artefact: the metric subtracts the speaker's own drop to silence. |
| 05 | level jump at dub-pause onsets | −6.8 → 1.2 dB | Same artefact. |

Pause match (W1) improves on 04 and 05. On 03 it gets worse by 0.06 s, while the longest gap beyond the source pause improves there by 0.3 s.
