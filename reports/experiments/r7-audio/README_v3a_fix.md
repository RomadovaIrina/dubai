# r7_audio_v3a-fix: the 05 "fart"

The user heard a "fart" ("пердеж") somewhere in 05 v3a. It is **05 at 0:31.45–0:31.98** in v3a (u4, "I'm just walking around, / filming and saying something.").

In `05_r7_audio_v3a-fix_AB_prev-then-new.mp4` (v3a, then the fix), it plays at **0:31.5** and the fixed version at **1:23.9**.

## What it was

Chatterbox's take for u4 has a 0.45 s voiced hesitation murmur between "around," and "filming":
- F0 70–77 Hz, −20 to −29 dBFS;
- CTC puts "around" at 31.20–31.46 and assigns the murmur to no word;
- Whisper hears nothing there.

`cut_parts` cuts inner pieces at a point in the inter-word gap, so the murmur stayed at the end of piece 1. Placement then cut it off into digital silence at 31.98, which gives a low burp that stops dead.

The dub gate didn't remove it, because it only gates unvoiced frames or frames more than 30 dB under the voice.

## Generic fix (retake only; the r6 takes the user approved are untouched)

1. `tts_align.cut_parts(..., tight_inner=True)`: inner piece edges get the same rule as the outer ones (the first word − 0.06 s, the last word + 0.08 s).
2. `tts_align.hush_gaps`: in a re-take, anything in an inter-word gap ≥ 0.25 s that no CTC word covers is ducked to −30 dB with 20 ms sin² ramps.
   - The duck runs from the word + 0.08 s to the next word's acoustic onset − 0.03 s. The onset is the first frame within 8 dB of that word's level, or within 6 dB of its level above 2.5 kHz.
   - A first version ended the duck at CTC start − 0.06. That left 0.1 s of murmur on "filming", and the mix ASR heard "a few minutes" / "coming in". The onset rule fixed it.

With the murmur gone, lip placement sees a real pause at "around," (a pb1 comma) and places "filming…" at the speaker's next burst (32.58 s).

## Checks (new: `scripts/r7/piece_tails.py`, `scripts/r7/creak.py`)

- **Piece tails / heads** (audible material outside a placed piece's words):
  - 05: 1 → 0.
  - The other five hits are in the original r6 takes and were already there in r7v2: 04 u4 tail at 47.0 s (0.36 s, −22 dB, after "video,"), 03 u1 tail and u8 head, 01 u18 head, 02 u10 tail. They are left as they were (the user approved those takes; 04 is "great").
- **creak.py:** pitch alone does not separate this class; the 05 dub voice sits around 75 Hz on many normal words. It is kept as a diagnostic.
- **Clicks:** no new ones in the final mix (05: 2 → 1).
- **Gate vs v3a:** PASS on all five.

| video | talks, dub silent (s) | pause W1 (s) | source speaks, dub silent (s) | WER |
|---|---|---|---|---|
| 05 | 1.5 → 1.5 | 0.236 → 0.254 | 7.45 → 7.85 | 0.057 → 0.057 |
| 04 | 2.35 → 2.35 | 0.544 → 0.567 | | 0.043 → 0.032 |
| 02 | 0 | | | 0.039 → 0.044 |

- **01 and 03:** sample-identical to r7v2.

## Noted, not chased (per the user)

Pause placement is "not always lucky".

## Build

```
REUSE=r6v1src R7_PREV=r7v2 REMIX_ARGS="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on" bash scripts/r7/r7_audio.sh r7v3a8
bash scripts/r7/r7_publish.sh v3a-fix r7v3a8 r7v3a5
```

The build took 302 s for the five videos, about 138 s/video-min.
