# r7_audio_v3a.1: the 05 "fart", fixed with v3a's timing kept

**The junk:** 05 at **0:31.45–0:31.98**, the end of the re-taken u4 piece "I'm just walking around,".

Chatterbox's take has a 0.45 s voiced hesitation murmur between "around," and "filming":
- F0 70–77 Hz, −20 to −29 dBFS;
- CTC puts "around" at 31.20–31.5 and assigns the murmur to no word; Whisper hears nothing there.

`cut_parts` cut the pieces in the middle of that gap, so the murmur stayed at the end of the first piece. Placement then cut it off into digital silence at 31.98 s ("filming" follows at 32.9 s).

The dub gate kept it because it only gates unvoiced frames or frames more than 30 dB under the voice.

## Where it plays

- `05_r7_audio_v3a1_AB_prev-then-new.mp4`: v3a at **0:31.5**, v3a.1 at **1:23.9**.
- `05_r7_audio_v3a1_retake_units_v3a-then-v3a1.m4a`: every re-taken unit of 05 as v3a, then v3a.1:
  - u2, 11.3–21.6 s;
  - u3, 22.7–29.0 s;
  - u4, 30.2–34.6 s (the fix).

## Fix (generic, timing-preserving)

New script `scripts/r7/hush_placed.py`, run on v3a's lip-placed dub. Nothing is re-placed, so piece starts, ends and the track length are unchanged and the face render stays valid.

Every placed piece of a re-taken unit is CTC-aligned to its text, and non-word TTS material is ducked to −30 dB with 20 ms sin² ramps:
- **Tail:** from the last word + 0.08 s to the piece end.
- **Head:** before the first word's onset.
- **Gaps inside a piece:** inter-word gaps ≥ 0.25 s, only after punctuation. Ducking inside a phrase made a new inside-phrase pause (05 u3 "our / app"), so that is excluded.

On 05 this ducks only u4's tail, 31.61–31.98 s. The dub differs from v3a only in 31.62–31.98 s, and no unit changes length.

## Gate

- vs v3a: PASS on all five videos.
- 01–04: bit-identical to v3a (`audio-r7-v3a-all5`).

| 05 | v2 | v3a | v3a.1 |
|---|---:|---:|---:|
| clicks in the final mix | 3 | 2 | 2 |
| audible non-word junk at piece edges (`piece_tails.py`) | 1 | 1 (the fart) | **0** |
| leaked voice in gaps (s) / words | 0.14 / 0 | 0.12 / 0 | 0.12 / 0 |
| bed bursts in gaps (s) | 0.56 | 0.68 | 0.68 (unchanged from v3a; v3a has 15 gaps, v2 has 12) |
| talks, dub silent (s) | 4.3 | 1.5 | 1.5 |
| pause W1 (s) / pauses inside a phrase | 0.667 / 0 | 0.236 / 1 | 0.243 / 1 |
| WER / faithfulness | 0.024 / 4.29 | 0.057 / 4.71 | 0.057 / 4.71 |

## Other changes in this commit

- `clicks.py`: with `--tag`, an id given as the first argument is no longer swallowed as a wav path.
- `tts_align.hush_gaps` takes `after=` (only duck gaps after these word indices).

## Superseded

`audio-r7-v3a-fix` (34867eb) fixed the same thing by re-cutting and re-placing. That moved "filming…" on 05 (32.9 → 32.58 s) and changed 04 / 02. v3a.1 replaces it for the face's sake.

## Noted, not chased

Pause placement is "not always lucky" (the user forgave it for now).
