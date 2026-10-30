# r7_audio_v3 (WATCH): longer dub takes where the speaker is on screen

v3 is v2 (gap clean, dub gate, lip placement 2.0 / 1.5 / 0.5) plus `scripts/r7/retake.py`. Only the selected units are re-translated and re-taken; every other unit's audio is bit-identical to r6_audio_v1. 01, 02 and 03 come out bit-identical to v2, because no candidate there passed the bar.

```
for id in 01 02 03 04 05: PYTHONPATH=third_party/llama_cuda:$PYTHONPATH python scripts/r7/retake.py --work .../r6v1src/runs/$id/work --out .../r7v3src/runs/$id/work \
    --face .../sep/$id/source_mediapipe.npz --faith .../r7v2/$id.metrics.json --src16 .../asrchk/${id}_16k.wav
REUSE=r7v3src R7_PREV=r7v2 REMIX_ARGS="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on" bash scripts/r7/r7_audio.sh r7v3
FORCE_WATCH=1 bash scripts/r7/r7_publish.sh v3 r7v3 r7v2
```

## How it works (generic, no hand-edited strings)

1. **Selection.** A unit is selected when either:
   - the face-visible source speech in its bursts exceeds its take by ≥ 0.6 s, or
   - the r7v2 faithfulness judge scored it below 4.

   Content the judge listed as added ("shall we?") counts only together with one of those two reasons.
2. **Candidates.** Qwen 14B writes per-burst pieces with word budgets, plus whole sentences with an enforced budget (+15 / 30 / 50 % words, capped at the source time at this voice's rate).
   - The deterministic guards reject candidates that lose numbers or names or add filler. In retake they are narrowed so that "this kind of" and a synonym for a common word are allowed.
   - The same judge (back-translation, meaning 1–5, missing / added) then requires: score ≥ max(4, the old score), nothing added, naturalness ≥ min(4, the old naturalness).
3. **Takes.** Chatterbox makes takes at normal pace (cfg 0.5) and at its slower, more deliberate pace (cfg 0.3), with no atempo stretching.
   - The QA ladder tries 3 seeds, then the named-entity respellings.
   - Every name must be heard by the QA's Whisper ("Sochi" synthesised as "Soki" is rejected).
   - A CTC word alignment cuts the take between words.
4. **Placement.** Pieces go to the source bursts, or to the speaker's own pauses inside them (Silero pauses ≥ 0.4 s), at tempo 1.0.
   - Cost = uncovered face-visible source speech + 0.3 × dub over source silence + 0.8 s per pause that isn't at a pb1 boundary (punctuation / clause start) + 0.5 s per one-word piece.
   - A lone function word is never left as its own piece, and a take never runs into the next sentence.
   - The new take must beat the old one by ≥ 0.3 s. It is level-matched to the old take, which is faded out with 20 ms ramps.

## Changed sentences

| video | unit | RU | old EN (r7v2) | new EN (r7v3; / = a pause at the speaker's pause) | judge |
|---|---:|---|---|---|---|
| 04 | 2 | сегодня и мы начинаем рабочие процессы по созданию нашего магического приложения которое облегчит жизнь людям | today we begin the work processes to create our magical app that will make life easier for people | Today we are / starting the work processes to create our magical app which is going to make life / a lot easier for people. | 5 → 5 |
| 05 | 2 | И такая функция добавляет, конечно же, у нас популярности и возможностей. | And such a function, of course, adds to our popularity and opportunities. | And this kind of feature, / of course, helps to increase our / popularity / and gives us more opportunities. | 5 → 5 |
| 05 | 3 | Вот, хочется просто посмотреть, как справится наше приложение в таком формате. | Well, I just want to see how, our app handles this format. | Well, I just want to see how / our app performs in this format. | 4 → 4 |
| 05 | 4 | Просто вот иду, снимаю и что-то говорю. | I'm just walking, taking pictures, and saying something. | I'm just walking around, / filming and saying something. | 3 → 5 |

## Results vs r7v2

| video | speaker on screen talks, dub silent (s) | source speaks, dub silent (s) | pause W1 (s) | longest excess gap (s) | faithfulness (units < 4) | leaked voice / words in gaps | transients |
|---|---|---|---|---|---|---|---|
| 01 | 0 → 0 | 6.5 → 6.5 | 0.261 → 0.261 | 0.7 → 0.7 | 4.81 (1) → same | same | 251 → 251 |
| 02 | 0 → 0 | 14.9 → 14.9 | 0.266 → 0.266 | 2.5 → 2.5 | 4.55 (2) → same | same | 237 → 237 |
| 03 | 0 → 0 | 5.7 → 5.7 | 0.306 → 0.306 | 1.15 → 1.15 | 4.55 (0) → same | same | 235 → 235 |
| 04 | **5.25 → 2.35** (target ≤ 2) | 9.2 → 6.75 | 0.692 → **0.544** | 4.25 → 4.25 | 5.0 (0) → 5.0 (0) | 0.06 → 0.02 s / 0 | 32 → 35 (r6v1 37) |
| 05 | **4.3 → 1.5** (target ≤ 1.5) | 8.65 → 7.45 | 0.667 → **0.236** | 2.6 → **0.8** | 4.29 (2) → **4.71 (0)** | 0 → 0 s / 0 | 46 → 35 (r6v1 52) |

The v2 crackle fix is kept: the dub gate runs on the new takes too, and 04 and 05 stay at or below r6v1's transient count.

**Speed.** Retake took 130 s for the five videos (12–35 s per video, including the Qwen, Chatterbox and QA loads), which is 19 s/video-min. The full audio stage is about 110.7 (14B TTS stage) + 8 (r7) + 19 ≈ 138 s/video-min, within ~150.

## Open items (WATCH)

- **04: 2.35 s against the ≤ 2 target.** What's left is u1, "Город сочи 28 августа", which the speaker says word by word (11.1, 14.6–15.6, 17.5–18.8 s). The only take that fills those pauses runs 0.5 s into the next sentence's take ("on the twenty-eighth of August"), and two voices would overlap, so it's refused.
- **05: one pause inside an English phrase** ("helps to increase our / popularity"). Following a speaker who pauses mid-sentence on screen means pausing the English there too; each such pause costs 0.8 s in the selection. With the r6 rule made hard (pauses only at punctuation or a clause start), the coverage gain disappears: 04 stays at 5.25 s and 05 at 4.3 s. This needs the user's ear.
- **05: WER of the final mix 0.024 → 0.057.** Whisper misses a soft "And", hears "super close" for "superfluous" (unit 5, unchanged: its judge score is 5, so it isn't selected; the TTS QA accepted the take), and hallucinates "Thank you" on the closing silence.
- **"shall we?" padding (05 u0): kept.** Every candidate without it dropped "ещё", and the judge scored those 3.
- **Gate rows against r7v2:**
  - 05 dub pauses > 1 s: 7 → 10 (the source has 10);
  - 05 bed in pauses: +1 dB;
  - 05 WER and the one inside-phrase pause, both above;
  - 04 speech-band residue in dub silences: the same metric effect as in v2; the gap metrics show 0.02 s of leaked voice and 0 words.
