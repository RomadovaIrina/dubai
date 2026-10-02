# Video 03 — analysis of the current result and improvement options (2026-10-02, analysis only, nothing changed)

Source: talking head, 90.33 s, Russian monologue without pauses between sentences (Whisper returns 10 punctuation-less segments cut mid-sentence,
27 Silero speech bursts, 60.7 s of speech). Output `/tmp/pilot_refresh/final/03_final.mp4` (manifest next to it; page `refresh_videos/03.md`).
Pipeline facts: validation PASS; SyncNet **5.02 / offset 0** (original 4.89 / 2; old formal 0.8: 4.65; week-3 burst interim: 5.88; TTS-v2 interim: 5.19);
LatentSync on 1927 / 2260 frames (85 %), 9 segments, 435.5 s; CodeFormer 1927 faces 57.1 s; total 746 s; peak VRAM 25.2 GB.
Intelligibility of the final audio (independent ASR, `scripts/diag/eval_run.py`): Whisper WER 0.11, CTC CER 0.099 unit-mean, **9 of 10 units clean**,
the 10th is the one-word unit "вперед" -> "forward" (see 3). Face quality (`face_metrics.py`): mouth sharpness 0.72-0.79 and upper face 0.77-0.86 of the
master, flicker 0.92-1.10, boundary jumps at the 18 LS/pass-through transitions equal to the master's — no seams, no flicker added
(contact sheets `refresh_videos/03_mouth_*.png`, `/tmp/pilot_refresh/scratch/qm_frames/03_final/`).

## 1. Dominant defect: the dubbed speech covers only 2/3 of the original speech -> closed mouth while the speaker visibly talks

```text
original speech 60.7 s, dubbed speech 48.3 s, overlap 41.5 s  ->  19.2 s of original speech with NO dubbed speech (IoU 0.61; dub_coverage.py)
LatentSync frames driven by silent dubbed audio: 762 / 1927 (40 %); of these 376 frames while the original speaker is audible
   (week-3 burst interim: 141; old slot baseline: 444)   -> this is what moved SyncNet 5.88 -> 5.02
bursts < 50 % covered: 8 of 27; fully uncovered: 27.8-28.9 s, 42.4-44.3 s; 85.2-87.3 s 9 % covered; 51.4-55.8 s 27 %
tts/slot ratio per unit: 0.49 (u4) 0.56 (u7) 0.60 (u8) 0.68-0.72 (u0-u3, u5) 0.92 (u6)   -> the English is spoken 30-50 % faster than the Russian
```
Frames 28.3 s / 43.3 s / 86.2 s in the mouth sheet: original mouth open mid-word, output mouth closed (LatentSync draws a neutral mouth for silence).

Why it got worse than the week-3 interim although the same burst placement is used:
- natural-phrase chunking (min 3 / soft 4 words, `consolidate_parts`) and the duration-aware fit (`merge_parts`) **union neighbouring bursts**
  into one window and put ONE chunk at the start of the union: u4 "nothing would obstruct our way, I hope everything" (2.9 s) sits at 39.3-42.1 s
  of a 39.3-46.7 s union that contains three Russian bursts (39.3-41.1, 42.4-44.3, 46.3-46.7); u5 p1 (4.5 s) at 48.2-52.7 s of 48.2-55.5 s;
  u8 p2 (1.6 s) at 83.9-85.5 s of 83.9-87.1 s; u0 is a single 6 s chunk (Qwen split fell back to `single_fallback`) over five bursts 3.3-11.3 s.
- the atempo cap 1.15 / hard 1.2 and `fill_slowdown 1.0` mean nothing is stretched any more: a chunk shorter than its burst leaves the tail silent.
- Qwen's duration-aware re-translation only SHORTENS over-long parts (ratio > 1.15); there is no step for the opposite, much more common case
  (ratio 0.3-0.7), so the English never grows to the Russian duration.

## 2. Unit segmentation / translation (text layer)

- Whisper segments = translation units and they cut sentences: "...bring us all the desired | the result we wished for so that ... I hope everything | will be good
  because earthly actions with you ..." and "...let's all have success [я] | I want to wish that ...". Each fragment is translated on its own, so the
  English is grammatical per fragment but not as a sentence, and Chatterbox gives each fragment sentence-final prosody.
- ASR text errors propagate: "ибо земные действия" (-> "earthly actions") is almost certainly a mis-hearing; a stray " . " token in segment 3 reaches the
  translation ("let's consider this as . the beginning"); the first unit is lower-case / unpunctuated ("friends hey what's up everyone hello").
- The number "28" is kept (number-retry works), "28 августа" -> "August 28th" fine.

## 3. One-word unit "вперед" -> "forward" (u9, slot 0.42 s)

Chatterbox renders the single word as 2.16 s ("forward forward", "stretched"), TTS QA flags all three attempts (CER 3.57 / 0.43 / 1.00), the best one is
placed 88.56-90.27 s, the timeline ends at 90.33 s, so the second "forward" is cut -> independent ASR hears "Forward, Wadi". Known weak spot
(first_res review: 1-3-word fragments were 7 of 7 residual bad parts); here it is a whole unit, so chunking cannot merge it.

## 4. What is fine on 03

TTS fp16 guard 0 violations, no part above atempo 1.15, no slot fallback, no cut groups, 1 of 2 QA retries fixed a part (u7 "ah" interjection),
CodeFormer restores all 1927 faces (no fallback detection), 0 verify splits, AV offset 0, output geometry/frames/duration preserved.

## 5. Improvement options (ordered by expected effect on 03; NOT implemented — decision pending)

A. Coverage (fixes 1; expected to recover most of the 5.88 -> 5.02 SyncNet drop and the "dead mouth" frames)
   1. Do not leave merged/union bursts with one chunk at the start: when a part spans k source bursts, split the TTS chunk at its own Silero pauses
      (phrase granularity, never inside a word) and distribute the phrases over the k bursts (the E1 placement did this per phrase; the burst path
      dropped it when it merged bursts). Audio-only change in `e1_burst_align.place_groups` / `tts_burst.py` fit step.
   2. Add the missing "fit-up" round to the duration-aware re-translation: a part with ratio < ~0.6 is re-asked from Qwen as a fuller natural
      English sentence with a target word count ≈ burst seconds x 2.3 words/s (meaning preserved, no new information), then re-synthesised and
      QA-checked exactly like the existing shorten round (`tts_burst.py` fit loop; prompt in `concise_rewrite`).
   3. Cheap knob already present: `--fill-slowdown 0.85` (week 3: −20 % silent frames; costs slightly slowed chunks). With ratios of 0.5-0.7 it only
      fills part of the gap, so it is a complement to A1/A2, not a substitute.
   4. Chatterbox pacing instead of atempo: `cfg_weight` 0 gave ~15 % slower speech in the first_res study; a 0.3/0.5 sweep on 03's short parts could
      lengthen chunks naturally (A/B on intelligibility first).
   5. Video-side fallback for bursts that stay uncovered: gate LatentSync by VAD(dubbed) only instead of VAD(original) ∪ VAD(dubbed), so frames where
      only the original speaks keep the original mouth (moving, Russian) instead of a closed LatentSync mouth. One flag-level A/B in
      `build_speech_gate`; needs a human look (which artefact is less bad).

B. Text layer (fixes 2)
   1. Re-segment the Russian transcript into sentences before translation (Qwen punctuation pass over the whisper words, units = sentences or
      sentence halves at ≤ --max-unit-s, word timestamps kept) -> whole-sentence translation and prosody; one extra CPU Qwen call (~30 s).
   2. Translation prompt: spoken-style, punctuated, capitalised output; drop stray punctuation tokens from the ASR text before translating.
   3. ASR: `initial_prompt` with a punctuated Russian sample or `condition_on_previous_text` tuning to reduce mis-hearings — lower expected gain.

C. Tiny units (fixes 3)
   1. Merge units of < 3 words or slot < 1 s into the neighbouring unit before translation ("...the result we desire. Go!") so Chatterbox never gets a
      one-word prompt; and never place a chunk past the end of the timeline (trim at the speech span, or drop the repeat).

D. Lip-sync lever independent of audio: LatentSync guidance 2.0 (E5 probe on 04 seg 1: SyncNet 4.45 -> 5.49, same sharpness/flicker, +4 % time).
   Production-setting change -> only as an explicit A/B on 03 after A/B above.

E. Performance (03 is the slowest video: 746 s, LatentSync 289 s/video-min because 85 % of frames are lip-synced): no audio-side change helps;
   the levers are fewer eligible frames (A5 reduces them as a side effect) or UNet compile/TensorRT (excluded so far).

Suggested first batch for 03: A1 + A2 + C1 (audio only, no production video setting touched), measured with dub_coverage (orig_without_dub_s,
silent-while-speaking frames), eval_run (WER/CER, bad units), SyncNet, and a listen; then A5 and D as separate video-side A/Bs.
