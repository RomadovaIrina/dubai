# Audio candidate comparison — 05.mp4: OLD current vs NEW (2026-10-02)

**Verdict: PASS** (hard criteria: intelligibility_not_worse, no_systematic_interjections, no_repeated_words, no_cut_words, no_lost_numbers, atempo_hard_le_1_20, orig_without_dub_lt_target, coverage_improved; target original-speech-without-dub < 10 s)

| metric | OLD | NEW |
|---|---:|---:|
| dubbed speech s (VAD) | 20.30 | 24.10 |
| original speech s | 27.90 | 27.90 |
| overlap s | 13.30 | 22.20 |
| original speech WITHOUT dub s | 14.60 | 5.70 |
| dub outside original s | 7.00 | 1.90 |
| VAD IoU | 0.381 | 0.745 |
| bursts < 50 % covered / uncovered (of 17) | 10 / 5 | 2 / 0 |
| silent-while-speaking frames, VAD proxy, whole timeline | 366 | 140 |
| LS frames (reference segments) dub-silent (RMS < -50 dB) | 467 | 362 |
| … of which original speaking (> -35 dB) = pilot metric | 467 | 361 |
| units / tts mode / segmentation / tiny merges | 7 / parts / None / 0 | 7 / phrase / sentence / 0 |
| TTS parts (generations) / attempts / retried (ok) / bad final | 11 / 11 / 0 (0) / 0 | 7 / 16 / 0 (0) / 0 |
| fit-up units accepted / candidates / rejected | - | 6 / 7 / 2 |
| Whisper WER / CTC CER / Whisper CER (unit mean) | 0.039 / 0.221 / 0.019 | 0.020 / 0.162 / 0.008 |
| bad units (CTC CER >= 0.3) / interjection units / final word missing | 2 / 0 / 0 | 1 / 0 / 0 |
| repeated words (ASR hypotheses) | 0 | 0 |
| lost numbers (units) | 0 | 0 |
| atempo max / > 1.15 / > 1.20 / mean (groups) | 1.025 / 0 / 0 / 1.0023 (11) | 1.15 / 0 / 0 / 0.9529 (21) |
| atempo histogram <1 / 1.0 / <=1.05 / <=1.10 / <=1.15 / <=1.20 / >1.20 | 0 / 10 / 1 / 0 / 0 / 0 / 0 | 13 / 7 / 0 / 0 / 1 / 0 / 0 |
| cut groups (> 50 ms) / cut s total / max per span / overflow groups / slot-fallback units | 0 / 0 / 0 / 0 / 0 | 0 / 0.003 / 0.003 / 0 / 0 |

Acceptance: intelligibility_not_worse=PASS, no_systematic_interjections=PASS, no_repeated_words=PASS, no_cut_words=PASS, no_lost_numbers=PASS, atempo_hard_le_1_20=PASS, atempo_preferred_le_1_15=PASS, orig_without_dub_lt_target=PASS, coverage_improved=PASS, no_qa_bad_parts=PASS

## NEW units

| id | slot | source | final text (fitted*) | spans | tts s | tts/src-speech | fit-up | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|---:|---|
| 0 | 1.02-5.84 | Давайте протестируем еще вот такой формат съемки видео. | Let's test another format of video shooting to see how it works.* | 2 | 3.16 | 0.831 | accepted | 0 | 1.8-4.0 4.8-5.8 |
| 1 | 5.84-11.98 | Все -таки современные телефоны уже позволяют это делать. | After all, modern phones already allow this. | 3 | 2.92 | 0.64 | rejected (not closer) | 0 | 5.8-7.0 7.4-8.1 10.9-11.8 |
| 2 | 11.98-21.24 | И такая функция добавляет, конечно же, у нас популярности и возможностей. | And such a function, of course, adds to our popularity and offers potential opportunities.* | 4 | 4.72 | 0.857 | accepted accepted accepted | 0 | 12.0-12.9 13.7-15.6 17.0-17.3 19.6-21.2 |
| 3 | 21.24-28.86 | Вот, хочется просто посмотреть, как справится наше приложение в таком формате. | Here, we just want to see how our application will manage in this format.* | 3 | 4.56 | 0.878 | accepted | 0 | 21.2-21.7 22.9-24.5 26.2-28.8 |
| 4 | 29.78-33.52 | Просто вот иду, снимаю и что -то говорю. | I just come here, go, take off, and start saying something.* | 2 | 3.0 | 0.967 | accepted | 0 | 30.4-32.4 32.9-33.5 |
| 5 | 35.50-41.58 | Так что пускай будет такое видео, лишним оно точно не будет. | So let such a video be made; it certainly won't be unnecessary.* | 3 | 3.84 | 0.971 | accepted | 0 | 36.3-37.3 38.4-39.4 40.2-41.6 |
| 6 | 41.58-48.78 | Тестим полностью все варианты, которые возможно придумать. | We will thoroughly test all variants that can possibly be imagined.* | 4 | 3.4 | 0.68 | accepted rewrite_rejected | 0 | 41.6-41.9 42.8-44.2 46.0-46.8 47.6-48.3 |

## OLD units

| id | slot | source | final text | parts | tts s | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|
| 0 | 1.02-5.84 | Давайте протестируем еще вот такой формат съемки видео | Let's test another such format for video shooting | 1 | 2.84 | 0 | 1.8-4.5 |
| 1 | 5.84-11.98 | Все-таки современные телефоны уже позволяют это делать | After all, modern phones already allow this to be done | 2 | 3.36 | 0 | 5.8-7.9 10.9-11.9 |
| 2 | 11.98-21.24 | И такая функция добавляет, конечно же, у нас популярности и возможностей | And such a function, of course, adds us popularity and opportunities. | 2 | 3.8 | 0 | 12.0-13.7 19.6-21.2 |
| 3 | 21.24-28.86 | Вот, хочется просто посмотреть, как справится наше приложение в таком формате | Here just want to see how our application will handle this format | 2 | 4.2 | 0 | 21.2-22.4 26.2-28.9 |
| 4 | 29.78-33.52 | Просто вот иду, снимаю и что-то говорю | Just here I go, take it off and say something | 1 | 2.76 | 0 | 30.4-33.0 |
| 5 | 35.50-41.58 | Так что пускай будет такое видео, лишним оно точно не будет | So let there be such a video; it will definitely not be多余的 | 1 | 3.08 | 0 | 36.3-39.2 |
| 6 | 41.58-48.78 | Тестим полностью все варианты, которые возможно придумать | We test all possible variants that can be imagined. | 2 | 3.28 | 0 | 41.6-42.9 46.0-47.6 |
