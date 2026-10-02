# Audio candidate comparison — 04.mp4: OLD current vs NEW (2026-10-02)

**Verdict: FAIL** (hard criteria: intelligibility_not_worse, no_systematic_interjections, no_repeated_words, no_cut_words, no_lost_numbers, atempo_hard_le_1_20, orig_without_dub_lt_target, coverage_improved; target original-speech-without-dub < 10 s)

| metric | OLD | NEW |
|---|---:|---:|
| dubbed speech s (VAD) | 23.90 | 26.20 |
| original speech s | 29.50 | 29.50 |
| overlap s | 18.70 | 25.40 |
| original speech WITHOUT dub s | 10.80 | 4.10 |
| dub outside original s | 5.20 | 0.80 |
| VAD IoU | 0.539 | 0.838 |
| bursts < 50 % covered / uncovered (of 15) | 5 / 2 | 0 / 0 |
| silent-while-speaking frames, VAD proxy, whole timeline | 269 | 100 |
| LS frames (reference segments) dub-silent (RMS < -50 dB) | 451 | 384 |
| … of which original speaking (> -35 dB) = pilot metric | 305 | 215 |
| units / tts mode / segmentation / tiny merges | 5 / parts / None / 0 | 6 / phrase / sentence / 0 |
| TTS parts (generations) / attempts / retried (ok) / bad final | 14 / 16 / 1 (1) / 0 | 6 / 9 / 0 (0) / 0 |
| fit-up units accepted / candidates / rejected | - | 3 / 4 / 2 |
| Whisper WER / CTC CER / Whisper CER (unit mean) | 0.024 / 0.059 / 0.009 | 0.040 / 0.053 / 0.017 |
| bad units (CTC CER >= 0.3) / interjection units / final word missing | 0 / 0 / 0 | 0 / 0 / 0 |
| repeated words (ASR hypotheses) | 0 | 0 |
| lost numbers (units) | 0 | 0 |
| atempo max / > 1.15 / > 1.20 / mean (groups) | 1.15 / 0 / 0 / 1.0115 (13) | 1.0 / 0 / 0 / 0.9353 (18) |
| atempo histogram <1 / 1.0 / <=1.05 / <=1.10 / <=1.15 / <=1.20 / >1.20 | 0 / 12 / 0 / 0 / 1 / 0 / 0 | 12 / 6 / 0 / 0 / 0 / 0 / 0 |
| cut groups (> 50 ms) / cut s total / max per span / overflow groups / slot-fallback units | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |

Acceptance: intelligibility_not_worse=FAIL, no_systematic_interjections=PASS, no_repeated_words=PASS, no_cut_words=PASS, no_lost_numbers=PASS, atempo_hard_le_1_20=PASS, atempo_preferred_le_1_15=PASS, orig_without_dub_lt_target=PASS, coverage_improved=PASS, no_qa_bad_parts=PASS

## NEW units

| id | slot | source | final text (fitted*) | spans | tts s | tts/src-speech | fit-up | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|---:|---|
| 0 | 1.60-4.96 | Сделаем видео с очками посмотрим. | Let's make a video with glasses and watch it. | 1 | 2.36 | 1.056 | - | 0 | 2.3-4.3 |
| 1 | 4.96-11.08 | Опять же потестируем как наше приложение будет на это все реагировать. | Once more, we will test how our application will react to all of this.* | 3 | 3.64 | 0.663 | accepted rewrite_rejected | 0 | 5.0-6.0 6.6-7.3 8.4-10.3 |
| 2 | 11.08-29.46 | город сочи 28 августа сегодня и мы начинаем рабочие процессы по созданию нашего магического приложения которое | Sochi city today, August 28, and we start the working processes for creating our magical application which | 4 | 8.08 | 0.822 | - | 0 | 11.1-11.5 14.6-15.6 17.5-19.4 22.0-27.1 |
| 3 | 29.46-42.28 | облегчит жизнь людям которая я надеюсь будет пользоваться популярностью блогеров всего мира и не только блогеров | will make life easier for people whom I hope will be popular with bloggers from all over the world and not just bloggers | 5 | 7.04 | 0.774 | rewrite_rejected | 0 | 29.5-31.1 33.1-35.7 36.4-37.4 38.5-39.7 41.2-42.1 |
| 4 | 42.28-48.20 | Поэтому вот такое вот прекрасное видео. | Therefore, here is a video that is quite wonderful.* | 3 | 2.88 | 0.891 | accepted | 0 | 42.3-42.9 44.5-45.0 46.6-48.2 |
| 5 | 48.20-52.34 | Классная погода что вперед. | The weather is really nice, isn't it? That's great.* | 2 | 2.92 | 1.033 | accepted | 0 | 48.2-49.7 51.6-52.6 |

## OLD units

| id | slot | source | final text | parts | tts s | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|
| 0 | 1.60-10.24 | сделаем видео с очками посмотрим опять же потестируем как наше приложение будет на это | we'll make a video with glasses look at it again test how our application will react to this | 4 | 7.16 | 0 | 2.3-4.0 4.8-6.0 6.4-8.3 8.4-9.8 |
| 1 | 10.24-25.40 | все реагировать город сочи 28 августа сегодня и мы начинаем рабочие процессы по созданию нашего | everyone react the city Sochi August 28 today and we start working processes for creating our | 3 | 7.52 | 0 | 10.2-12.2 14.6-16.6 17.5-20.5 |
| 2 | 25.40-35.38 | магического приложения которое облегчит жизнь людям которая я надеюсь будет пользоваться | magical application which will ease people's lives which I hope will be used | 2 | 4.8 | 0 | 25.4-27.7 29.3-31.6 |
| 3 | 35.38-48.20 | популярностью блогеров всего мира и не только блогеров поэтому вот такое вот прекрасное видео | popularity of bloggers from all over the world and not only bloggers so here's such a wonderful video | 3 | 6.84 | 0 | 35.4-37.9 38.5-40.0 41.2-43.5 |
| 4 | 48.20-52.34 | классная погода что вперед | nice weather let's go ahead | 1 | 1.8 | 0 | 48.2-49.8 |
