# Audio candidate comparison — 03.mp4: OLD current vs NEW (2026-10-02)

**Verdict: PASS** (hard criteria: intelligibility_not_worse, no_systematic_interjections, no_repeated_words, no_cut_words, no_lost_numbers, atempo_hard_le_1_20, orig_without_dub_lt_target, coverage_improved; target original-speech-without-dub < 10 s)

| metric | OLD | NEW |
|---|---:|---:|
| dubbed speech s (VAD) | 48.30 | 53.00 |
| original speech s | 60.70 | 60.70 |
| overlap s | 41.50 | 52.20 |
| original speech WITHOUT dub s | 19.20 | 8.50 |
| dub outside original s | 6.80 | 0.80 |
| VAD IoU | 0.615 | 0.849 |
| bursts < 50 % covered / uncovered (of 27) | 8 / 2 | 2 / 1 |
| silent-while-speaking frames, VAD proxy, whole timeline | 481 | 210 |
| LS frames (reference segments) dub-silent (RMS < -50 dB) | 761 | 670 |
| … of which original speaking (> -35 dB) = pilot metric | 376 | 202 |
| units / tts mode / segmentation / tiny merges | 10 / parts / None / 0 | 10 / phrase / sentence / 1 |
| TTS parts (generations) / attempts / retried (ok) / bad final | 21 / 26 / 2 (1) / 1 | 10 / 18 / 1 (1) / 0 |
| fit-up units accepted / candidates / rejected | - | 4 / 7 / 4 |
| Whisper WER / CTC CER / Whisper CER (unit mean) | 0.110 / 0.087 / 0.062 | 0.004 / 0.041 / 0.001 |
| bad units (CTC CER >= 0.3) / interjection units / final word missing | 0 / 0 / 0 | 0 / 0 / 0 |
| repeated words (ASR hypotheses) | 0 | 0 |
| lost numbers (units) | 0 | 0 |
| atempo max / > 1.15 / > 1.20 / mean (groups) | 1.15 / 0 / 0 / 1.0152 (19) | 1.15 / 0 / 0 / 0.9444 (32) |
| atempo histogram <1 / 1.0 / <=1.05 / <=1.10 / <=1.15 / <=1.20 / >1.20 | 0 / 17 / 0 / 0 / 2 / 0 / 0 | 24 / 5 / 0 / 1 / 2 / 0 / 0 |
| cut groups (> 50 ms) / cut s total / max per span / overflow groups / slot-fallback units | 0 / 0.005 / 0.005 / 0 / 0 | 0 / 0.012 / 0.012 / 0 / 0 |

Acceptance: intelligibility_not_worse=PASS, no_systematic_interjections=PASS, no_repeated_words=PASS, no_cut_words=PASS, no_lost_numbers=PASS, atempo_hard_le_1_20=PASS, atempo_preferred_le_1_15=PASS, orig_without_dub_lt_target=PASS, coverage_improved=PASS, no_qa_bad_parts=PASS

## NEW units

| id | slot | source | final text (fitted*) | spans | tts s | tts/src-speech | fit-up | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|---:|---|
| 0 | 2.60-5.10 | Друзья ну что всем привет. | Friends, hey everyone, greetings. | 2 | 1.88 | 0.842 | - | 0 | 3.2-4.1 4.2-4.8 |
| 1 | 5.10-11.36 | Запишу первое видео такое ознакомительное целью познакомиться. | I will record the first video as an introductory one to acquaint myself with it.* | 4 | 4.12 | 0.806 | accepted | 0 | 5.1-5.4 6.7-8.1 8.4-9.8 10.3-11.2 |
| 2 | 11.36-18.66 | Наверно больше в общем что хочу сказать рад знакомству с вами рад сотрудничеству. | Probably more in general what I want to say is that I am glad to meet you and glad to cooperate. | 3 | 5.16 | 0.987 | - | 0 | 11.4-12.4 14.3-17.3 17.7-18.6 |
| 3 | 18.66-28.56 | Сегодня пятница 28 августа будем считать что это начало нашего сотрудничества. | Today is Friday, August 28th, and we will consider this the start of our cooperation.* | 4 | 5.44 | 0.768 | accepted rewrite_rejected | 0 | 18.7-20.9 22.3-23.8 23.9-24.6 25.5-26.6 |
| 4 | 30.26-43.94 | Что хочется сказать пусть это сотрудничество будет легким принесет нам всем тот желаемый результат который мы загадали чтобы на пути у нас никаких преград не возникало. | Let this cooperation be easy, bringing us the desired result we wished for, ensuring no obstacles arise on our path, and may it be smooth, facilitating our journey.* | 4 | 10.28 | 0.899 | accepted | 0 | 30.5-35.5 35.9-38.6 39.3-40.5 42.4-43.9 |
| 5 | 43.94-55.54 | Надеюсь все будет хорошо ибо земные действия с вами и в целом с тем продуктом который мы сегодня собираем и делаем. | I hope everything will be well, for the earthly actions with you and in general with the product which we are collecting and making today. | 4 | 6.56 | 0.721 | rewrite_rejected | 0 | 43.9-44.3 46.3-47.6 48.2-50.0 51.4-54.8 |
| 6 | 58.86-68.12 | Дальше отправлю еще несколько видео запишу различных форматах в очках без очков посмотрим как будет работать наша система. | Next, I will send a few more videos recorded in different formats, both with and without glasses, to see how our system will work. | 1 | 6.76 | 0.712 | rewrite_rejected | 0 | 59.2-66.4 |
| 7 | 68.12-71.14 | Ну и в принципе все. | Well, and basically that's it. | 2 | 1.88 | 0.982 | - | 0 | 68.1-68.5 70.0-71.1 |
| 8 | 71.14-76.54 | Отправлю в наш чат так что давайте ребят всем нам успеха. | I'll send it to our chat so let's all have success. | 2 | 3.12 | 0.651 | rejected (not closer) | 0 | 71.1-72.2 73.8-75.9 |
| 9 | 76.54-88.98 | Я хочу пожелать пусть наше взаимодействие будет легким и принесет нам тот результат который мы желаем. Вперед. | I wish to express my hope that our interaction will be smooth and easy, bringing us the result that we desire, moving forward.* | 6 | 7.24 | 0.874 | accepted accepted accepted | 0 | 76.5-78.0 78.8-79.3 79.9-82.2 83.9-84.3 85.2-87.3 88.5-89.0 |

## OLD units

| id | slot | source | final text | parts | tts s | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|
| 0 | 2.60-11.36 | друзья ну что всем привет запишу первое видео такое ознакомительное целью познакомиться | friends hey what's up everyone hello i'll record the first video kind of an introduction to get to know everyone | 1 | 6.04 | 0 | 3.3-9.2 |
| 1 | 11.36-18.66 | наверно больше в общем что хочу сказать рад знакомству с вами рад сотрудничеству | probably more in general what I want to say is glad to meet you glad to cooperate | 2 | 5.24 | 0 | 11.4-12.8 14.3-17.7 |
| 2 | 18.66-28.56 | сегодня пятница 28 августа будем считать что это . начало нашего сотрудничества | today is Friday, August 28th let's consider this as the beginning of our cooperation | 3 | 6.76 | 0 | 18.7-20.7 22.2-23.8 23.9-25.9 |
| 3 | 30.26-36.56 | что хочется сказать пусть это сотрудничество будет легким принесет нам всем тот желаемый | what wants to be said let this cooperation be easy and bring us all the desired | 1 | 4.4 | 0 | 30.5-34.8 |
| 4 | 36.56-46.74 | результат который мы загадали чтобы на пути у нас никаких преград не возникало надеюсь все | the result we wished for so that nothing would obstruct our way, I hope everything | 2 | 4.96 | 0 | 36.6-38.4 39.3-42.1 |
| 5 | 46.74-55.54 | будет хорошо ибо земные действия с вами и в целом с тем продуктом который мы сегодня собираем и делаем | will be good because earthly actions with you and in general with the product which we are collecting and making today | 2 | 6.08 | 0 | 46.7-48.0 48.2-52.7 |
| 6 | 58.86-66.42 | дальше отправлю еще несколько видео запишу различных форматах в очках без очков посмотрим как будет | I will send a few more videos further, record them in different formats with and without glasses, and see how it will be. | 1 | 6.96 | 0 | 59.2-65.9 |
| 7 | 66.42-77.00 | работать наша система ну и в принципе все отправлю в наш чат так что давайте ребят всем нам успеха я | our system works well and anyway I'll send everything to our chat so let's all have success | 3 | 5.88 | 0 | 66.4-68.5 70.1-71.9 73.8-75.2 |
| 8 | 77.00-87.08 | хочу пожелать пусть наше взаимодействие будет легким и принесет нам тот результат который мы желаем | I want to wish that our interaction be smooth and bring us the result we desire | 3 | 6.0 | 0 | 77.0-78.3 78.8-81.3 83.9-85.5 |
| 9 | 88.56-88.98 | вперед | forward | 1 | 2.16 | 1 | 88.6-90.3 |
