# Audio candidate comparison — 02.mp4: OLD current vs NEW (2026-10-02)

**Verdict: FAIL** (hard criteria: intelligibility_not_worse, no_systematic_interjections, no_repeated_words, no_cut_words, no_lost_numbers, atempo_hard_le_1_20, orig_without_dub_lt_target, coverage_improved; target original-speech-without-dub < 10 s)

| metric | OLD | NEW |
|---|---:|---:|
| dubbed speech s (VAD) | 49.70 | 52.60 |
| original speech s | 56.90 | 56.90 |
| overlap s | 33.70 | 48.30 |
| original speech WITHOUT dub s | 23.20 | 8.60 |
| dub outside original s | 16.00 | 4.30 |
| VAD IoU | 0.462 | 0.789 |
| bursts < 50 % covered / uncovered (of 29) | 12 / 8 | 2 / 1 |
| silent-while-speaking frames, VAD proxy, whole timeline | 581 | 212 |
| LS frames (reference segments) dub-silent (RMS < -50 dB) | 0 | 0 |
| … of which original speaking (> -35 dB) = pilot metric | 0 | 0 |
| units / tts mode / segmentation / tiny merges | 12 / parts / None / 0 | 19 / phrase / sentence / 0 |
| TTS parts (generations) / attempts / retried (ok) / bad final | 19 / 24 / 3 (3) / 0 | 19 / 33 / 0 (0) / 0 |
| fit-up units accepted / candidates / rejected | - | 7 / 10 / 5 |
| Whisper WER / CTC CER / Whisper CER (unit mean) | - / - / - | - / - / - |
| bad units (CTC CER >= 0.3) / interjection units / final word missing | None / None / None | None / None / None |
| repeated words (ASR hypotheses) | - | - |
| lost numbers (units) | 0 | 0 |
| atempo max / > 1.15 / > 1.20 / mean (groups) | 1.2 / 1 / 0 / 1.0278 (18) | 1.2 / 1 / 0 / 0.9903 (41) |
| atempo histogram <1 / 1.0 / <=1.05 / <=1.10 / <=1.15 / <=1.20 / >1.20 | 0 / 15 / 0 / 0 / 2 / 1 / 0 | 18 / 12 / 2 / 1 / 7 / 1 / 0 |
| cut groups (> 50 ms) / cut s total / max per span / overflow groups / slot-fallback units | 1 / 0.261 / 0.252 / 1 / 0 | 1 / 0.471 / 0.397 / 1 / 0 |

Acceptance: no_repeated_words=PASS, no_cut_words=FAIL, no_lost_numbers=PASS, atempo_hard_le_1_20=PASS, atempo_preferred_le_1_15=FAIL, orig_without_dub_lt_target=PASS, coverage_improved=PASS, no_qa_bad_parts=PASS

## NEW units

| id | slot | source | final text (fitted*) | spans | tts s | tts/src-speech | fit-up | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|---:|---|
| 0 | 0.00-7.38 | Также на территории у нас банный комплекс все готово к первым парением. | Also on our premises, there is a bath complex that is ready for the first steam room sessions. | 3 | 5.04 | 1.123 | - | 0 | 1.0-3.9 5.1-6.2 6.7-7.4 |
| 1 | 7.38-12.36 | Через 3 -4 дня буквально уже тут начнутся парения. | Within 3-4 days, the floating will start right here. | 2 | 5.24 | 1.175 | - | 0 | 7.4-11.0 11.4-12.3 |
| 2 | 12.36-15.08 | Сейчас покажу бассейн. | Right now, I am going to demonstrate the swimming pool.* | 2 | 2.8 | 1.314 | accepted | 0 | 12.4-13.1 13.1-15.0 |
| 3 | 15.08-26.36 | Вот такие вот у нас тут разные штуки зона для переодевания бочки шезлонги. | Here are our various things here zone for changing clothes barrel chaise longues. | 4 | 5.4 | 0.797 | rejected (not closer) | 0 | 15.1-15.6 16.4-18.2 19.0-21.2 24.0-24.8 |
| 4 | 26.36-30.42 | Здесь у нас финская сауна. | In this place, we have a Finnish sauna.* | 2 | 2.4 | 0.829 | accepted | 0 | 26.4-27.3 28.7-29.8 |
| 5 | 30.42-41.52 | Покажу русскую баню. | I will show the Russian sauna. | 2 | 2.52 | 1.353 | - | 0 | 30.4-31.7 40.3-41.3 |
| 6 | 41.52-45.72 | Русская баня как положено на дровах печка. | The Russian bath should have a stove burning wood. | 2 | 2.32 | 0.631 | rewrite_rejected | 0 | 41.5-41.7 43.2-45.1 |
| 7 | 45.72-47.96 | Так все выглядит. | So it all looks like this. | 2 | 1.84 | 1.318 | - | 0 | 45.7-46.2 47.1-47.9 |
| 8 | 47.96-51.52 | Какой запах здесь хороший. | What is such a nice smell here?* | 3 | 1.84 | 0.758 | accepted rejected (not closer) | 0 | 48.0-48.3 49.6-50.6 51.0-51.3 |
| 9 | 51.52-70.38 | Есть еще у нас второй этаж это также все входит в инфраструктуру комплекса. | There is also our second floor, this also all falls into the infrastructure of the complex. | 2 | 5.52 | 0.913 | - | 0 | 57.9-60.2 65.5-68.4 |
| 10 | 70.38-74.44 | Конечно за банный комплекс за эти процедуры все нужно будет платить. | Of course, for the bath complex for these procedures, you will need to pay. | 1 | 4.72 | 1.069 | - | 0 | 70.4-74.4 |
| 11 | 74.44-79.00 | Ну это точно вас не разорит. | Well, this definitely won't break the bank. | 2 | 2.24 | 0.996 | - | 0 | 74.4-74.7 77.6-79.0 |
| 12 | 79.00-83.44 | То есть все мы хотим в бане все мы платим за это деньги. | That is, we all want everything in the sauna, and we all pay for it with money. | 2 | 4.56 | 1.242 | - | 0 | 79.0-79.9 80.6-83.4 |
| 13 | 83.44-85.80 | И хамам также здесь есть. | And the hamam is also available here.* | 2 | 2.8 | 1.172 | accepted | 0 | 83.4-84.0 84.2-85.8 |
| 14 | 85.80-94.90 | Массажная комната холодильничек. | The massage room has a small refrigerator.* | 3 | 2.52 | 0.934 | accepted accepted | 0 | 85.8-85.9 88.2-89.5 94.2-94.9 |
| 15 | 94.90-101.20 | И вот здесь такое место для распития чая или минералочки. | And here is a place where one can drink tea or mineral water.* | 2 | 3.92 | 0.762 | accepted rejected (not closer) | 0 | 94.9-96.9 99.0-100.9 |
| 16 | 101.20-103.04 | Зона отдыха также лежачки. | Relaxation zone: lying areas.* | 1 | 4.44 | 1.46 | - | 0 | 101.0-103.0 |
| 17 | 103.04-106.76 | Вот такая вот у нас интересная баня сауна. | Here is a rather interesting sauna we have.* | 2 | 2.84 | 0.866 | accepted | 0 | 103.0-104.9 106.0-106.7 |
| 18 | 106.76-109.56 | Хамам ну и все это пространство. | Hamam and all this space. | 2 | 1.96 | 0.738 | rewrite_rejected | 0 | 106.8-107.5 108.3-109.2 |

## OLD units

| id | slot | source | final text | parts | tts s | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|
| 0 | 0.00-9.36 | также на территории у нас банный комплекс все готово к первым парением через 3-4 дня буквально | Also on our premises, we have a bath complex ready for the first steam bath in just 3-4 days literally. | 3 | 8.28 | 0 | 1.0-4.1 4.8-6.7 6.7-9.0 |
| 1 | 9.36-20.94 | уже тут начнутся парения сейчас покажу бассейн вот такие вот у нас тут разные штуки зона для | already there will be floatations now I'll show the pool it's something like this we have different things here zone for | 1 | 5.4 | 0 | 9.4-14.6 |
| 2 | 20.94-29.32 | переодевания бочки шезлонги здесь у нас | barrel costumes loungers here we have | 1 | 3.68 | 0 | 20.9-24.3 |
| 3 | 29.32-43.60 | финская сауна покажу русскую баню русская | Finnish sauna show Russian banya, Russian | 1 | 4.04 | 0 | 29.3-33.2 |
| 4 | 43.60-50.26 | баня как положено на дровах печка так все выглядит какой запах здесь | banya as it should be on wood stove so everything looks how does the smell here | 2 | 5.36 | 0 | 43.6-46.3 47.3-49.5 |
| 5 | 50.94-70.66 | хороший есть еще у нас второй этаж это также все входит в инфраструктуру комплекса конечно | good there is also another one for us on the second floor this also all includes the infrastructure of the complex of course | 1 | 6.48 | 0 | 51.0-57.3 |
| 6 | 70.66-79.28 | за банный комплекс за эти процедуры все нужно будет платить ну это точно вас не разорит то есть | for the bath complex for these procedures you will have to pay, well this definitely won't bankrupt you that is | 2 | 7.08 | 0 | 70.7-75.0 77.6-79.8 |
| 7 | 79.98-80.92 | все мы хотим | we all want | 1 | 1.4 | 0 | 80.3-80.9 |
| 8 | 80.94-89.26 | в бане все мы платим за это деньги и хамам также здесь есть массажная комната | In the sauna, we all pay for this money, and the hammam also has a massage room here. | 3 | 6.36 | 0 | 80.9-84.0 84.3-85.7 88.2-89.4 |
| 9 | 94.16-96.76 | холодильничек и вот здесь такое место для | refrigerator and here is a place for | 1 | 5.68 | 0 | 94.2-98.9 |
| 10 | 98.78-106.76 | распития чая или минералочки зона отдыха также лежачки вот такая вот у нас интересная баня сауна | zone of tea or mineral water drinking, also some lying areas. This is how our interesting bath sauna looks. | 1 | 6.36 | 0 | 99.0-105.1 |
| 11 | 106.76-109.56 | хамам ну и все это пространство | hamam and all this space | 1 | 2.28 | 0 | 106.8-108.8 |
