# Audio candidate comparison — 01.MP4: OLD current vs NEW (2026-10-02)

**Verdict: PASS** (hard criteria: intelligibility_not_worse, no_systematic_interjections, no_repeated_words, no_cut_words, no_lost_numbers, atempo_hard_le_1_20, orig_without_dub_lt_target, coverage_improved; target original-speech-without-dub < 10 s)

| metric | OLD | NEW |
|---|---:|---:|
| dubbed speech s (VAD) | 78.40 | 86.40 |
| original speech s | 91.90 | 91.90 |
| overlap s | 75.30 | 84.00 |
| original speech WITHOUT dub s | 16.60 | 7.90 |
| dub outside original s | 3.10 | 2.40 |
| VAD IoU | 0.793 | 0.891 |
| bursts < 50 % covered / uncovered (of 25) | 0 / 0 | 0 / 0 |
| silent-while-speaking frames, VAD proxy, whole timeline | 403 | 191 |
| LS frames (reference segments) dub-silent (RMS < -50 dB) | 51 | 23 |
| … of which original speaking (> -35 dB) = pilot metric | 45 | 21 |
| units / tts mode / segmentation / tiny merges | 29 / parts / None / 0 | 26 / phrase / sentence / 6 |
| TTS parts (generations) / attempts / retried (ok) / bad final | 35 / 42 / 3 (3) / 0 | 26 / 39 / 0 (0) / 0 |
| fit-up units accepted / candidates / rejected | - | 11 / 13 / 4 |
| Whisper WER / CTC CER / Whisper CER (unit mean) | 0.023 / 0.069 / 0.010 | 0.016 / 0.055 / 0.005 |
| bad units (CTC CER >= 0.3) / interjection units / final word missing | 1 / 0 / 0 | 0 / 0 / 0 |
| repeated words (ASR hypotheses) | 1 | 1 |
| lost numbers (units) | 0 | 0 |
| atempo max / > 1.15 / > 1.20 / mean (groups) | 1.2 / 1 / 0 / 1.0329 (32) | 1.15 / 0 / 0 / 1.0011 (40) |
| atempo histogram <1 / 1.0 / <=1.05 / <=1.10 / <=1.15 / <=1.20 / >1.20 | 0 / 23 / 2 / 0 / 6 / 1 / 0 | 14 / 13 / 3 / 1 / 9 / 0 / 0 |
| cut groups (> 50 ms) / cut s total / max per span / overflow groups / slot-fallback units | 0 / 0.046 / 0.016 / 0 / 0 | 0 / 0.051 / 0.01 / 0 / 0 |

Acceptance: intelligibility_not_worse=PASS, no_systematic_interjections=PASS, no_repeated_words=PASS, no_cut_words=PASS, no_lost_numbers=PASS, atempo_hard_le_1_20=PASS, atempo_preferred_le_1_15=PASS, orig_without_dub_lt_target=PASS, coverage_improved=PASS, no_qa_bad_parts=PASS

Repeated words (NEW): [{"unit": null, "decoder": "ctc", "repeated": ["twenty"]}]

## NEW units

| id | slot | source | final text (fitted*) | spans | tts s | tts/src-speech | fit-up | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|---:|---|
| 0 | 0.00-1.42 | Отделка Prime от Neometria.. | Prime finishing from Neometria.. | 1 | 1.84 | 0.925 | - | 0 | 0.0-1.5 |
| 1 | 1.52-5.24 | Давайте посмотрим, что включает в себя ремонт, предлагаемый застройщиком.. | Let's take a look at what the developer's repair work entails.* | 2 | 3.28 | 0.778 | accepted rewrite_rejected | 0 | 1.5-1.8 1.9-4.8 |
| 2 | 5.38-10.08 | Его вы можете заказать на момент покупки квартиры в нашем жилом комплексе Flora.. | You can order it at the time of purchasing an apartment in our Flora residential complex.... | 1 | 5.0 | 1.022 | - | 0 | 5.4-10.0 |
| 3 | 10.24-13.14 | Потолок натяжной, белый, матовый.. | The ceiling is a stretched, white, matte type of ceiling.* | 1 | 3.4 | 0.985 | accepted | 0 | 10.2-13.1 |
| 4 | 13.18-16.56 | В каждой комнате срок службы более 10 лет.. | In each room, the service life is more than 10 years. | 2 | 3.2 | 0.834 | - | 0 | 13.2-14.3 14.4-16.5 |
| 5 | 16.80-20.50 | Стены.. Обои под покраску на флизелиновой основе.. | The walls. Wallpaper on textile-based backing for painting.* | 2 | 3.76 | 1.051 | accepted | 0 | 17.2-18.0 18.1-20.5 |
| 6 | 20.54-26.66 | Рельефная текстура с рисунком вертикальная струна создает необходимую глубину и объем поверхности.. | The relief texture with a vertical stripe pattern creates the necessary depth and volume on the surface.. | 1 | 5.56 | 0.846 | - | 0 | 20.5-26.4 |
| 7 | 26.66-30.52 | Такие обои выдерживают до 10 окрашиваний.. Пол.. | Such wallpapers withstand up to 10 repaintings.. Floor.. | 2 | 3.36 | 0.901 | - | 0 | 26.7-26.9 27.2-30.2 |
| 8 | 30.66-32.88 | Ламинат 33 класса.. | A laminate of 33 class.* | 1 | 2.04 | 0.707 | accepted rewrite_rejected | 0 | 30.7-32.6 |
| 9 | 32.88-34.38 | Эггер с подложкой.. | Egger with a substrate.. | 1 | 1.6 | 0.994 | - | 0 | 33.3-34.4 |
| 10 | 34.44-38.68 | Имеет высокую стебель тепла, влага и звукоизоляции.. | It has a high stem of heat, moisture, and sound insulation.* | 4 | 3.84 | 0.919 | accepted | 0 | 34.4-34.7 34.8-36.1 36.2-37.3 37.8-38.7 |
| 11 | 38.72-43.04 | И рассчитан на 20 -25 лет активного использования.. | And is designed for 20-25 years of active use. | 2 | 4.04 | 0.848 | - | 0 | 38.7-40.6 40.7-42.4 |
| 12 | 43.38-50.24 | Межкомнатные двери.. Лаконичное полотно из МДФ высокой плотности с премиальным финишным покрытием.. | Interior doors... A concise panel made of high-density MDF with a premium finish... | 1 | 4.96 | 0.671 | rewrite_rejected | 0 | 43.6-48.7 |
| 13 | 50.36-56.58 | Ванная.. Матовый широкоформатный керамогранит 600х600 толщиной 9 мм.. | Bathroom... Matte wide-format ceramic tile 600x600 with a thickness of 9 mm... | 2 | 6.84 | 1.154 | - | 0 | 50.8-51.8 51.8-56.6 |
| 14 | 56.66-61.10 | С влагопоглощением менее 0 ,5%.. | With moisture absorption less than 0.5%. | 3 | 3.64 | 0.791 | rejected (not closer) | 0 | 56.7-56.9 57.2-57.6 57.7-60.7 |
| 15 | 61.10-63.66 | А значит достаточно прочный.. | This therefore means it is sufficiently strong.* | 2 | 2.52 | 0.901 | accepted accepted | 0 | 61.1-61.7 61.8-63.5 |
| 16 | 63.70-68.74 | Вся необходимая сантехника, включая унитаз, микролифт и сверхпрочную глазурь.. | All necessary plumbing, including a toilet, microlift, and super-strong glaze.. | 3 | 4.76 | 0.933 | - | 0 | 63.7-63.9 64.2-65.8 66.0-68.7 |
| 17 | 68.88-71.88 | Ванная или душевая зона.. На ваше усмотрение.. | The bathroom or the shower area—whichever you prefer.* | 1 | 3.36 | 0.946 | accepted | 0 | 68.9-71.9 |
| 18 | 71.98-74.90 | Тумба с высокопрочным латунным смесителем.. Балкон.. | Sink with a high-strength brass faucet... Balcony... | 1 | 3.0 | 0.998 | - | 0 | 72.0-74.6 |
| 19 | 75.02-78.22 | Покрытие потолка и стен в зоне остекленного балкона.. | The coverage of the ceiling and the walls in the area of the glazed balcony.* | 1 | 4.48 | 1.243 | accepted | 0 | 74.7-78.3 |
| 20 | 78.36-81.18 | Штукатурка с белой матовой влагостойкой покраской.. | The plaster has a white, matte, water-resistant paint finish.* | 1 | 3.48 | 1.098 | accepted | 0 | 78.4-81.4 |
| 21 | 81.36-85.12 | Такое покрытие устойчиво к водяному пару и высокой влажности.. | This coating resists water vapor and remains stable in high humidity conditions.* | 1 | 4.48 | 1.192 | accepted | 0 | 81.6-85.1 |
| 22 | 85.18-86.38 | При правильном использовании.. | When used correctly, it is.* | 1 | 1.72 | 1.045 | accepted | 0 | 85.2-86.6 |
| 23 | 86.66-89.42 | Влагостойкие краски могут прослужить до 10 лет.. | Water-resistant paints can serve up to 10 years. | 1 | 2.8 | 0.825 | - | 0 | 86.7-89.3 |
| 24 | 89.60-93.68 | Еще больше подробностей и деталей о делке Prime узнаете у менеджеров отдела продаж.. | You will find more details and information about the Prime deal from the sales department managers.... | 1 | 4.8 | 1.158 | - | 0 | 89.9-93.8 |
| 25 | 93.86-96.74 | Приезжайте к нам в шоурум и убедитесь в качестве отделки.. | Come to our showroom and see the quality of the finishing for yourself. | 1 | 3.32 | 0.978 | - | 0 | 93.9-96.8 |

## OLD units

| id | slot | source | final text | parts | tts s | QA bad | placed |
|---:|---|---|---|---:|---:|---:|---|
| 0 | 0.00-1.42 | Отделка Prime от Neometria. | Prime finishing from Neometria. | 1 | 1.96 | 0 | 0.0-1.5 |
| 1 | 1.52-5.24 | Давайте посмотрим, что включает в себя ремонт, предлагаемый застройщиком. | Let's see what the developer's repair includes. | 1 | 2.6 | 0 | 1.5-3.9 |
| 2 | 5.38-10.08 | Его вы можете заказать на момент покупки квартиры в нашем жилом комплексе Flora. | You can order it at the time of purchasing an apartment in our Flora residential complex. | 1 | 5.0 | 0 | 5.4-10.2 |
| 3 | 10.24-13.14 | Потолок натяжной, белый, матовый. | The ceiling is a stretched white matte one. | 1 | 2.32 | 0 | 10.2-12.4 |
| 4 | 13.18-16.56 | В каждой комнате срок службы более 10 лет. | In each room, the service life is more than 10 years. | 1 | 3.2 | 0 | 13.2-16.1 |
| 5 | 16.80-20.50 | Стены. Обои под покраску на флизелиновой основе. | Walls. Wallpaper on a textile basis for painting. | 1 | 3.4 | 0 | 17.2-20.5 |
| 6 | 20.54-26.66 | Рельефная текстура с рисунком вертикальная струна создает необходимую глубину и объем поверхности. | The relief texture with a vertical stripe pattern creates the necessary depth and volume on the surface. | 1 | 5.24 | 0 | 20.5-25.6 |
| 7 | 26.66-30.00 | Такие обои выдерживают до 10 окрашиваний. | Such wallpaper withstands up to 10 repaintings. | 1 | 2.8 | 0 | 26.7-29.3 |
| 8 | 30.12-32.88 | Пол. Ламинат 33 класса. | Floor. Laminate, class 33. | 1 | 3.4 | 0 | 30.4-33.2 |
| 9 | 32.88-34.38 | Эггер с подложкой. | Egger with a substrate. | 1 | 1.64 | 0 | 33.1-34.4 |
| 10 | 34.44-38.68 | Имеет высокую стебель тепла, влага и звукоизоляции. | Has a high stem of heat, moisture, and sound insulation. | 2 | 3.64 | 0 | 34.4-35.9 36.2-38.1 |
| 11 | 38.72-43.04 | И рассчитан на 20-25 лет активного использования. | And it is designed for 20-25 years of active use. | 2 | 4.48 | 0 | 38.7-40.3 40.7-43.3 |
| 12 | 43.38-44.58 | Межкомнатные двери. | Interior doors. | 1 | 1.28 | 0 | 43.6-44.6 |
| 13 | 44.66-50.24 | Лаконичное полотно из МДФ высокой плотности с премиальным финишным покрытием. | A concise panel made of high-density MDF with premium finish coating. | 1 | 4.4 | 0 | 44.7-48.9 |
| 14 | 50.36-51.34 | Ванная. | Bathroom. | 1 | 0.92 | 0 | 50.7-51.4 |
| 15 | 51.42-56.58 | Матовый широкоформатный керамогранит 600х600 толщиной 9 мм. | Matte wide-format ceramic tile 600x600 with a thickness of 9 mm. | 1 | 4.96 | 0 | 51.4-56.2 |
| 16 | 56.66-61.10 | С влагопоглощением менее 0,5%. | With moisture absorption less than 0.5% | 1 | 3.36 | 0 | 56.7-59.9 |
| 17 | 61.10-63.66 | А значит достаточно прочный. | Which means sufficiently strong. | 1 | 2.04 | 0 | 61.1-63.0 |
| 18 | 63.70-68.74 | Вся необходимая сантехника, включая унитаз, микролифт и сверхпрочную глазурь. | All necessary plumbing, including a toilet, microlift and super-strong glaze. | 2 | 4.8 | 0 | 63.7-65.9 66.0-68.3 |
| 19 | 68.88-71.88 | Ванная или душевая зона. На ваше усмотрение. | Bathroom or shower area. Up to you. | 1 | 2.56 | 0 | 68.9-71.1 |
| 20 | 71.98-74.14 | Тумба с высокопрочным латунным смесителем. | Sink with high-strength brass faucet. | 1 | 2.2 | 0 | 72.0-73.9 |
| 21 | 74.24-74.90 | Балкон. | Balcony. | 1 | 1.08 | 0 | 74.2-74.9 |
| 22 | 75.02-78.22 | Покрытие потолка и стен в зоне остекленного балкона. | Coating the ceiling and walls in the area of the glazed balcony. | 1 | 2.84 | 0 | 75.0-77.8 |
| 23 | 78.36-81.18 | Штукатурка с белой матовой влагостойкой покраской. | Plaster with white matte water-resistant paint. | 1 | 2.84 | 0 | 78.4-81.0 |
| 24 | 81.36-85.12 | Такое покрытие устойчиво к водяному пару и высокой влажности. | Such a coating is resistant to water vapor and high humidity. | 1 | 3.28 | 0 | 81.8-85.0 |
| 25 | 85.18-86.38 | При правильном использовании. | With proper use. | 1 | 1.4 | 0 | 85.2-86.4 |
| 26 | 86.66-89.42 | Влагостойкие краски могут прослужить до 10 лет. | Water-resistant paints can serve up to 10 years. | 1 | 2.76 | 0 | 86.7-89.2 |
| 27 | 89.60-93.68 | Еще больше подробностей и деталей о делке Prime узнаете у менеджеров отдела продаж. | You can get more details and information about the Prime deal from the sales department managers. | 1 | 4.48 | 0 | 89.9-93.8 |
| 28 | 93.86-96.74 | Приезжайте к нам в шоурум и убедитесь в качестве отделки. | Come to our showroom and see the quality of the finishing for yourself. | 1 | 3.16 | 0 | 93.9-96.8 |
