# 05: what is weak (r7_audio_v1), with timestamps

05 is a phone video shot while walking outdoors. One speaker, face visible 96 % of the time, and the speech is slow and drawn out.

## 1. Pace: the English is much shorter than the Russian (the main problem)

Even at the tempo floor (atempo 0.95 on every burst), each English phrase is 29–57 % of its Russian burst. The dub finishes early and goes silent while the speaker on screen keeps talking. The face agent sees exactly this as a forced-shut mouth.

| source burst | dub (r7v1) | source still talking after the dub ends | text |
|---|---|---:|---|
| 0:05.84–0:11.98 | 0:05.44–0:10.46 | 1.5 s | Все-таки современные телефоны… → After all, these modern phones… |
| 0:12.00–0:15.80 | 0:11.08–0:12.57 | 3.2 s | И такая функция добавляет, конечно же, у → And such a function, of course, |
| 0:17.00–0:21.22 | 0:16.64–0:19.02 | 2.2 s | нас популярности и возможностей → adds to our popularity and opportunities. |
| 0:21.24–0:24.60 | 0:20.82–0:22.46 | 2.1 s | Вот, хочется просто посмотреть, как → Well, I just want to see how, |
| 0:26.20–0:28.86 | 0:26.34–0:28.05 | 0.8 s | справится наше приложение… → our app handles this format. |
| 0:41.58–0:48.78 | 0:41.44–0:44.64 | 4.1 s | Тестим полностью все варианты… → We are testing all possible variants… |

- **Longest dub gap beyond the source's own pause:** 3.05 s at 0:12.6, with others at 0:22.5, 0:28.1 and 0:33.7.
- **Pause match (W1):** 0.78 s, against 0.24–0.27 on 01–03.

Lip placement cannot split these phrases at their commas. Chatterbox reads them without a quiet point there (the level between words is −2.5 to −16 dB of the voice), and a cut through continuous voice would click.

**Fixed in v2 (generic, cheap):** wider lip-placement limits (max shift 2.0 s, unit gap growth 1.5 s, in-unit growth 0.5 s). The dub is never stretched and its takes are unchanged. Speaker on screen talking while the dub is silent: 7.7 → 4.3 s, source speaking with the dub silent 10.8 → 8.65 s, pause W1 0.78 → 0.67 s, longest gap beyond the source pause 3.05 → 2.6 s.

**Not fixed (needs a new take):** a longer translation for the low-ratio units (dur_translate fit-up), or a slower tempo than 0.95. Either changes the takes the user liked, so it's the user's call.

## 2. Translation and pronunciation (not cheap: needs Qwen and TTS reruns)

- **0:04.8 "съемки видео" → "shall we?"** Padding the duration stage added to fill the burst (faithfulness "added: shall we?").
- **0:21 U3, faithfulness 3:** "Well, I just want to see how," with "just want to see" judged missing.
- **0:30 U4, faithfulness 3:** "снимаю" (I'm filming) became "taking pictures".
- **0:40 "superfluous":** Whisper hears "super close" / "super true" on the dub, so Chatterbox pronounces it unclearly.

## 3. Bed

- **No room tone.** The ambience is outdoor street noise with a −28 dBFS floor and a 7 dB spread, so gap_clean uses non-stationary denoise instead of a room tone.
- **Speech-band bed under the dub:** −19.2 dB relative to the voice, the highest of the five. It's mostly street noise: Silero hears voice in the bed for only 0.3 s.
- **Leaked voice replaced in r7v1:** 0:14.94–0:15.86 and 0:23.76–0:24.00.
- In the gaps, 0 words are heard and leaked voice is 0.02 s.

## 4. Crackle

The final mix has 52 transients against r6v1's 52, so nothing new. The dub gate finds no long noisy TTS pause on 05.
