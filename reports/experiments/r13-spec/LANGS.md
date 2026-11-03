# Languages: local measurements (0.9 / 6.4 / 7.2 / 12.2) and the routing proposal (6.5)

r13, 03.11. Everything is local, no external key needed. Script: `scripts/r13/lang_eval.py`. Raw data: `data/lang/` (`mt_<lang>.json`, `tts_<lang>.json`).

## Setup
- **Languages:** the Chatterbox Multilingual v3 list: 23 languages incl. RU = 22 targets.
  - Licence: Chatterbox MIT (`chatterbox-tts` 0.1.7), Qwen 2.5 14B Apache-2.0. Both already in the pipeline.
- **Source:** 54 Russian sentences from the 5 test videos (production ASR, hallucination filter on), taken round-robin.
  - Translation: 16 sentences per language (4 × 01, 3 × each of 02–05).
  - TTS: the first 8 of them.
- **Translation:** Qwen 2.5 14B Q5_K_M (llama.cpp CUDA) with the production prompt (`SYSTEM_PROMPT` + "Translate from Russian to X").
  - The plain prompt is used, without the English-only duration-aware / drift-guard steps of the accepted EN chain.
- **Translation checks:**
  - back-translation X → RU by the same model + chrF against the source;
  - an LLM judge (the same 14B): adequacy / fluency 1–5;
  - **mixed-language lines:** letters of a script that is neither the target's nor Latin (Russian left untranslated, Chinese / Korean / Arabic leaking in), or a character loop.
- **TTS:** Chatterbox Multilingual v3 with a real speaker reference (old03), then faster-whisper large-v3 forced to the language. CER after normalisation (lower case, no punctuation; zh / ja without spaces).

**Caveats:**
- The judge is the translator itself: optimistic, and it gives English, the accepted language, 4.0 / 5. So compare each language against EN, not against 5.
- chrF of the back-translation is naturally lower for distant scripts (ja 34.6 with a fluency of 4.8).
- The TTS CER includes Qwen's broken text (he) and Whisper's own weakness in that language.
- 16 / 8 sentences per language: enough to separate "works" from "broken", not to rank close languages.

## Results

| lang | chrF back | adequacy | fluency | mixed-language lines | TTS CER mean | median | CER > 0.2 |
|---|---|---|---|---|---|---|---|
| **en** English (accepted) | 55.3 | 4.0 | 4.19 | 0/16 | 0.007 | 0.0 | 0/8 |
| de German | 51.0 | 3.5 | 3.81 | 0/16 | 0.048 | 0.012 | 0/8 |
| es Spanish | 47.8 | 3.75 | 4.56 | 0/16 | 0.009 | 0.0 | 0/8 |
| fr French | 52.0 | 3.62 | 4.19 | 0/16 | 0.009 | 0.008 | 0/8 |
| it Italian | 54.7 | 3.62 | 4.5 | 1/16 | 0.009 | 0.0 | 0/8 |
| pt Portuguese | 50.2 | 4.0 | 4.5 | 2/16 | 0.035 | 0.0 | 1/8 |
| tr Turkish | 44.4 | 3.38 | 4.12 | 3/16 | 0.089 | 0.066 | 1/8 |
| pl Polish | 55.0 | 3.56 | 4.12 | 4/16 | 0.104 | 0.093 | 2/8 |
| nl Dutch | 49.9 | 3.5 | 3.75 | 0/16 | 0.016 | 0.007 | 0/8 |
| sv Swedish | 47.6 | 3.62 | 4.38 | 1/16 | 0.093 | 0.074 | 1/8 |
| da Danish | 51.8 | 3.56 | 4.19 | 0/16 | 0.134 | 0.173 | 3/8 |
| no Norwegian | 51.4 | 3.5 | 3.94 | 2/16 | 0.079 | 0.035 | 1/8 |
| fi Finnish | 43.6 | 2.88 | 3.62 | 1/16 | 0.171 | 0.152 | 2/8 |
| el Greek | 46.3 | 2.94 | 3.12 | 1/16 | 0.163 | 0.064 | 2/8 |
| he Hebrew | 42.2 | 2.81 | 2.5 | 16/16 (+ loops) | 0.735 | 0.739 | 8/8 |
| ar Arabic | 50.1 | 3.31 | 3.5 | 7/16 | 0.206 | 0.152 | 2/8 |
| hi Hindi | 38.5 | 3.0 | 3.56 | 3/16 | 0.090 | 0.076 | 0/8 |
| zh Chinese | 49.3 | 4.12 | 4.75 | 0/16 | 0.013 | 0.0 | 0/8 |
| ja Japanese | 34.6 | 3.94 | 4.81 | 0/16 | 0.038 | 0.0 | 0/8 |
| ko Korean | 39.6 | 3.75 | 4.44 | 8/16 | 0.179 | 0.078 | 3/8 |
| ms Malay | 48.4 | 3.31 | 4.0 | 0/16 | 0.046 | 0.044 | 0/8 |
| sw Swahili | 23.3 | 2.19 | 3.19 | 0/16 | 0.065 | 0.062 | 0/8 |

### What the failures look like (Qwen 14B, plain prompt)
- **tr:** "все реагировать город сочи 28 августа…" → `herkes今天的索契市做出反应8月28日…`: it switched to Chinese mid-sentence.
- **pt:** `agora已经开始，不再是翻译的内容…`: Chinese, then a remark about the translation itself.
- **he:** `давайте נテスト גם את Định dạng צילום الفيديو`: Russian, Japanese, Vietnamese and Arabic in one line. Another line ends in "чччч…" (a loop).
- **el:** an English `Note: The term "застройщик"…` aside appended to the translation.
- **de / es** are clean, for example:
  - de: "Давайте посмотрим, что включает в себя ремонт, предлагаемый застройщиком." → "Lassen Sie uns sehen, was der von dem Bauherren angebotene Reparaturarbeiten enthält." The meaning is right, the grammar is slightly off ("der … Reparaturarbeiten").
  - es: "Vamos a ver qué incluye la reparación que propone el constructor."

## Routing proposal (6.5): which languages need the external path

The automatic rule in `lang_eval.py routes` ("mixed lines ≥ 2, or adequacy < EN − 0.5, or chrF(back) < 0.85 × EN"; donor TTS if "CER > max(0.10, 2 × EN), or > 25 % of sentences with CER > 0.2") is deliberately strict. It flags 15 of 22 for external MT, partly on the noisy chrF (ja). The proposal below is my reading of the table:

| Group | Languages | Translation | Voice | Why |
|---|---|---|---|---|
| **A. local, ready** | en, de, es, fr, it, nl, zh, ja | Qwen 14B | Chatterbox | No mixed lines (it 1/16); judge within 0.5 of EN; CER ≤ 0.05 |
| **B. local + a script guard** | pt, sv, no, ms | Qwen 14B + a mixed-script retry (the EN drift guard extended to all languages) | Chatterbox | 1–2 mixed lines of 16, otherwise like A; CER ≤ 0.09. ms: adequacy 3.3, check with a native speaker |
| **C. external translation** | tr, pl, hi, sw | external API | Chatterbox (pl borderline: CER 0.10) | tr / pl / hi: 3–4 mixed lines of 16; sw: adequacy 2.2, chrF 23 |
| **D. external voice only** | da | Qwen 14B | donor TTS + ChatterboxVC | Translation is clean; Chatterbox CER 0.13, 3/8 bad |
| **E. external translation + voice** | he, ar, ko, el, fi | external API | donor TTS + ChatterboxVC | Mixed-language output (he 16/16, ar 7/16, ko 8/16) and / or low adequacy (el 2.9, fi 2.9) + CER 0.16–0.74 |

**Answer to "is OpenAI needed at all?"**
- **For the 8 main European + zh / ja targets (group A): no.**
- **For pt / sv / no / ms: probably not.** A cheap local fix (a mixed-script retry, like the EN one) is the next step before deciding.
- **For tr / pl / hi / sw: yes for the translation.**
- **For he / ar / ko / el / fi: yes for both translation and voice.**
- **da:** only the voice.

`scripts/r13/lang_routes.json` holds this curated table; the router (`scripts/r13/donor.py`) falls back to the local path, flagged, while there is no key.
