# Phase C — WhisperX forced alignment vs faster-whisper on the synthesized TTS (2026-09-29)

B2B requirement: WhisperX either in the working path or a measured A/B showing it adds nothing over the simpler solution.
This is the A/B, on the chunks of the chosen burst-aware path (`burst_s`, 04: 19 chunks / 70 words, 05: 21 chunks / 72 words).
Script `scripts/quality/phaseC_whisperx_ab.py`; raw `/tmp/dabai_quality/manifests/0{4,5}_phaseC_whisperx.json`
(copied to `result_videos/week3_audio/phaseC_whisperx/`).

## Install (deviation from the frozen environment, recorded)
`whisperx==3.4.2` declares `ctranslate2<4.5`, `numpy>=2`, `onnxruntime`; 3.8.x declares `torch~=2.8`, `pyannote-audio>=4`, `numpy>=2.1`.
Both conflict with the immutable baseline (torch 2.7.1+cu128, ctranslate2 4.8.2, numpy 1.26.4, pyannote 3.4.0), so WhisperX was
installed with `uv pip install --no-deps whisperx==3.4.2 nltk` (+ nltk `punkt`/`punkt_tab` data). Lock diff: `+ nltk==3.10.3`,
`+ whisperx==3.4.2`, nothing else changed (verified by freeze diff). Only `whisperx.alignment` is used (wav2vec2 `WAV2VEC2_ASR_BASE_960H`
via torchaudio, 360 MB download); WhisperX's own ASR / diarization stack is not used, the project keeps faster-whisper 1.2.1 + pyannote 3.4.

## What was compared, per TTS chunk (known text)
- **FW**: faster-whisper large-v3 transcription with `word_timestamps=True` (the pipeline's ASR call) on the 1-3 s synthetic chunk;
- **WX**: `whisperx.align()` forced on the known part text -> per-word start/end;
- **VAD**: Silero VAD speech span of the chunk = what the placement uses today to trim it (-40 ms / +60 ms margins).

| | 04 | 05 |
|---|---|---|
| chunks / words | 19 / 70 | 21 / 72 |
| FW transcript word count == text | **5 / 19** | **1 / 21** |
| WX word count == text (forced) | 19 / 19 | 21 / 21 |
| chunk START: WX vs VAD, mean / p95 abs | 69.5 / 170 ms | 55 / 160 ms |
| chunk START: FW vs VAD | 69.5 / 170 ms | 55 / 160 ms |
| chunk START: WX vs FW | 0 / 0 ms | 0 / 0 ms |
| chunk END: WX vs VAD, mean / p95 abs | 161 / 368 ms | 134 / 281 ms |
| chunk END: FW vs VAD | 582 / 866 ms | 470 / 800 ms |
| interior word boundaries WX vs FW (chunks with equal word count) | 216 ms mean (5 chunks) | 298 ms (1 chunk) |
| per-chunk cost | FW 0.71 s, WX 0.11 s, VAD 0.12 s | FW 0.63 s, WX 0.10 s |

Reading:
1. **faster-whisper cannot give word timestamps of the TTS**: on short synthetic chunks it hallucinates the transcript
   ("we'll make a video with glasses" -> "Welcome to a new video of the classes", "August 28 today and" -> "Focus 28, thank you"),
   the word count matches the text in 5/19 and 1/21 chunks, and its word ends are 0.5 s early on average. Any word-level operation on
   the TTS (splitting a part at a word boundary, checking cut words) needs forced alignment of the known text, i.e. WhisperX.
2. **For what the chosen path does — one whole part per burst, trimmed to its speech span — WhisperX changes nothing**: its first word
   always starts at 0.0 (leading silence absorbed) and its last word ends at the chunk end (trailing silence absorbed), so trimming by
   WhisperX boundaries would be worse than Silero VAD (chunk start would move by 55-70 ms on 16-17 of ~20 chunks, end by 130-160 ms),
   and the placement on the timeline is driven by the SOURCE bursts (Silero VAD + whisper word starts on the original), which
   WhisperX does not improve. No candidate video was built with WhisperX trimming for that reason: the placement-level numbers
   already show it would move chunk starts away from the speech onset.
3. **Where WhisperX is right and was adopted**: verifying "no cut words". Re-running the Phase A E1 cut check with WhisperX forced
   alignment (`e1_cut_words.py --aligner wx`) confirms the E1 verdict with correct word boundaries: 7/14 valley cuts inside words on 04
   ('test', 'application', 'Sochi', 'for', 'ease', 'not', 'wonderful') and 10/14 on 05 — faster-whisper had reported 11/14 and 4/14
   on wrong transcripts. The burst-aware path has no intra-part cuts by construction, so it passes this check trivially.

## Verdict
WhisperX is in the repository as the measurement tool for TTS word boundaries (`scripts/quality/phaseC_whisperx_ab.py`,
`e1_cut_words.py --aligner wx`) and would be the required component if a future variant splits a translation part across two bursts
at a word boundary. It is **not in the production audio path**: the measured A/B shows no improvement in chunk placement over
Silero VAD + source-side whisper word starts, and the residual errors of the burst-aware path (chunk duration vs burst duration,
semantic quality of the Qwen cut) are not word-boundary errors.
