# Agent C round 3: voice / audio (`experiment/r3-audio`)

Base: `experiment/r2-audio` @ a1fd764. Round-2 log: `reports/quality/audio_timing/r2/CHANGES.md`. Server A, shared GPU (≤ 10 GB VRAM for C).

## 1. Device-agnostic audio stage
- `pilot_common.dev()` (`DUB_DEVICE`, otherwise cuda if visible, otherwise cpu). It replaces the hard-coded `"cuda"` in the runner, `tts_burst` and `tts_qa`.
- On the CPU: Chatterbox runs fp32 only and Whisper uses int8.
- Why: the brief first said CPU only. Since the 19.10 coordinator update, the GPU is used and the CPU path is the fallback.
- Qwen stays on the CPU: the shared llama-cpp-python is a CPU build, and this keeps the 8 GB of weights off A's GPU.
- `scripts/quality/r3_audio_run.sh <id>`: audio-only run with the r2 defaults. The Spleeter venv is in `/workspace/dub/c/venv-spleeter`.

## 2. audio-r2 regenerated for 01–05 (19.10), handed off to A / B / D (02 as `audio-r2-02`: 8 / 4.3 s, r2 12 / 7.0)
- `r2_deliver.py` reuses the runner's own Spleeter bed and mix when the flags match, so there is no second Spleeter pass.
- The face reference (`face_ref.py`) now comes from the r2 dub itself. In r2 it came from a cand5 run.
- Residual gaps (n / s / LS frames) vs the r2 delivery before refit:
  - 03: 7 / 4.6 / 118 (r2 8 / 5.6 / 143)
  - 05: 4 / 2.2 / 43 (r2 5 / 3.5 / 58)
  - 04: 5 / 2.6 / 67 (unchanged)
  - 01: 7 / 3.8 / 13 (r2 5 / 3.2 / 13)
- Not yet re-applied: the layer-3 refit (01 u23, 04 u05).
- Time: 478–632 s per video-minute, with 4 videos run in parallel on a cgroup quota of about 30 CPUs.
  - Spleeter: 160–404 s per video (r2: 5 s). The first `separate()` call takes about 80 s, and the CPU is contended.
  - Qwen on CPU: segmentation about 120 s.
  - Without Spleeter: 283–389 s/min. Speed work is next.

## 3. Spleeter 2stems PyTorch port on the GPU (`scripts/pilot/spleeter_torch.py`), default bed backend
- **Why:** the TF/librosa path took 160–468 s per video. It ran on about 1 core of a saturated cgroup quota of about 31 CPUs, mostly used by other jobs.
- **Port:** U-Net, ratio masks and STFT/iSTFT, with the same weights converted to npz by `setup_spleeter.sh` (MIT release asset).
- **Fidelity:** SNR vs TF 68 / 82 dB (bed / vocals) on 04 0–20 s, and 119 dB for the bed on the full 03.
- **Speed:** 1.2 s per video (was 404 s on 03).
- `DUB_SPLEETER_BACKEND=tf` keeps the old path.

## 4. Qwen on the GPU in my own CUDA build; VRAM fitting
- **Build:** llama-cpp-python 0.3.35, CUDA 13.2, sm_120, built into `/workspace/dub/c/pylib-llama-cuda` and put first on PYTHONPATH. The shared CPU build is untouched.
- **Measured VRAM (GB):**

  | component | GB |
  |---|---:|
  | CUDA context | 0.6 |
  | Whisper large-v3 int8_f16 | 1.8 |
  | Whisper large-v3 fp16 | 3.6 |
  | wav2vec2 | 0.4 |
  | Chatterbox fp16 | 1.7 |
  | Qwen Q8_0 | 7.7 |
  | Qwen Q5_K_M | about 5 |

- **New flags:**
  - `--qwen-quant q8_0|q5_k_m`;
  - `--qa-whisper-compute float16|int8_float16`.
- **Memory handling in the TTS stage:**
  - Chatterbox is cast to fp16 at load and the cache is released (-1.4 GB reserved);
  - on CUDA OOM, the same call backs off and retries with its RNG state restored;
  - `expandable_segments` is on.
- **Result on 03, Q5_K_M + int8 QA:**
  - total 180.6 s = **120 s/video-min** (was 632);
  - segmentation 76 → 7.8 s, translation 55 → 5.6 s, TTS 345 → 103 s;
  - peak VRAM 10.6 GB.
- **Problem:** Q5 drifts. On 03 it produced "…will proceed顺利地" (CJK) in u05 and "[it better]" in u01; Q8 has neither. Q5 is not adopted. A translation-output guard is needed anyway.

## 5. Intelligibility guard (`scripts/quality/intel_guard.py`) + speakable text (`scripts/pilot/tts_text.py`, `--tts-text-norm`)
- **audio-r2 (Whisper WER / CTC CER):**

  | video | WER | CTC CER |
  |---|---:|---:|
  | 01 | 0.028 | 0.057 |
  | 02 | 0.060 | 0.164 |
  | 03 | 0.008 | 0.038 |
  | 04 | 0.009 | 0.037 |
  | 05 | 0.011 | 0.149 |

- The worst 20 are in `/workspace/dub/c/outbox/listen/audio-r2_worst*.wav`, with a README of expected vs heard text.
- **Causes:**
  - digits and units ("600x600 … 9 mm" heard as "600 x 67");
  - rare names lose their onset ("Hamam" → "Tamam");
  - slowed units (×0.875) dominate the CTC-only misses.
- **`speakable()`:** numbers, ordinals, years, decimals, N×N, unit symbols, currencies and ".." become spoken words.
  - It runs before TTS, QA, alignment and the part split. The spoken text is kept as `translation_fitted` (recut / refit / eval stay consistent).
  - Opt-in for now; it goes into the r3 policy after the A/B.

## 6. Recipe: CUDA llama-cpp-python for Qwen on the RTX 5090 (for D and the coordinator)
- **Box:** nvcc 13.2.86 at `/usr/local/cuda`, gcc 13.3, cmake 3.28, driver 595.84, sm_120. The shared `/venv/dabai` has the CPU build of llama-cpp-python 0.3.35.
- **Build** (2 min 49 s, 399 MB, into a side dir so the shared env is untouched):
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=off" CMAKE_BUILD_PARALLEL_LEVEL=24 \
  CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH \
  uv pip install --python /venv/dabai/bin/python --target /workspace/dub/c/pylib-llama-cuda \
     --no-deps --no-cache --no-binary llama-cpp-python "llama-cpp-python==0.3.35"
  ```
  - cmake maps `120` to `120a`.
  - `--no-deps` keeps numpy and jinja2 from the shared env.
- **Use:** `PYTHONPATH=/workspace/dub/c/pylib-llama-cuda python …`. Then `llama_cpp.llama_supports_gpu_offload()` is True; runner flag `--qwen-gpu-layers -1` (`r3_audio_run.sh` does both).
- **Measured on the shared GPU** (`qwen_gpu_check.py`; replays the stored ASR of the Q8 CPU audio-r2 runs):
  - Q8_0, all layers: load 2.4–2.8 s, peak 8.34 GB.
  - Segmentation 03 76 → 6.9 s, 01 121 → 14.2 s, 04 11.7 → 4.5 s.
  - Translation 03 55 → 4.8 s, 01 31 → 8.2 s, 04 29 → 1.5 s.
  - Segmentation chunk outputs are **identical** to CPU on 01 / 03 / 04.
  - Translations identical on 8/10, 23/25 and 6/6 units. The rest differ in wording only (float reduction order, as the pilot saw between CPU thread counts).
  - Partial offload, 14 layers: 4.7 GB, segmentation 17.7 s / translation 15.1 s on 03, segmentation still identical.
- **Not used:** Q5_K_M (drift on 03, entry 4).

## 7. Qwen drift guard + TTS-session reload (`--qwen-tts-gpu-layers`)
- **Drift guard:** an English translation with foreign-script characters or a bracketed aside (`tts_text.script_defects`) gets one strict retry. If that also fails, the text is stripped. Qwen rewrites with such defects are rejected.
- **Reload:** Qwen Q8 runs on all GPU layers for segmentation and translation (nothing else is loaded then). It is reloaded with 14 layers for the TTS session, where the fit rewrites run next to Chatterbox and QA, so the audio stage stays ≤ ~10 GB.

## 8. Fixes from the first audio-r3 attempt (03)
- **Qwen reload:** `llm.close()` before reloading with fewer layers. The `tr` tuple still referenced the model, so both copies sat on the GPU: 01 hit 15.5 GB, and on 03 the reload failed.
- **Diarization order:** diarization now runs before Qwen is loaded (pyannote 2.8 GB + Q8 8.3 GB had made an 11.2 GB spike). The output is unchanged.
- **Text normaliser + Qwen:** with the normaliser on, fit-up rewrites got the spoken text ("twenty-eighth"). The number check then compared it with the source "28", and Qwen appended "…cooperation, August twenty-eight."
  - Now Qwen and its number checks see the written text. Synthesis, QA and alignment see the spoken form (`_spk` inside `gen_sentence` / `align`).
  - The final unit text is converted once, before the part split.
- **First 03 r3a numbers** (before these fixes): 231 s = 153 s/video-min; TTS-stage VRAM 8.4–8.9 GB; WER 0.016 (audio-r2: 0.008, CPU eval).
- **Batch:** `scripts/quality/r3a_batch.sh` runs the audio-r3 candidate one video at a time.

## 9. Licences of everything new in r3 (user rule: the licence of the files decides)

| file / package | where | licence | note |
|---|---|---|---|
| Spleeter 2stems weights (`2stems.tar.gz`, deezer/spleeter release v1.4.0) | `models/spleeter/pretrained_models/2stems/` | MIT (repo release asset) | sha256-checked by `setup_spleeter.sh`; same file as r2 |
| `2stems_tf_vars.npz` | `models/spleeter/` | MIT (same weights) | **converted, not retrained**: the TF checkpoint tensors saved as numpy (`spleeter_torch.py convert`), bit-identical values. Not committed |
| `spleeter_torch.py` | repo | ours | re-implementation of the MIT Spleeter inference graph. Bed SNR vs TF 118–119 dB on 03; listen A/B in `/workspace/dub/c/outbox/listen/bed_spleeter_AB_video03_*` |
| spleeter 2.3.2 + tensorflow 2.9.3 (TF venv, now only for `convert` and the `tf` fallback) | `/workspace/dub/c/venv-spleeter` | MIT / Apache-2.0 | |
| llama-cpp-python 0.3.35 CUDA build (ggml / llama.cpp) | `/workspace/dub/c/pylib-llama-cuda` | MIT | links the system CUDA 13.2 runtime / cuBLAS dynamically (NVIDIA CUDA EULA; nothing is redistributed) |
| Qwen2.5-7B-Instruct GGUF Q5_K_M | `/workspace/dub/c/models/qwen2.5-7b-instruct-gguf-q5` | Apache-2.0 (LICENSE in the folder, same official repo as Q8_0) | tested, **not adopted** (drift) |

- HDEMUCS: there are no weights on this server. A `find /` turns up only torchaudio's model *code* (`_hdemucs.py`, BSD).
- `--bed-separator hdemucs` stays refused without `DUB_ALLOW_NC_BED`, which is never set.

## 10. audio-r3 delivered (01–05) to A / B / D + tables
- **Flags:** r2 policy + `--qwen-quant q8_0 --qwen-gpu-layers -1 --qwen-tts-gpu-layers 14 --qa-whisper-compute int8_float16 --tts-text-norm on` (`r3a_batch.sh`), then `r2_deliver.py --tag r3`.
- **Report:** `reports/experiments/r3-audio/README.md` (WER / CER, sounds, residual gaps, LUFS, s/video-min, VRAM). Detail tables: `intel_audio-r{2,3}.md`, `sounds_audio-r{2,3}.md`, `listen_README.md` (worst-20 lists).
- **WER r2 → r3:**

  | video | WER | note |
  |---|---|---|
  | 01 | 0.028 → 0.017 | number sentences fixed |
  | 02 | 0.060 → 0.075 | "Hamam" heard as "Our mom" |
  | 03 | 0.008 → 0.010 | |
  | 04 | 0.009 | unchanged |
  | 05 | 0.011 | unchanged |

- Speed 151–227 s per video-minute (was 478–632). Peak VRAM 9.2–9.3 GB.
- **New tools:** `sound_table.py` (extraneous-sound table; bed steps "created by the mix" checked against the stem). `declick.py` now iterates up to 3 passes and never raises a peak (02 / 04: 1 click left → 0). `r2_deliver.py --tag / --redeclick`.
- **Open:**
  - loudness: −12.9…−20.4 LUFS, true peak up to −0.2 dBTP;
  - layer-3 refit on top of r3;
  - Qwen fully on the GPU in the TTS session (alternating phases);
  - rare-word onsets ("Hamam");
  - TTS repetition at a sentence end (01 u07 "floor to floor");
  - the `r2_deliver --redeclick` re-mix is slow (680 s on 02; Silero frame loop in `bed_speech_runs`).

## 11. audio-r3b: final-mix loudness (-16 LUFS, TP <= -1 dBTP), handed off to A / B / D
`scripts/quality/loudness.py` (two-pass ffmpeg loudnorm, I -16, TP -1.5, linear where possible). AAC after: -15.9..-16.1 LUFS, TP -1.2..-2.7 dBTP (was -12.9..-20.4 LUFS, TP up to -0.2). Dynamic mode on 02 / 03 / 05 (02 LRA 11.5 -> 9.4). Dub voice stems untouched. Listen mp4: `/workspace/dub/c/outbox/<id>_dub-audio-r3b_*.mp4`.

## 12. 02 "Hamam": TTS lexicon + fit-up guard for names / loanwords
- **`tts_text.LEXICON`:** the transliteration "Hamam" is spoken as "hammam" (an English dictionary word). audio-r3 02 1m46s "Hamam and all this space" was heard as "Our mom…".
- **`tts_burst.dropped_words` guard:** a fit-up rewrite may not drop a word that is a transliteration of a source word (names, brands, loanwords).
  - Seen on 02 u17: the rewrite turned "Hamam and all this space." into "All this space, indeed, that is everything." Qwen's own faithfulness judge passed it.
  - A plain content-word rule would have rejected 17 legitimate paraphrases. This guard flags only the real case among the 38 accepted rewrites of 01–05.
- **02 rerun:** WER 0.075 → **0.022** (audio-r2 0.060). Both hammam units are heard exactly. u17 is now "The hammam and all this space, indeed."
