# Отчёт: окружение `dabai` для пилота дубляжа (2026-09-08)

## Baseline (immutable) — подтверждён
| | |
|---|---|
| GPU / драйвер | RTX 5090, 32 607 MiB, driver 595.71.05 (CUDA ≤ 13.2), compute capability (12, 0) |
| Python | 3.11.16, conda env `/venv/dabai` |
| torch / torchvision / torchaudio | 2.7.1+cu128 / 0.22.1+cu128 / 2.7.1+cu128, cuDNN 9.7.1 (из wheel) |
| arch list | sm_75 … sm_100, **sm_120**, compute_120; `check_torch.py` matmul 4096² на GPU — OK |
| системный CUDA | toolkit 12.8 (`/usr/local/cuda`), cuDNN 9.8, NCCL 2.31 — не трогались |

## Версии компонентов (полный список: `reports/versions.txt`, lock: `reports/pip-freeze.lock.txt`)
| Компонент | Версия | Режим пилота | Смоук |
|---|---|---|---|
| Silero VAD | silero-vad 6.2.1 | jit-модель, 16 kHz | 3 сегмента речи на demo-аудио |
| faster-whisper | 1.2.1 + ctranslate2 4.8.2 | large-v3, `int8_float16`, word_timestamps | 31 слово с таймстампами; на sm_120 int8 отключён в CTranslate2 → фактически float16 |
| PyAnnote | pyannote.audio 3.4.0 (пайплайн speaker-diarization-3.1) | diarization | импорт OK; загрузка пайплайна **SKIP — нет HF_TOKEN** |
| RetinaFace | facexlib 0.3.0, ResNet50, fp16 | детекция лиц | 1 лицо на кадре demo-видео |
| Qwen2.5-7B | llama-cpp-python 0.3.35 (CPU wheel, AVX2/FMA), GGUF q4_k_m (2 шарда) | CPU, n_threads=16 | связный перевод, ~10 tok/s |
| Chatterbox Multilingual | git master 5de7a54 (v3 checkpoint `t3_mtl23ls_v3`) | fp16 без квантования | EN и RU синтез в fp16, 3.0 GiB VRAM |
| librosa / atempo | librosa 0.11.0, ffmpeg 6.1.1 | изменение длительности | 9.60 s → 7.68 s (librosa) / 7.71 s (atempo=1.25) |
| LatentSync | 1.6, commit a229c39, unet 5 GB + whisper tiny | stage2_512, 20 steps, DeepCache, fp16 | demo1 → `reports/latentsync_demo.mp4` за 208 s |
| CodeFormer | commit b33cc7d, вендорные basicsr 1.3.2 + facelib | restoration после LatentSync, w=0.5 | 242 кадра за 143 s → `reports/codeformer_demo.mp4` |
| FFmpeg | 6.1.1, libx264/libx265/libopus/aac | финальная сборка (CPU) | `reports/final_demo.mp4` h264 1080×1920 + aac |
| insightface / onnxruntime | 0.7.3 / onnxruntime-gpu 1.26.0 | детектор buffalo_l для LatentSync | CUDAExecutionProvider работает на sm_120 |

Общее ядро: numpy 1.26.4, scipy 1.15.3, numba 0.61.2, transformers 4.48.0, tokenizers 0.21.4, huggingface_hub 0.30.2, diffusers 0.32.2, safetensors 0.5.3, accelerate 0.26.1, opencv-python 4.9.0.80.

## Отклонения от upstream-пинов и почему
1. **Chatterbox** пинит torch==2.6.0, transformers==5.2.0, diffusers==0.29.0, gradio — установлен с `--no-deps`; код проверен на совместимость с transformers 4.48 / diffusers 0.32.2. Multilingual v3 существует только в git master (PyPI 0.1.7 старее).
2. **pyannote.audio 3.4.0** вместо буквального 3.1.x: 4.x требует torch ≥ 2.8 (конфликт с baseline); 3.1.1 без верхних пинов тянет pyannote.core 6, ломающий 3.x. Пайплайн `pyannote/speaker-diarization-3.1` тот же. huggingface_hub зафиксирован 0.30.2, потому что hub ≥ 1.0 убрал `use_auth_token`, которым пользуется pyannote 3.x.
3. **LatentSync**: torch 2.5.1 → 2.7.1 (baseline), librosa 0.10.1 → 0.11.0 (нужно Chatterbox/Perth), mediapipe и gradio не ставились (не нужны для инференса), insightface 0.7.3 собран из sdist (нужен Cython до сборки).
4. **xformers не установлен**: у релизов под torch 2.7 нет ядер sm_120, а PyPI-версия молча заменяет torch. LatentSync и Chatterbox работают на PyTorch SDPA.
5. **onnxruntime**: оставлен только `onnxruntime-gpu==1.26.0` (CUDA 12.8; ≥ 1.27 собран под CUDA 13). CPU-`onnxruntime`, который тянут faster-whisper/silero, удалён — общий namespace. `pip check` из-за этого жалуется на faster-whisper.
6. **opencv-python-headless** (через insightface → albumentations) удалён: два дистрибутива делят каталог `cv2`, headless 4.11 затенял пин 4.9.0.80; после удаления opencv-python переустановлен.
7. **CodeFormer**: вместо устаревшего `python basicsr/setup.py develop` генерируется только `basicsr/version.py`; запуск через `PYTHONPATH=third_party/CodeFormer`.

`pip check` (reports/pip-check.txt) содержит только эти намеренные расхождения плюс `decord 0.6.0 is not supported on this platform` (метаданные py3-none-manylinux2010; импорт и инференс работают).

## Ограничения инстанса, влияющие на пайплайн
- **NVENC/NVDEC не работают** (ошибка на уровне хоста, не контейнера) — все кодеки CPU. Финальная сборка: libx264 preset medium.
- **cgroup ≈ 16 vCPU / ≈ 85 GiB RAM**, но `os.cpu_count()` = 128. llama.cpp без явного `n_threads_batch` берёт 64 потока и упирается в квоту: 0.4 tok/s вместо 10. Все скрипты передают `DUB_THREADS=16`.
- **Диск не персистентный** (`workspace_is_volume=false`): при recycle/destroy теряется всё, включая `/venv/dabai` и `models/`. Всё для восстановления — в `/workspace/dub` (README.md).
- **HF_TOKEN отсутствует**: pyannote-модели gated. После добавления `HF_TOKEN=hf_...` в `/workspace/.env` (и принятия лицензий speaker-diarization-3.1 и segmentation-3.0) — `scripts/run_smoke.sh pyannote`.
- **Chatterbox fp16** — не upstream-режим (код fp32). Каст модулей в fp16 + autocast сработал без NaN на EN/RU; при артефактах в тесте есть автоматический откат на bf16/fp32 (`check_chatterbox.py`).
- **faster-whisper `int8_float16`** на Blackwell = float16 (CTranslate2 ≥ 4.6.3 отключает int8 для sm_120; версии 4.5–4.6.2 падали бы с CUBLAS_STATUS_NOT_SUPPORTED). Экономии VRAM от int8 нет.

## Артефакты
`reports/`: versions.txt, pip-freeze.lock.txt, pip-check.txt, third_party-commits.txt, smoke.md, tts_en_fp16.wav, tts_ru_fp16.wav, atempo_*.wav, retinaface_demo.jpg, latentsync_demo.mp4, codeformer_demo.mp4, final_demo.mp4.
`logs/`: setup_<stage>_*.log, download_models_*.log, check_<component>.log.

## Смоук-цепочка (финальный прогон)
| component | result | time | log |
|---|---|---|---|
| chatterbox | PASS | 28s | logs/check_chatterbox.log |
| faster-whisper | PASS | 12s | logs/check_faster-whisper.log |
| silero-vad | PASS | 2s | logs/check_silero-vad.log |
| pyannote | SKIP | 6s | logs/check_pyannote.log |
| librosa/atempo | PASS | 2s | logs/check_librosa_atempo.log |
| llama.cpp | PASS | 8s | logs/check_llama.cpp.log |
| retinaface | PASS | 5s | logs/check_retinaface.log |
| latentsync | PASS | 118s | logs/check_latentsync.log |
| codeformer | PASS | 144s | logs/check_codeformer.log |
| ffmpeg-mux | PASS | 3s | logs/check_ffmpeg-mux.log |
