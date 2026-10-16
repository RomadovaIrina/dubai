# Agent C licence record — audio timing track + clean background bed candidates (2026-10-16)

Rule (user decision relayed by the coordinator, 2026-10-16): the licence **of the files** decides. Allowed: MIT / Apache-2.0 / BSD / ISC /
OpenRAIL. Not allowed: non-commercial / research-only licences and weights with **no** licence. Training data is recorded, no longer blocking.
Desk research + primary pages, checked 2026-10-16; not legal advice.

## Background-bed separators (replacement for torchaudio `HDEMUCS_HIGH_MUSDB_PLUS`)

| candidate | code licence | weight files (where, sha256) | weight licence (primary source) | training data (recorded) | verdict |
|---|---|---|---|---|---|
| torchaudio HDEMUCS_HIGH_MUSDB_PLUS (cand5 bed, `models/demucs/hdemucs_high_trained.pt`) | BSD-2 (torchaudio) | download.pytorch.org | **none stated** | MUSDB18-HQ + extra | **NO** (coordinator: weights have no licence) |
| Demucs v4 `htdemucs` / `htdemucs_ft` (facebookresearch/demucs) | MIT ([README](https://github.com/facebookresearch/demucs): "Demucs is released under the MIT license") | dl.fbaipublicfiles.com | **none stated** for the weights | MUSDB18-HQ + 800 internal songs | **NO / UNCLEAR** (same situation as HDEMUCS) |
| **Open-Unmix UMX-HQ** (`openunmix` 1.3.0) | MIT ([repo](https://github.com/sigsep/open-unmix-pytorch); wheel METADATA `License: MIT`) | [zenodo 3370489](https://zenodo.org/records/3370489) v1.0.1: `vocals-b62c91ce.pth` b62c91ce…d083, `bass-8d85a5bd.pth`, `drums-9619578f.pth`, `other-b52fbbf7.pth` (file hash prefixes verified) | **MIT License** (Zenodo record "Rights") | MUSDB18-HQ (NC dataset) | **YES** (weights MIT; NC data recorded) |
| Open-Unmix UMX (MUSDB18, 16 kHz bandwidth) | MIT | [zenodo 3370486](https://doi.org/10.5281/zenodo.3370486) | (same family; not fetched) | MUSDB18 | not tested (UMX-HQ supersedes it) |
| **Open-Unmix UMX-SE** (speech enhancement: speech vs noise) | MIT | [zenodo 3786908](https://zenodo.org/records/3786908): `speech_f5e0d9f9.pth` f5e0d9f9…9f94, `noise_04a6fc2d.pth` 04a6fc2d…353f | **MIT License** (Zenodo record "Rights") | Voicebank+DEMAND (28 speakers) | **YES** |
| Open-Unmix UMX-L | MIT | [zenodo 5069601](https://doi.org/10.5281/zenodo.5069601) | README: "the weights are only licensed for non-commercial use (CC BY-NC-SA 4.0)" | private compressed stems | **NO** |
| **Deezer Spleeter 2stems** | MIT ([README](https://github.com/deezer/spleeter): "The code of Spleeter is MIT-licensed.") | GitHub release asset of the MIT repo, [v1.4.0 `2stems.tar.gz`](https://github.com/deezer/spleeter/releases/tag/v1.4.0) sha256 f3a90b39…d692 (= the release's `checksum.json`) | no model-specific licence file; distributed as a release asset of the MIT repository; the [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.02154) presents code + pretrained models as the open release (training data withheld for copyright) | Deezer internal catalogue (not released) | **YES, with a note**: MIT by repository release; ask counsel whether the repo LICENSE covers the release assets |
| no model: the video's own room tone (`background_bed.room_tone`, steady noise-floor frames outside source speech, shuffled + crossfaded) | ours | – | – | the input video itself | **YES** (fallback / complement; no music bed) |

Downloaded copies (not in git): `/workspace/dub/artifacts/audio-timing/bed_models/{umxhq,umxse,spleeter}/`. `openunmix` code installed with
`--no-deps --target /workspace/dub/artifacts/audio-timing/pylib` (shared `/venv/dabai` untouched; its torch / torchaudio / numpy are used).

## Models this track runs (all CPU, offline A/B)

| component | use here | licence | verdict |
|---|---|---|---|
| Silero VAD | source / dub speech masks (`dub_gaps.py`, `pause_rhythm.py`) | MIT | YES |
| faster-whisper large-v3 (CTranslate2) | WER of the dub track (`eval_cpu.py`) | MIT (openai/whisper weights) / MIT (faster-whisper) | YES |
| wav2vec2-base-960h | CTC CER / word spans | Apache-2.0 | YES |
| Chatterbox Multilingual v3, Qwen2.5-7B-Instruct | not run in P1 (raw TTS reused); layer-3 design only | MIT / Apache-2.0 | YES |
| pyannote diarization | not re-run (stored diarization reused) | MIT (coordinator: now allowed) | YES |
