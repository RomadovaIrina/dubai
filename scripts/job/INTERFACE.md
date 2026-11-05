# DabAI ML worker: job interface `dabai-job/2` and image (r15)

Audience: the backend developer. Russian summary: [INTERFACE_RU.md](INTERFACE_RU.md).

- **One job = one video in.** Out come the dubbed video, subtitles, a metrics JSON and a status.
- **One worker = one GPU = one job at a time.** Jobs wait in a FIFO queue. Two jobs on one 32 GB card would run out of memory (TZ 3.5).
- **Entry points** (pick one):
  - the HTTP service `scripts/job/job_api.py` (FastAPI);
  - the CLI `scripts/job/dabai_job.py`;
  - the Python call `dabai_job.run_job(job)`.
- **Machine-readable contract:** [job.schema.json](job.schema.json) (input) and [status.schema.json](status.schema.json) (output), JSON Schema 2020-12. `GET /schema/job` and `GET /schema/status` serve them too.
- **Image:** `infra/docker/` (section 8). Its **profile (`clean` / `current`) is an open product / licence decision**: a required build arg, no default.
- **What is and is not validated:** section 10.

## 1. Calling it

**HTTP** (recommended; one container per GPU):
```bash
curl -X POST http://worker:8000/jobs -H 'Authorization: Bearer $TOKEN' -H 'content-type: application/json' \
     -d '{"job_id": "a1b2c3", "input": "https://<bucket>.r2.cloudflarestorage.com/up/a1b2c3.mp4?X-Amz-Signature=...", "target_lang": "en",
          "mode": "quality", "subs": {"enabled": true, "style": "yellow"}}'
# -> 202 {"job_id": "a1b2c3", "status": "QUEUED", "queue_position": 1, "status_url": "/jobs/a1b2c3"}
curl http://worker:8000/jobs/a1b2c3 -H 'Authorization: Bearer $TOKEN'                  # poll every 3-5 s (TZ 3.6)
curl -OJ http://worker:8000/jobs/a1b2c3/files/video -H 'Authorization: Bearer $TOKEN'   # when status is SUCCESS / PARTIAL_SUCCESS
```

**CLI:**
```bash
python scripts/job/dabai_job.py job.json
python scripts/job/dabai_job.py --input in.mp4 --out /data/jobs/x --mode fast --target-lang de --subs trendy
python scripts/job/dabai_job.py --validate job.json
```
CLI exit codes:
- 0 = SUCCESS or PARTIAL_SUCCESS;
- 1 = ERROR or TIMEOUT;
- 2 = bad request.

**Python:**
```python
st = run_job(job, on_event=cb, cancel=threading.Event())
```
It never raises. A crash anywhere in the pipeline ends as a status, and the calling process survives (TZ 11.1).

## 2. Input (`job.json`)

| Field | Values | Default | Meaning |
|---|---|---|---|
| `input` | path or http(s) URL | required | Any ffmpeg-readable video with an audio track. A URL (e.g. a presigned R2 GET) is downloaded into `out_dir/input/`. **Its query string never appears in status, logs or events.** |
| `job_id` | `[A-Za-z0-9_-]{1,64}` | HTTP: generated; CLI: the file stem | Names the outputs. **Must be unique:** HTTP returns 409 on reuse. |
| `out_dir` | path | CLI: required; HTTP: **set by the service** (`$DABAI_JOBS_DIR/<job_id>`) | Where the job writes everything. |
| `source_lang` | `auto` or a code | `auto` | Whisper detects the language. RU is the validated source. A code that differs from the detected language adds the warning `W_SOURCE_LANG`. |
| `target_lang` | 22 Chatterbox ids | `en` | `en` = the accepted RU→EN path. The others and their quality group are in `reports/experiments/r13-spec/LANGS.md`. 7 of them need the external API (OpenAI key pending); until then they use the built-in voice and get the warning `W_DONOR_FALLBACK`. |
| `polyglot` | null / object | null | A target language per sentence (TZ 2.7). Forms: `{"mode": "units", "units": {"0": "de", "3": "fr"}}` (by sentence index, the rest English); `{"mode": "ranges", "ranges": [{"start": 0, "end": 20, "lang": "de"}]}` (by source time); `{"mode": "cycle", "langs": [...]}`; `{"mode": "all", "lang": "de"}`. |
| `mode` | `quality` / `fast` | `quality` | See below. |
| `lipsync` | `auto` / `off` | `auto` | `off` = a voice-only dub, with no face work. |
| `subs` | `{enabled, lang: dub/src, style: white/yellow/trendy/outline, karaoke}` | off | Burned-in subtitles (TZ 9.1–9.3). The `.ass` file is returned as well. |
| `watermark` | `{enabled, image (PNG path) / text, corner tl/tr/bl/br, opacity 0..1}` | off | TZ 9.4. |
| `output` | `{bed: auto/demucs/spleeter, both_beds, keep_intermediate}` | auto / false / false | `keep_intermediate=false` deletes the GBs of work files after the job. Outputs, logs and metrics stay. |
| `max_voices` | int ≥ 1 | 2 | TZ 6.3: with more speakers than this, everyone gets one voice + `W_MANY_SPEAKERS`. |
| `timeout_s` | number / null | max(1800, 1200 × minutes) | Past it the job is killed and the status is `TIMEOUT` (TZ 3.8). |
| `debug_env` | object | – | Tests and research only (fault injection). Never set it from the product. |

Unknown fields are rejected with `E_BAD_REQUEST` (HTTP 422), and the response lists every problem. v1 jobs (`scripts/r13/dabai_job.py`) are valid as they are.

**Modes** (the lip-sync stack; the audio chain is the same, user-accepted one in both):

| Mode | What runs | Quality (by eye) | Face cost |
|---|---|---|---|
| `quality` | LatentSync 1.6 teacher + the q2 + F3 composite + c8 overlapping windows | The eye-approved production stack | ≈ 300–360 GPU-s per video minute for the whole job |
| `fast` | The distilled 4-step student (r2, step 1500 EMA) + c8 | Sync 0.5–1.0 SyncNet below quality on m03 / m18; less tremble on m03; a smeared lip spell on m18 ~4 s | ≈ −25…35 % face GPU |

## 3. Outputs (`<out_dir>/`)

| File | What |
|---|---|
| `<job_id>.mp4` | The dubbed video: H.264 + AAC, the source resolution / fps / aspect (TZ 5.6). Rotation is baked in, so it plays upright everywhere. |
| `<job_id>.ass` | The subtitles (when `subs.enabled`). |
| `status.json` | The status (section 4). Rewritten atomically every ~3 s; final when `status` is not QUEUED / RUNNING. |
| `events.jsonl` | Append-only events, one JSON per line (section 4). |
| `metrics.json` | Final: video minutes, wall and GPU seconds (total, per video minute, per stage), VRAM peak per stage, RAM peak, lip-sync frame verdict, output format, speakers, versions. |
| `job.log`, `e2e/logs/*.log` | The pipeline log and one log per stage (for support, not for the user). |

## 4. Status, progress, events

`status`: `QUEUED → RUNNING → SUCCESS | PARTIAL_SUCCESS | ERROR | TIMEOUT` (TZ 3.3).

| Status | When | What the user gets | `refund_suggested` |
|---|---|---|---|
| SUCCESS | Everything worked; ≤ 2 % of the lip-sync frames substituted (TZ 11.3) | The video | false (true if `W_NO_FACE`) |
| PARTIAL_SUCCESS | A degraded result: > 2 % of frames substituted, lip-sync failed (the dub over the original picture), voice-only because of head pose, subtitles failed | The video + `warnings` | true (TZ 11.4) |
| ERROR | No usable input, no speech, the dub failed, no output, cancelled, worker restart | `error` | true |
| TIMEOUT | Past `timeout_s` | `error.code = E_TIMEOUT` | true |

**Progress:**
- `step` is one of: `queued`, `starting`, `download`, `precheck`, `upright`, `cfr`, `face_res`, `audio`, `sep`, `polyglot`, `pace_fix`, `face_gen`, `face_comp`, `face_hires`, `subs`, `mux`, `end`.
- `step_label` is an English label for the waiting screen (TZ 2.10).
- `progress_pct` = the finished steps' weights + the running step's elapsed / expected time, capped at 95 % of its share.
- `eta_s` = the remaining expected seconds.
- Both are estimates (expected stage time = fixed + per-video-minute, measured on r14-d) and they never go backwards.

**Events** (`events.jsonl`, `GET /jobs/{id}/events?after=<seq>`):
```json
{"seq": 3, "t": 1793865763.12, "ts": "2026-11-05T08:02:43+0000", "type": "step", "step": "audio", "step_label": "Recognising, translating and voicing the speech", "progress_pct": 2.0, "eta_s": 231}
```
- `type` is one of:
  - `status`: QUEUED or RUNNING;
  - `step`: every stage start, short ones included;
  - `progress`: every +5 %;
  - `done`: the final status + error code.
- `seq` increases by 1 per event; resume with `after=<last seq>`.

**Error codes** (`status.error.code`; ERROR / TIMEOUT only):

| Code | Retryable | Meaning |
|---|---|---|
| E_BAD_REQUEST | no | Invalid job (`error.detail` lists every problem). |
| E_INPUT_NOT_FOUND | no | The local input path does not exist. |
| E_DOWNLOAD_FAILED | yes | The input URL could not be fetched. |
| E_INPUT_TOO_LARGE | no | Over the worker limits (`DABAI_MAX_VIDEO_MIN` = 15, `DABAI_MAX_INPUT_MB` = 2048). The product limit (5 min, 1 GB, TZ 2.4) is the backend's check. |
| E_INPUT_UNREADABLE | no | No readable video or no audio track. |
| E_NO_SPEECH | no | Silence or music only: nothing to dub. |
| E_AUDIO_FAILED | yes | The dub could not be produced. |
| E_NO_OUTPUT | yes | The final video could not be assembled. |
| E_TIMEOUT | yes | Killed at `timeout_s`. |
| E_CANCELLED | no | `DELETE /jobs/{id}`, or the CLI's cancel event. |
| E_WORKER_RESTART | yes | The service restarted while the job ran. Its orphan pipeline is killed; queued jobs are NOT lost, they run after the restart. |
| E_INTERNAL | yes | The interface itself failed (a bug; see `errors`). |

**Warnings** (`status.warnings[]`: `{flag, code, en, ru}`; SUCCESS / PARTIAL_SUCCESS):
- W_NO_FACE
- W_HEAD_POSE
- W_MANY_SPEAKERS
- W_TWO_FACES
- W_FRAMES_SUBSTITUTED
- W_LIPSYNC_FAILED
- W_SUBS_FAILED
- W_DONOR_FALLBACK
- W_POLYGLOT_FAILED
- W_SOURCE_LANG

Each comes with a ready user message in English and Russian. `flags[]` also lists the technical flags (`low_res`, `face_work_downscaled`, `bed_spleeter_only`, …), which have no message.

## 5. HTTP API (`scripts/job/job_api.py`)

| Method | Path | Answer |
|---|---|---|
| POST | `/jobs` | 202 `{job_id, status: QUEUED, queue_position}`. 422 = invalid (`{code: E_BAD_REQUEST, errors[]}`); 409 = the `job_id` exists. |
| GET | `/jobs` | The last jobs (id, status, step, progress), the queue, the running job. |
| GET | `/jobs/{id}` | `status.json`; QUEUED jobs also carry `queue_position`. |
| GET | `/jobs/{id}/events?after=N` | Events with `seq > N`. |
| GET | `/jobs/{id}/files/{video\|subs\|metrics\|status\|log}` | The file; 404 until it exists. |
| DELETE | `/jobs/{id}` | Cancel. A queued job is dropped; a running one has its pipeline killed. Result: ERROR `E_CANCELLED`. |
| GET | `/health` | 200 when the node can take jobs (GPU visible, weights present, ≥ `DABAI_MIN_FREE_GB` free); 503 otherwise. No auth. |
| GET | `/version` | `{interface, pipeline (git sha), image, profile: clean / current}`. |

**Service environment:**

| Variable | Default | Meaning |
|---|---|---|
| `DABAI_JOBS_DIR` | `/data/jobs` | Where job directories go. |
| `DABAI_API_TOKEN` | unset | When set, every call except `/health` needs `Authorization: Bearer <token>`. We create no tokens: choose one. |
| `DABAI_INPUT_ROOTS` | unset | Colon-separated dirs that local input paths must be under. |
| `DABAI_MIN_FREE_GB` | 30 | Free disk below this makes `/health` return 503. |
| `PORT` | 8000 | |

**Restart behaviour** (TZ 12.4, tested, section 10):
- A job that was RUNNING ends as `E_WORKER_RESTART`, and its leftover pipeline is killed.
- QUEUED jobs run after the restart, in submit order.
- Nothing is lost silently.

**Out of scope here** (the backend's): the Redis / Celery queue across nodes, balances and refunds (the status only says `refund_suggested`), e-mails, R2 upload of the result. The backend reads `output` from the shared disk or downloads it via `/files/video`.

## 6. Behaviour worth knowing

- **4K and other large sources:**
  - Above 1920 px on the long side, the face stages run on a ≤ 1920 copy. `scripts/r15/hires_merge.py` then adds only the composite's change back onto the full-resolution frames.
  - Everything outside the mouth keeps the source pixels; the output is at the source resolution.
  - This is the fix for the r13 failure (`res4k_vertical`: `face_gen_failed` on 2160×3840). Flag: `face_work_downscaled`.
  - Root cause (r15 repro, `E2E_FACE_MAXSIDE=0`): at full 4K the generator's frame preparation runs on the CPU. A 20 s clip took 15 min with the GPU at 0 % (13 GB VRAM parked) and 28–45 GB RSS; the repro was stopped at that point. RAM and time grow with length, so a 1-minute 4K clip would pass the container's memory limit (85 GiB on r14-d).
- **Rotated phone clips:**
  - The precheck now probes the picture as displayed (r15 fix). Before, sideways frames could read as "no face".
  - The rotation is baked into the output.
- **Short-lived workers:** `keep_intermediate=false` (the default) frees the work files after each job.
- **GPU accounting:**
  - `gpu_busy_s` comes from NVML. It is exact on a dedicated GPU.
  - On a GPU shared with other processes it includes their load.

## 7. Resource needs (measured on one RTX 5090, r14-d; see REPORT.md for the runs)

| Resource | Need |
|---|---|
| GPU | One 32 GB card (RTX 5090, sm_120, driver ≥ 570). VRAM peak 27.5 GB (audio stage: Qwen 14B + Chatterbox); face_gen ~14 GB; composite ~9–13 GB. |
| GPU time | ≈ 300–375 GPU-s per video minute (quality mode, short clips; the TZ norm is 300). 10-minute and 4K runs: REPORT.md. |
| Wall time | ≈ 9–10 min for a 20 s clip (short clips are dominated by fixed load times); long videos: REPORT.md. |
| RAM | See REPORT.md (peak RSS of the job's processes: 20 s clip, 4K, 10 min). The container needs ≥ 64 GB. |
| Disk | Weights 23.2 GB (`infra/docker/models.lock.json`); image ~14–18 GB (estimate from the clean-room venvs: /venv/dabai 8.3 GB + mediapipe 0.9 GB + the CUDA runtime base; not built); per job during the run see REPORT.md, freed after the job unless `keep_intermediate`. |
| CPU | 16 threads used (`DUB_THREADS=16`); more helps the composite. |

## 8. The image (`infra/docker/`)

| File | What |
|---|---|
| `Dockerfile` | Multi-stage: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` builds; `…-cudnn-runtime-…` runs. CPython 3.11.17 (uv), the three ML venvs + the API venv from pinned locks with `--no-deps` (no resolver), llama-cpp-python 0.3.35 built for sm_120, LatentSync code at a229c39, ffmpeg 6.1.1 (Ubuntu, libass + libx264), fonts. **No weights, no NC components.** |
| `requirements.*.lock.txt` | The exact r14-d environments: `dabai` (225 pins: torch 2.7.1+cu128, Chatterbox @5de7a54, …), `mediapipe` 0.10.21, `main` (NVML), `api` (FastAPI 0.162, uvicorn 0.54). |
| `install_env.sh` | The install steps; the Dockerfile and the clean-room validation run the same script. |
| `entrypoint.sh` | `api` (default) · `job <job.json>` · `fetch-models` · `check [--deep]` · `smoke <video>` · anything else. |
| `fetch_models.sh` | Fills the models volume once (clean set, ~24 GB). Needs `HF_TOKEN` for the gated pyannote 3.1. The token is not stored. |
| `models.lock.json` | 55 weight files, size + sha256. `check --deep` verifies them. |
| `check.sh` | Self-test: weights, a real sm_120 kernel, llama CUDA offload, MediaPipe, NVML, ffmpeg libass / libx264, fonts, the clean guard. |
| `build.sh` | `docker build` + the next commands printed for the user. **Pushes nothing:** we hold no registry credentials. |

**Weights:**
- **(a) Mounted (recommended, TZ 5.1 / 10.2):**
  1. Create a volume (`/workspace/dub/dubai/models`).
  2. Run `docker run --rm --gpus all -e HF_TOKEN -v dabai-models:/workspace/dub/dubai/models <image> fetch-models` once.
  3. Every worker mounts the volume read-mostly; `HF_HUB_OFFLINE=1` is baked in, so nothing downloads at start.
- **(b) Baked:** copy a filled models directory into a derived image (`FROM <image>` + `COPY models /workspace/dub/dubai/models`). The image becomes ~45 GB. Not recommended: every model change rebuilds the image.
- **The student LoRA (mode `fast`)** is our own file. It is not on a public hub, so the operator supplies it (`STUDENT_LORA_FILE` or `STUDENT_LORA_URL`). On the laptop: `C:\repos\chkpts\latentsync_student_r14_r2_s4\step_01500_ema.pt`.

**Profile: `DUB_PROFILE=clean|current`. OPEN DECISION (product / legal), REQUIRED, no default.**
- **How it is set:**
  - `build.sh` and the Dockerfile refuse to build without it.
  - The baked value is the run-time default; `-e DUB_PROFILE=…` overrides it.
  - The entrypoint refuses to start with anything else.
  - Every `status.json` / `metrics.json` reports it (`versions.profile`).
- **The profiles:**

| Profile | Face | Background bed | Non-commercial parts | Where they come from |
|---|---|---|---|---|
| `current` | The servers' production path: LatentSync + **CodeFormer detail** (S-Lab NC) + Real-ESRGAN | **htdemucs** (no weights licence, NC data) + Spleeter second file | Yes: the deviation the user accepted on 29.10 | A SEPARATE NC volume at `/opt/dabai-nc` (`fetch-models` with `DUB_PROFILE=current` fills it; the entrypoint links it; the warning is printed at every start). Never baked into the image or the models volume. |
| `clean` | LatentSync + Real-ESRGAN layer SR, no CodeFormer | Spleeter only | None (GREY items remain: pyannote / Spleeter / Real-ESRGAN data) | – |

- The licence inventory of every component: `reports/experiments/r15-tz/LICENCES_PROD_PATH.md`.
- The look of both profiles: the A/B clips in `outbox/r15_t/` (old03 30 s, m03 20 s) and REPORT.md.
- **On the research servers** (no `DUB_PROFILE`), `e2e.sh` keeps its old defaults (`DUB_ALLOW_NC=1`, `DUB_ALLOW_NC_BED=1` = `current`). Nothing flipped silently.

**Run** (one container per GPU):
```bash
docker run -d --gpus '"device=0"' --shm-size 16g -p 8000:8000 -e DABAI_API_TOKEN=<yours> \
    -v dabai-models:/workspace/dub/dubai/models [-v dabai-nc:/opt/dabai-nc   # profile current only] \
    -v /srv/dabai/jobs:/data/jobs <image>
```

## 9. Versioning

- `dabai-job/2` is this contract. Fields may be **added** within `/2`: clients must ignore unknown fields. Removing or renaming a field, or changing a meaning, bumps it to `/3`. The old version is then still accepted for one release.
- `versions.pipeline` = the git commit of the code (`VERSION` in the image). `versions.image` = the image tag. `versions.profile` = `clean` / `current`. All of them are in every `status.json` and `metrics.json`, so every result is traceable.
- The weights are pinned by `models.lock.json` (sha256) and the software by the locks: same image + same volume = same result.

## 10. Validated / not validated (r15, 05.11.2026)

| What | How | Result |
|---|---|---|
| The interface: validation, missing input, timeout, cancel, URL download (query redacted), 404 URL, no-audio input through the real pipeline, metrics | `scripts/job/test_interface.sh` (CPU) | 25 / 25 pass |
| The HTTP API: auth, 202 / 409 / 422, cancel while queued, events, files, path traversal, token never logged, restart closes a stale RUNNING job | same script | pass |
| A burst of 10 jobs; a hard restart mid-job | `scripts/job/test_load.sh` (stand-in pipeline) | 9 / 9: FIFO, never two at once, none lost; after kill -9: running → E_WORKER_RESTART + orphan killed, queued → resumed in order |
| The install recipe | `install_env.sh` in a clean room on r14-d | All imports OK; the freeze of all 3 venvs == the locks |
| Real GPU jobs through v2 (quality, fast, clean profile, 4K, 10 min, the clean room with HF offline and no token) | r14-d | REPORT.md |
| **`docker build` itself** | – | **NOT run** (no Docker on vast; a ~20 GB build was not to be made on the laptop without asking). The Dockerfile was reviewed by hand. |
| The llama.cpp CUDA build with nvcc 12.8 | – | **NOT re-run** (r14-d has nvcc 13.0). The clean room reused the sm_120 build of the same recipe (`scripts/setup_llama_cuda.sh`, r13). |
| `fetch-models` from zero | – | **NOT run** (a 24 GB download). Its steps are the r13-validated `download_models.sh` + `r9_models.sh` pieces. |
| The `current` profile inside the image (the NC volume, its fetch step) | – | Not run in the image; the profile itself is the servers' production path (the A/B "current" arm). |
| Subtitles for CJK / Arabic / Hindi scripts (Noto fallback fonts) | – | Not checked by eye. |
