# DabAI pipeline interface for the backend (r13)

One job = one video in, one dubbed video + a JSON status out. The pipeline is `scripts/r9/e2e.sh` (one command, no manual steps); the
backend talks to it through `scripts/r13/dabai_job.py`.

## Calling it

```bash
python scripts/r13/dabai_job.py job.json
```
```python
from dabai_job import run_job
status = run_job(params)   # never raises; the returned dict = the final status.json
```
The job runs in a child process. A crash anywhere inside it (a model error, an OOM, a dead process, a hung stage past the timeout) ends
as a status, never as an exception in the caller: the worker that calls `run_job` survives every job (spec 11.1). Exit code of the CLI:
0 for SUCCESS / PARTIAL_SUCCESS, 1 for ERROR.

## Input: `job.json`

```json
{
  "input": "/data/uploads/abc.mp4",
  "out_dir": "/data/jobs/abc",
  "job_id": "abc",
  "target_lang": "en",
  "polyglot": null,
  "subs": {"enabled": true, "lang": "dub", "style": "yellow", "karaoke": true},
  "watermark": {"enabled": true, "image": null, "text": "DabAI", "corner": "br", "opacity": 0.6},
  "lipsync": "auto",
  "bed": "demucs",
  "max_voices": 2,
  "timeout_s": null
}
```
| Field | Values | Default | Meaning |
|---|---|---|---|
| `input` | path | required | The uploaded video (any ffmpeg-readable container; phone rotation is handled). |
| `out_dir` | path | required | Where the job writes everything. |
| `job_id` | `[A-Za-z0-9_-]` | input file stem | Names the outputs. |
| `target_lang` | Chatterbox language id | `en` | RU→EN is the accepted path. Other languages: see the routing table (`scripts/r13/lang_routes.json`, LANGS.md). |
| `polyglot` | `null` or a map (see LANGS.md) | `null` | Per-sentence target language in one video (behind a flag). |
| `subs.enabled` | bool | false | Burn subtitles in. The `.ass` file is also returned for soft subtitles. |
| `subs.lang` | `dub` / `src` | `dub` | Subtitles in the dubbed language (timed on the dub) or in the original (Whisper word times). |
| `subs.style` | `white` / `yellow` / `trendy` / `outline` | `yellow` | The 4 presets of spec 9.3. |
| `subs.karaoke` | bool | true | Highlight the current word. |
| `watermark.enabled` | bool | false | Overlay a logo. |
| `watermark.image` | PNG path or null | null | The brand logo; null = a rendered text mark (`watermark.text`). |
| `watermark.corner` | `tl` / `tr` / `bl` / `br` | `br` | |
| `watermark.opacity` | 0..1 | 0.6 | |
| `lipsync` | `auto` / `off` | `auto` | `off` = voice-only dub (no face work). |
| `bed` | `demucs` / `spleeter` | `demucs` | Which background separation the main output uses (both files are produced). |
| `max_voices` | int | 2 | Spec 6.3: more speakers than this → one voice + a message. |

## Output: `<out_dir>/status.json`

Rewritten every ~3 s while the job runs, final when `status` is not `RUNNING`.

```json
{
  "version": "dabai-job/1",
  "job_id": "m03",
  "status": "SUCCESS",
  "step": "end", "step_label": "Done", "progress_pct": 100.0,
  "flags": ["two_faces"],
  "messages": [{"flag": "two_faces", "en": "Two people are on screen: only the active speaker's lips are synced.", "ru": "В кадре два человека: …"}],
  "output": "/data/jobs/m03/m03.mp4",
  "outputs": {"demucs": ".../m03_DabAI_r9_demucs.mp4", "spleeter": ".../m03_DabAI_r9_spleeter.mp4"},
  "subs_ass": "/data/jobs/m03/e2e/m03.ass",
  "video_minutes": 0.675,
  "gpu_busy_s": 241.0, "wall_s": 655.2,
  "refund_suggested": false,
  "spec": {"voice_policy": {...}, "speakers": ["SPEAKER_00", "SPEAKER_01"], "faces": {...}, "pose": {...}, "video_status": {...}},
  "output_format": {"width": 720, "height": 1280, "fps": 30.0, "matches_source": true},
  "started_at": "...", "finished_at": "...", "errors": []
}
```

### `status`
| Value | When | What the user gets |
|---|---|---|
| `RUNNING` | while the job runs | progress |
| `SUCCESS` | everything worked; ≤ 2 % of the lip-sync frames substituted (11.3) | the dubbed video |
| `PARTIAL_SUCCESS` | a degraded result: > 2 % of the frames substituted, lip-sync failed (source picture + dub), voice-only by head pose (5.5), no demucs bed, subtitles failed | the video + a message; the backend refunds per 11.4 |
| `ERROR` | no usable input (unreadable, no audio track), the dub failed, no output, timeout | nothing; refund |

### Progress
`step` is the current e2e stage: `precheck → audio → sep → face_gen → face_comp → subs → mux → end`. `progress_pct` = the finished stages'
shares + the running stage's elapsed / expected time (capped at 95 % of its share). The expected times are per video-minute from the r12
runs, so the bar is approximate, never backwards.

### Flags with a user message
| Flag | Spec | Message (EN; RU in the file) |
|---|---|---|
| `no_face` | 5.5 | No face: voice-only dub. `refund_suggested` = true. |
| `no_lipsync_head_pose` | 5.5 | Head turned > 30° in > 30 % of the frames: voice-only dub. |
| `many_speakers_one_voice` | 6.3 | More than two speakers: one voice. |
| `two_faces` | 8.2 | Two people on screen: only the speaker's lips are synced. |
| `frames_substituted_over_2pct` | 11.3 | Lip-sync failed on > 2 % of the frames; the original picture kept there. |
| `face_gen_failed` / `face_comp_failed` | 11.1 | Lip-sync failed: dub + original picture. |
| `subs_failed` | 9 | Subtitles could not be added. |
| `donor_unavailable_fallback` | 7.5 | The external voice service was unavailable: built-in voice used. |
| `no_usable_input` / `audio_failed` / `no_output` | | Error messages. |

Technical flags (`low_res`, `long_video`, `landscape`, `rotated_source`, `high_fps`, …) are listed without a message.

### Minutes
`video_minutes` = the input duration in minutes (what the user is charged for). `gpu_busy_s` = GPU-busy seconds of this job (NVML
sampler, spec 1:5 accounting). `refund_suggested` = true for ERROR, PARTIAL_SUCCESS and no-face jobs; the refund itself is the backend's (11.4).
