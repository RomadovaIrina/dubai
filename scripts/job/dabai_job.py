#!/usr/bin/env python3
"""DabAI job interface v2 (r15 T) for the backend: ONE job = one video in, one dubbed video + subtitles + a metrics JSON + a status out.
Stable contract: scripts/job/INTERFACE.md (+ job.schema.json / status.schema.json next to this file). Standard library only, so the
HTTP wrapper (job_api.py) can run in its own small venv; the pipeline itself (scripts/r9/e2e.sh) runs with the ML venv in a child process.

    python scripts/job/dabai_job.py job.json                 run one job (exit 0 = SUCCESS / PARTIAL_SUCCESS, 1 = ERROR / TIMEOUT, 2 = bad request)
    python scripts/job/dabai_job.py --validate job.json      only validate + print the normalised job
    python scripts/job/dabai_job.py --input X.mp4 --out DIR [--mode fast] [--target-lang de] [--subs yellow] [--watermark text:DabAI]
    from dabai_job import run_job; st = run_job(job)         never raises; returns the final status (= <out_dir>/status.json)

Files of a job (<out_dir>): status.json (rewritten every ~3 s, atomically), events.jsonl (append-only progress / status events),
metrics.json (final), <job_id>.mp4 (the video), <job_id>.ass (subtitles, when on), job.log (pipeline stdout), e2e/logs/*.log (per stage).
v1 jobs (scripts/r13/dabai_job.py format) are accepted unchanged."""
from __future__ import annotations
import argparse, json, math, os, pathlib, re, shutil, signal, subprocess, sys, threading, time, traceback, urllib.parse, urllib.request

ROOT = pathlib.Path(os.environ.get("DABAI_ROOT") or pathlib.Path(__file__).resolve().parents[2])
HERE = pathlib.Path(__file__).resolve().parent
INTERFACE = "dabai-job/2"
WORK = pathlib.Path(os.environ.get("DABAI_WORK", "/workspace/dub"))   # e2e.sh's fixed work root (artifacts/, r8a/, gpu.lock)
MAX_MIN = float(os.environ.get("DABAI_MAX_VIDEO_MIN", "15"))          # worker safety net; the product limit (5 min, TZ 2.4) is the backend's
MAX_MB = float(os.environ.get("DABAI_MAX_INPUT_MB", "2048"))
STUDENT_LORA = os.environ.get("DABAI_STUDENT_LORA", str(ROOT / "models" / "student" / "latentsync_student_r2_s4_step01500_ema.pt"))

# progress model: e2e stage -> (label, weight, expected wall s = fixed + per_vmin * video minutes); measured on r14-d / r15 (INTERFACE.md)
STEPS = {
    "queued": ("Queued", 0, 0, 0), "starting": ("Starting", 0, 0, 0), "download": ("Downloading the video", 1, 10, 5), "precheck": ("Checking the video", 1, 5, 4),
    "upright": ("Preparing the video", 1, 5, 20), "cfr": ("Preparing the video", 1, 5, 20), "face_res": ("Preparing the video", 1, 5, 30),
    "wait_heavy": ("Waiting for the GPU", 0, 0, 0), "audio": ("Recognising, translating and voicing the speech", 30, 60, 120),
    "sep": ("Separating the background", 2, 5, 10), "polyglot": ("Voicing the other languages", 4, 30, 60), "pace_fix": ("Voicing", 2, 20, 30),
    "face_route": ("Finding who speaks on screen", 2, 20, 20),   # r16 Q: active speaker (SyncNet on the source) + beard -> the engine
    "face_keysync": ("Generating the lip movements", 50, 300, 260),   # r16: KeySync fast2; the base covers a cold start (TensorRT build + VAE autotune)
    "face_gen": ("Generating the lip movements", 40, 60, 330), "wait_comp": ("Waiting for the GPU", 0, 0, 0),
    "face_comp": ("Compositing the face", 17, 40, 330), "face_hires": ("Restoring the full resolution", 3, 10, 150),
    "subs": ("Burning subtitles / watermark", 3, 5, 25), "mux": ("Assembling the final video", 1, 2, 3), "metrics": ("Assembling the final video", 0, 1, 1),
    "end": ("Done", 0, 0, 0),
}
# flag -> (code, EN, RU); None = technical flag (listed, no user message)
MESSAGES = {
    "face_keysync_no_face": ("W_NO_FACE", "No face suitable for lip-sync was found (too small, inset or absent): the video is dubbed with the translated voice, without lip-sync.",
                             "Не найдено лицо, подходящее для синхронизации губ (слишком мелкое, во врезке или отсутствует): видео озвучено переведённым голосом, без синхронизации губ."),
    "face_keep_source_no_visible_speaker": ("W_OFFSCREEN_VOICE", "Nobody on screen speaks the dubbed lines (an off-screen voice): the picture is left as it is, without lip-sync.",
                                            "Никто в кадре не произносит озвученные фразы (голос за кадром): изображение оставлено без изменений, без синхронизации губ."),
    "no_face": ("W_NO_FACE", "No face was found in the video: it is dubbed with the translated voice only, without lip-sync.",
                "В видео не найдено лицо: озвучили переведённым голосом, без синхронизации губ."),
    "no_lipsync_head_pose": ("W_HEAD_POSE", "The head is turned away too often (more than 30 % of the frames): the video is dubbed with the translated voice, without lip-sync.",
                             "Голова слишком часто повёрнута (больше 30 % кадров): видео озвучено переведённым голосом, без синхронизации губ."),
    "many_speakers_one_voice": ("W_MANY_SPEAKERS", "More than two speakers were detected: all of them are dubbed with one voice.",
                                "В видео больше двух говорящих: все озвучены одним голосом."),
    "frames_substituted_over_2pct": ("W_FRAMES_SUBSTITUTED", "Lip-sync failed on some frames (more than 2 %); the original picture was kept there.",
                                     "На части кадров (больше 2 %) синхронизация губ не удалась, там оставлена исходная картинка."),
    "face_gen_failed": ("W_LIPSYNC_FAILED", "Lip-sync failed: the video is dubbed with the translated voice and the original picture.",
                        "Синхронизация губ не удалась: видео озвучено переведённым голосом с исходной картинкой."),
    "face_comp_failed": ("W_LIPSYNC_FAILED", "Lip-sync failed: the video is dubbed with the translated voice and the original picture.",
                         "Синхронизация губ не удалась: видео озвучено переведённым голосом с исходной картинкой."),
    "face_hires_failed": ("W_LIPSYNC_FAILED", "Lip-sync failed: the video is dubbed with the translated voice and the original picture.",
                          "Синхронизация губ не удалась: видео озвучено переведённым голосом с исходной картинкой."),
    "face_res_failed": ("W_LIPSYNC_FAILED", "Lip-sync failed: the video is dubbed with the translated voice and the original picture.",
                        "Синхронизация губ не удалась: видео озвучено переведённым голосом с исходной картинкой."),
    "subs_failed": ("W_SUBS_FAILED", "Subtitles could not be added; the dubbed video is delivered without them.", "Субтитры добавить не удалось; видео отдано без них."),
    "two_faces": ("W_TWO_FACES", "Two people are on screen: only the active speaker’s lips are synced.", "В кадре два человека: губы синхронизированы только у того, кто говорит."),
    "donor_unavailable_fallback": ("W_DONOR_FALLBACK", "The external voice service was unavailable: this language was voiced by the built-in engine; quality may be lower.",
                                   "Внешний сервис озвучки был недоступен: язык озвучен встроенным движком, качество может быть ниже."),
    "polyglot_failed": ("W_POLYGLOT_FAILED", "The other languages could not be voiced: the video is dubbed in English.",
                        "Другие языки озвучить не удалось: видео озвучено на английском."),
    "source_lang_mismatch": ("W_SOURCE_LANG", "The spoken language differs from the one selected; the detected language was used.",
                             "Язык речи отличается от выбранного; использован распознанный язык."),
    "no_usable_input": ("E_INPUT_UNREADABLE", "The file has no readable video or no audio track.", "В файле нет читаемого видео или звуковой дорожки."),
    "no_speech": ("E_NO_SPEECH", "No speech was found in the video (silence or music only): there is nothing to dub.", "В видео не найдено речи (тишина или только музыка): озвучивать нечего."),
    "audio_failed": ("E_AUDIO_FAILED", "The speech could not be dubbed.", "Не удалось озвучить речь."),
    "no_output": ("E_NO_OUTPUT", "The final video could not be assembled.", "Не удалось собрать итоговое видео."),
}
ERRORS = {   # terminal error code -> (status, EN, RU, retryable)
    "E_BAD_REQUEST": ("ERROR", "The job request is invalid.", "Некорректные параметры задачи.", False),
    "E_INPUT_NOT_FOUND": ("ERROR", "The input video was not found.", "Исходное видео не найдено.", False),
    "E_DOWNLOAD_FAILED": ("ERROR", "The input video could not be downloaded.", "Не удалось скачать исходное видео.", True),
    "E_INPUT_TOO_LARGE": ("ERROR", "The video is too long or too large.", "Видео слишком длинное или слишком большое.", False),
    "E_INPUT_UNREADABLE": ("ERROR", MESSAGES["no_usable_input"][1], MESSAGES["no_usable_input"][2], False),
    "E_NO_SPEECH": ("ERROR", MESSAGES["no_speech"][1], MESSAGES["no_speech"][2], False),
    "E_AUDIO_FAILED": ("ERROR", MESSAGES["audio_failed"][1], MESSAGES["audio_failed"][2], True),
    "E_NO_OUTPUT": ("ERROR", MESSAGES["no_output"][1], MESSAGES["no_output"][2], True),
    "E_TIMEOUT": ("TIMEOUT", "The job took too long and was stopped.", "Обработка заняла слишком много времени и была остановлена.", True),
    "E_CANCELLED": ("ERROR", "The job was cancelled.", "Задача отменена.", False),
    "E_JOB_ID_BUSY": ("ERROR", "Another job with the same job_id is running on this worker (job ids name the shared work folders): choose another job_id.",
                      "На этом воркере уже выполняется задача с таким же job_id (по нему называются общие рабочие папки): укажите другой job_id.", True),
    "E_WORKER_RESTART": ("ERROR", "The worker restarted while the job was running.", "Воркер перезапустился во время обработки.", True),
    "E_INTERNAL": ("ERROR", "Internal error of the pipeline.", "Внутренняя ошибка конвейера.", True),
}
TECH = {"low_res", "long_video", "landscape", "rotated_source", "high_fps", "odd_fps", "variable_fps", "hdr_or_10bit", "face_partly_absent",
        "face_work_downscaled", "bed_spleeter_only", "face_clean_no_codeformer", "lipsync_off_by_request", "face_skipped_no_face",
        "voice_only_pose_precheck", "htdemucs_stems_failed", "htdemucs_mix_failed", "face_keysync", "face_keysync_multi_face", "face_keysync_no_face", "face_route_failed", "codeformer_missing_clean_face", "face_route_given", "audio_reused", "face_latentsync_unavailable", "ks_shots",
        "ks_probe_failed", "ks_probe_low_conf", "ks_probe_clamped", "ks_check_offset", "ks_paste_par_retry", "beard_keysync", "voice_env_mismatch"}
STYLES = ("white", "yellow", "trendy", "outline")
_LANGS = None


def langs() -> dict:
    global _LANGS
    if _LANGS is None:
        try: _LANGS = json.loads((ROOT / "scripts" / "r13" / "lang_routes.json").read_text())["langs"]
        except Exception: _LANGS = {"en": {"group": "local"}}
    return _LANGS


def now_iso() -> str: return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def redact(u: str) -> str:
    """an input URL without its query / fragment (presigned URLs carry signatures: never in status / logs)"""
    if not re.match(r"^https?://", str(u or "")): return u
    p = urllib.parse.urlsplit(u); return urllib.parse.urlunsplit((p.scheme, p.hostname or "", p.path, "", "")) + ("?<redacted>" if p.query else "")


def versions(env: dict | None = None) -> dict:
    """env = the job's own overrides (debug_env): the profile is the one the pipeline really runs with"""
    E = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
    sha = os.environ.get("DABAI_GIT_SHA")
    if not sha:
        try: sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
        except Exception: sha = None
    if not sha and (ROOT / "VERSION").exists(): sha = (ROOT / "VERSION").read_text().strip()
    nc = E.get("DUB_ALLOW_NC", "1") == "1" or E.get("DUB_ALLOW_NC_BED", "1") == "1"   # e2e.sh defaults: 1 / 1 (= current)
    prof = "current" if nc else "clean"   # what the pipeline really loads (the image entrypoint sets both switches from DUB_PROFILE)
    return {"interface": INTERFACE, "pipeline": sha or "unknown", "image": os.environ.get("DABAI_IMAGE", "none"), "profile": prof}


# ------------------------------------------------------------------ validation
def validate(job: dict) -> tuple[dict, list[str]]:
    """-> (normalised job with every default filled, errors). v1 jobs are valid v2 jobs."""
    e: list[str] = []
    if not isinstance(job, dict): return {}, ["the job must be a JSON object"]
    allowed = {"schema", "input", "out_dir", "job_id", "source_lang", "target_lang", "polyglot", "mode", "lipsync", "subs", "watermark", "output",
               "bed", "max_voices", "timeout_s", "sync", "debug_env", "face"}
    for k in job:
        if k not in allowed: e.append(f"unknown field: {k}")
    p = {k: v for k, v in job.items() if k in allowed}
    if p.get("schema", INTERFACE) not in (INTERFACE, "dabai-job/1"): e.append(f"schema must be {INTERFACE}")
    if not p.get("input") or not isinstance(p.get("input"), str): e.append("input: a file path or an http(s) URL is required")
    if not p.get("out_dir") or not isinstance(p.get("out_dir"), str): e.append("out_dir is required")
    jid = str(p.get("job_id") or (pathlib.Path(urllib.parse.urlsplit(p["input"]).path).stem if isinstance(p.get("input"), str) else "job"))
    p["job_id"] = re.sub(r"[^A-Za-z0-9_-]", "_", jid)[:64] or "job"
    p["source_lang"] = str(p.get("source_lang") or "auto").lower()
    tl = str(p.get("target_lang") or "en").lower(); p["target_lang"] = tl
    if tl not in langs(): e.append(f"target_lang {tl!r} is not supported (supported: {', '.join(sorted(langs()))})")
    pg = p.get("polyglot")
    if pg is not None and not isinstance(pg, dict): e.append("polyglot must be null or an object (LANGS.md)")
    p["mode"] = str(p.get("mode") or "quality")
    if p["mode"] not in ("quality", "fast"): e.append("mode must be quality or fast")
    p["face"] = str(p.get("face") or os.environ.get("DABAI_FACE_DEFAULT", "auto"))   # r16: auto = KeySync, LatentSync for a bearded speaker (Q route)
    if p["face"] not in ("auto", "keysync", "latentsync"): e.append("face must be auto, keysync or latentsync")
    p["lipsync"] = str(p.get("lipsync") or "auto")
    if p["lipsync"] not in ("auto", "off"): e.append("lipsync must be auto or off")
    s = p.get("subs") or {}
    if not isinstance(s, dict): e.append("subs must be an object"); s = {}
    s = {"enabled": bool(s.get("enabled", False)), "lang": s.get("lang", "dub"), "style": s.get("style", "yellow"), "karaoke": bool(s.get("karaoke", True))}
    if s["lang"] not in ("dub", "src"): e.append("subs.lang must be dub or src")
    if s["style"] not in STYLES: e.append(f"subs.style must be one of {', '.join(STYLES)}")
    p["subs"] = s
    w = p.get("watermark") or {}
    if not isinstance(w, dict): e.append("watermark must be an object"); w = {}
    w = {"enabled": bool(w.get("enabled", False)), "image": w.get("image"), "text": w.get("text", "DabAI"), "corner": w.get("corner", "br"), "opacity": w.get("opacity", 0.6)}
    if w["corner"] not in ("tl", "tr", "bl", "br"): e.append("watermark.corner must be tl, tr, bl or br")
    try:
        w["opacity"] = float(w["opacity"]); assert 0 <= w["opacity"] <= 1
    except Exception: e.append("watermark.opacity must be a number 0..1")
    if w["enabled"] and w["image"] and not pathlib.Path(str(w["image"])).is_file(): e.append("watermark.image: file not found")
    p["watermark"] = w
    o = p.get("output") or {}
    if not isinstance(o, dict): e.append("output must be an object"); o = {}
    o = {"bed": o.get("bed", p.get("bed", "auto")), "keep_intermediate": bool(o.get("keep_intermediate", False)), "both_beds": bool(o.get("both_beds", False))}
    if o["bed"] not in ("auto", "demucs", "spleeter"): e.append("output.bed must be auto, demucs or spleeter")
    p["output"] = o; p.pop("bed", None)
    try: p["max_voices"] = int(p.get("max_voices", 2)); assert p["max_voices"] >= 1
    except Exception: e.append("max_voices must be an integer >= 1")
    if p.get("timeout_s") is not None:
        try: p["timeout_s"] = float(p["timeout_s"]); assert p["timeout_s"] > 0
        except Exception: e.append("timeout_s must be a positive number")
    if p.get("debug_env") is not None and not isinstance(p.get("debug_env"), dict): e.append("debug_env must be an object (tests only)")
    p["schema"] = INTERFACE
    return p, e


def mode_env(p: dict) -> dict:
    """the generator / composite stack per mode (brief r15: quality = q2 + F3 + c8 teacher, fast = the 4-step student r2 step 1500 EMA + c8)"""
    if p.get("face", "auto") == "keysync":   # r16: the KeySync chain for both modes (scripts/keysync/r16/ks_face.sh)
        return {"DUB_FACE": "keysync"}
    q = {"DUB_FACE": p.get("face", "auto"), "LS_CMS_READ": "1", "FC_CMS": "1", "LS_STRIDE": "8", "LS_OVERLAP": "continue",
         "FC_EXTRA": "--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2 --reg-smooth 4 --mouth-region 0.45"}
    if p["mode"] == "fast":
        q.update(R10_GEN_PY="scripts/r13/distill/student_gen.py", R13_STUDENT_LORA=STUDENT_LORA, R13_STUDENT_STEPS="4")
    return q


def e2e_env(p: dict) -> dict:
    e = {"ID": p["job_id"], "E2E_METRICS": "off", "E2E_FRAME_GATE": "off", "E2E_SYNC": str(p.get("sync", "1")), "DUB_MAX_VOICES": str(p["max_voices"])}
    e.update(mode_env(p))
    tl = p["target_lang"]; pg = p.get("polyglot") or ({"mode": "all", "lang": tl} if tl != "en" else None)
    if pg: e["DUB_POLYGLOT"] = json.dumps(pg, ensure_ascii=False)
    s = p["subs"]
    if s["enabled"]: e.update(E2E_SUBS="on", E2E_SUBS_LANG=s["lang"], E2E_SUBS_STYLE=s["style"], E2E_SUBS_KARAOKE="on" if s["karaoke"] else "off")
    w = p["watermark"]
    if w["enabled"]: e.update(E2E_WATERMARK=w["image"] or f"text:{w['text']}", E2E_WM_CORNER=w["corner"], E2E_WM_OPACITY=str(w["opacity"]))
    if p["lipsync"] == "off": e["E2E_LIPSYNC"] = "off"
    e.update({k: str(v) for k, v in (p.get("debug_env") or {}).items()})   # tests only (fault injection, research arms)
    return e


# ------------------------------------------------------------------ helpers
def _write(path: pathlib.Path, obj) -> None:
    tmp = path.with_name("." + path.name + ".tmp"); tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False)); os.replace(tmp, path)


class Events:
    def __init__(self, path: pathlib.Path, cb=None):
        self.path, self.cb, self.seq = path, cb, 0
        if path.exists():
            try: self.seq = max((json.loads(l)["seq"] for l in path.read_text().splitlines() if l.strip()), default=0)
            except Exception: pass

    def emit(self, typ: str, **kv):
        self.seq += 1; ev = {"seq": self.seq, "t": round(time.time(), 3), "ts": now_iso(), "type": typ, **kv}
        with open(self.path, "a") as f: f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if self.cb:
            try: self.cb(ev)
            except Exception: pass


def probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,width,height", "-of", "json", path],
                       capture_output=True, text=True, timeout=120)
    j = json.loads(r.stdout or "{}"); f = j.get("format") or {}; st = j.get("streams") or []
    v = next((s for s in st if s.get("codec_type") == "video"), {})
    return {"duration_s": float(f.get("duration") or 0), "size_mb": round(int(f.get("size") or 0) / 1e6, 2), "width": v.get("width"), "height": v.get("height"),
            "has_video": bool(v), "has_audio": any(s.get("codec_type") == "audio" for s in st)}


def download(url: str, dst_dir: pathlib.Path, cancel: threading.Event | None) -> pathlib.Path:
    dst_dir.mkdir(parents=True, exist_ok=True); ext = pathlib.Path(urllib.parse.urlsplit(url).path).suffix.lower()
    dst = dst_dir / ("source" + (ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext or "") else ".mp4")); tmp = dst.with_suffix(dst.suffix + ".part"); n = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dabai-worker"}), timeout=60) as r, open(tmp, "wb") as f:
        while True:
            if cancel is not None and cancel.is_set(): raise InterruptedError("cancelled")
            b = r.read(1 << 20)
            if not b: break
            n += len(b); f.write(b)
            if n > MAX_MB * 1e6: raise OverflowError(f"larger than {MAX_MB:.0f} MB")
    os.replace(tmp, dst); return dst


def _session_rss_mb(sid: int) -> float:
    """resident memory of every process of the job's session (RSS sum: shared pages counted per process, so an upper bound)"""
    tot = 0; pg = os.sysconf("SC_PAGE_SIZE")
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            st = open(f"/proc/{d}/stat").read(); rest = st[st.rindex(")") + 2:].split()
            if int(rest[3]) == sid: tot += int(open(f"/proc/{d}/statm").read().split()[1]) * pg
        except Exception: continue
    return tot / 2**20


def _du(paths) -> int:
    n = 0
    for p in paths:
        p = pathlib.Path(p)
        if p.is_file(): n += p.stat().st_size
        elif p.is_dir():
            for q in p.rglob("*"):
                try:
                    if q.is_file() and not q.is_symlink(): n += q.stat().st_size
                except OSError: pass
    return n


def job_artifacts(jid: str, e2e: pathlib.Path) -> list[pathlib.Path]:
    """everything e2e.sh writes for this job id outside <out_dir> (exact names) + the big intermediates inside it"""
    A, B = WORK / "artifacts" / "r4a", WORK / "artifacts" / "r4m"
    xs = [A / f"r9_{jid}", A / f"r9_{jid}_htd", A / f"r9_{jid}pre", A / f"r9_{jid}src", A / "sep" / jid, B / f"r9g_{jid}", B / f"r9f_{jid}",
          WORK / "r8a" / "stems_htdemucs" / jid, WORK / "artifacts" / "r3d" / "prep" / jid, e2e / "work"]
    td = WORK / "artifacts" / "r3d" / "timing"
    if td.is_dir(): xs += [f for f in td.glob(f"{jid}_*") if f.is_file()]
    return [x for x in xs if x.exists()]


def keep_logs(arts: list[pathlib.Path], dst: pathlib.Path, cap_mb: float = 50.0) -> None:
    """copy the *.log / *.json (<= 2 MB each, <= cap_mb in all) of the artifact dirs into dst before they are deleted"""
    left = cap_mb * 1e6
    for a in arts:
        if not a.is_dir() or a.name == "work": continue
        for f in sorted(a.rglob("*")):
            if f.suffix not in (".log", ".json") or not f.is_file() or f.stat().st_size > 2e6 or f.stat().st_size > left: continue
            try:
                t = dst / a.name / f.relative_to(a); t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, t); left -= f.stat().st_size
            except OSError: pass


# ------------------------------------------------------------------ the job
JL: list = []   # the job-id lock of the running job (released by run_job)


def run_job(job: dict, on_event=None, cancel: threading.Event | None = None) -> dict:
    try: return _run_job(job, on_event, cancel)
    finally:
        while JL: JL.pop().close()   # the job-id lock ends with the job (the API runs jobs one after another in one process)


def _run_job(job: dict, on_event=None, cancel: threading.Event | None = None) -> dict:
    t_start = time.time(); p, errs = validate(job)
    out = pathlib.Path(p.get("out_dir") or job.get("out_dir") or "/tmp/dabai_job_invalid").resolve(); out.mkdir(parents=True, exist_ok=True)
    sp, ev = out / "status.json", Events(out / "events.jsonl", on_event); e2e = out / "e2e"
    st = {"schema": INTERFACE, "job_id": p.get("job_id"), "status": "RUNNING", "step": "queued", "step_label": "Queued", "progress_pct": 0.0, "eta_s": None,
          "params": {**p, "input": redact(p.get("input", ""))}, "versions": versions(p.get("debug_env")), "flags": [], "warnings": [], "messages": [], "error": None,
          "output": None, "outputs": {}, "subs_ass": None, "metrics_json": None, "video_minutes": None, "gpu_busy_s": None, "wall_s": None,
          "refund_suggested": False, "started_at": now_iso(), "finished_at": None, "errors": []}

    def finish(status=None, code=None, detail=None):
        if code:
            s_, en, ru, retry = ERRORS[code]; st.update(status=s_, error={"code": code, "message": en, "message_ru": ru, "retryable": retry, "detail": detail})
            st["errors"].append(f"{code}: {detail}" if detail else code); st["refund_suggested"] = True
        elif status: st["status"] = status
        st.update(finished_at=now_iso(), wall_s=round(time.time() - t_start, 1), step="end", step_label="Done", progress_pct=100.0, eta_s=0)
        _write(sp, st); ev.emit("done", status=st["status"], error=(st["error"] or {}).get("code"), progress_pct=100.0)
        return st

    if errs:
        st["params"] = {k: (redact(v) if k == "input" else v) for k, v in job.items()} if isinstance(job, dict) else None
        return finish(code="E_BAD_REQUEST", detail="; ".join(errs))
    _write(sp, st); ev.emit("status", status="RUNNING", step="queued", progress_pct=0.0)
    try:
        src = p["input"]
        if re.match(r"^https?://", src):
            st.update(step="download", step_label=STEPS["download"][0]); _write(sp, st); ev.emit("step", step="download", progress_pct=0.0)
            try: src = str(download(src, out / "input", cancel))
            except InterruptedError: return finish(code="E_CANCELLED")
            except OverflowError as x: return finish(code="E_INPUT_TOO_LARGE", detail=str(x))
            except Exception as x: return finish(code="E_DOWNLOAD_FAILED", detail=f"{type(x).__name__}: {str(x)[:200]}")
        elif not pathlib.Path(src).is_file():
            return finish(code="E_INPUT_NOT_FOUND", detail=src)
        try: pr = probe(src)
        except Exception as x: pr = {"duration_s": 0, "size_mb": 0, "has_video": False, "has_audio": False, "probe_error": str(x)[:200]}
        vmin = pr["duration_s"] / 60; st["video_minutes"] = round(vmin, 3); st["input_probe"] = pr
        if pr["size_mb"] > MAX_MB or vmin > MAX_MIN:
            return finish(code="E_INPUT_TOO_LARGE", detail=f"{vmin:.2f} min / {pr['size_mb']} MB (worker limits {MAX_MIN} min / {MAX_MB:.0f} MB)")
        # r16 T (audit orch. 11): the job id names the GLOBAL work dirs (artifacts/r4a/r9_<id>, ...) that e2e clears; two jobs with the same id
        # on one worker destroyed each other. One exclusive lock per id for the whole job (the HTTP API already rejects a duplicate with 409).
        import fcntl
        jl_dir = pathlib.Path(os.environ.get("DABAI_JOBID_LOCKS", "/workspace/dub/artifacts/.jobid_locks")); jl_dir.mkdir(parents=True, exist_ok=True)
        jl = open(jl_dir / f"{p['job_id']}.lock", "w")
        try: fcntl.flock(jl, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: return finish(code="E_JOB_ID_BUSY", detail=p["job_id"])
        JL.append(jl)   # (released when run_job returns; Popen closes inherited fds, so e2e never holds it)
        env = {**os.environ, **e2e_env(p)}; log = open(out / "job.log", "w")
        proc = subprocess.Popen(["bash", str(ROOT / "scripts" / "r9" / "e2e.sh"), src, str(e2e)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        st["pipeline_pgid"] = proc.pid   # its own session / process group: a restarted service kills it (job_api._kill_orphan)
        to = float(p.get("timeout_s") or max(1800.0, 1200.0 * max(vmin, 0.5)))   # TZ 3.8; a hung stage must not hold the worker forever
        t0 = time.time(); n_seen = 0; last_pct = 0.0; rss_peak = 0.0; killed = None; k = 0; plan = plan_of(p)
        while proc.poll() is None:
            cur, pct, eta = progress(e2e, vmin, plan)
            st.update(step=cur, step_label=STEPS.get(cur, (cur,))[0], progress_pct=pct, eta_s=eta)
            if k % 2 == 0: rss_peak = max(rss_peak, _session_rss_mb(proc.pid))
            _write(sp, st); k += 1
            names = stage_names(e2e)   # every stage that started since the last poll gets its own event (short ones included)
            for n in names[n_seen:]: ev.emit("step", step=n, step_label=STEPS.get(n, (n,))[0], progress_pct=pct, eta_s=eta)
            if len(names) > n_seen: n_seen = len(names); last_pct = pct
            elif pct - last_pct >= 5: ev.emit("progress", step=cur, step_label=st["step_label"], progress_pct=pct, eta_s=eta); last_pct = pct
            if cancel is not None and cancel.is_set(): killed = "E_CANCELLED"
            elif time.time() - t0 > to: killed = "E_TIMEOUT"
            if killed:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                break
            time.sleep(3)
        proc.wait(); log.close(); st["peak_rss_mb"] = round(rss_peak)
        for n in stage_names(e2e)[n_seen:]:
            if n != "end": ev.emit("step", step=n, step_label=STEPS.get(n, (n,))[0], progress_pct=st["progress_pct"], eta_s=None)
        man = json.loads((e2e / "manifest.json").read_text()) if (e2e / "manifest.json").exists() else {}
        flags = list(dict.fromkeys(man.get("flags", []))); st["errors"] += man.get("errors", [])
        det = detected_lang(p["job_id"])
        if det and p["source_lang"] not in ("auto", det): flags.append("source_lang_mismatch")
        st["source_lang_detected"] = det
        outs = {n: e2e / f"{p['job_id']}_DabAI_r9_{n}.mp4" for n in ("demucs", "spleeter")}; outs = {n: f for n, f in outs.items() if f.exists()}
        bed = p["output"]["bed"]; main = outs.get("demucs" if bed == "auto" else bed) or next(iter(outs.values()), None)
        if main and not killed:
            final = out / f"{p['job_id']}.mp4"; final.unlink(missing_ok=True)
            try: os.link(main, final)
            except OSError: shutil.copy(main, final)
            st["output"] = str(final); st["bed_used"] = next(n for n, f in outs.items() if f == main)
        ass = e2e / f"{p['job_id']}.ass"
        if ass.exists() and not killed:
            fa = out / f"{p['job_id']}.ass"; shutil.copy(ass, fa); st["subs_ass"] = str(fa)
        st["outputs"] = {n: str(f) for n, f in outs.items()} if p["output"]["both_beds"] else ({st["bed_used"]: str(main)} if main and not killed else {})
        st["flags"] = flags
        st["warnings"] = [{"flag": f, "code": MESSAGES[f][0], "en": MESSAGES[f][1], "ru": MESSAGES[f][2]} for f in flags if f in MESSAGES and MESSAGES[f][0].startswith("W_")]
        st["messages"] = [{"flag": w["flag"], "en": w["en"], "ru": w["ru"]} for w in st["warnings"]]   # v1 name
        tm = man.get("timing") or {}
        st.update(gpu_busy_s=round((tm.get("gpu_busy_total_s_per_video_min") or 0) * vmin, 1) if vmin else None, spec=man.get("spec"),
                  output_format=(man.get("outputs") or {}).get(st.get("bed_used") or "spleeter"))
        mt = metrics(p, man, vmin, time.time() - t_start, rss_peak, e2e, out); _write(out / "metrics.json", mt); st["metrics_json"] = str(out / "metrics.json")
        if not p["output"]["keep_intermediate"]:
            arts = job_artifacts(p["job_id"], e2e); st["cleanup_freed_mb"] = round(_du(arts) / 1e6, 1)
            keep_logs(arts, e2e / "logs" / "artifacts")   # the stage logs (small) survive the cleanup: a failed job stays debuggable
            for a in arts: shutil.rmtree(a, ignore_errors=True) if a.is_dir() else a.unlink(missing_ok=True)
        if killed: return finish(code=killed, detail=f"after {time.time() - t0:.0f} s in step {st['step']}" if killed == "E_TIMEOUT" else None)
        status = man.get("status") or "ERROR"
        if not st["output"]:
            code = next((MESSAGES[f][0] for f in flags if f in MESSAGES and MESSAGES[f][0].startswith("E_")), "E_NO_OUTPUT")
            return finish(code=code, detail="; ".join(man.get("errors", []))[:500] or None)
        if status == "ERROR":   # an ERROR verdict with a video: keep the video, report the first error flag
            code = next((MESSAGES[f][0] for f in flags if f in MESSAGES and MESSAGES[f][0].startswith("E_")), "E_INTERNAL")
            return finish(code=code, detail="; ".join(man.get("errors", []))[:500] or None)
        st["refund_suggested"] = status != "SUCCESS" or "no_face" in flags   # 5.5: no face -> refund; 11.4: PARTIAL_SUCCESS -> refund (the backend decides)
        return finish(status=status)
    except Exception as x:   # the interface itself failed: still a status, never an exception for the caller
        st["errors"].append(traceback.format_exc()[-1500:])
        return finish(code="E_INTERNAL", detail=f"{type(x).__name__}: {x}")


def stage_names(e2e: pathlib.Path) -> list[str]:
    p = e2e / "work" / "stages.tsv"
    try: return [l.split("\t")[0] for l in p.read_text().splitlines() if "\t" in l] if p.exists() else []
    except OSError: return []


def plan_of(p: dict) -> list[str]:
    """the e2e steps this job is expected to pass (progress weights / ETA); steps that appear anyway add their weight"""
    pl = ["precheck", "audio"] + (["sep"] if os.environ.get("DUB_ALLOW_NC_BED", "1") == "1" else [])
    if p.get("polyglot") or p["target_lang"] != "en": pl.append("polyglot")
    if p["lipsync"] != "off": pl += {"keysync": ["face_route", "face_keysync"], "auto": ["face_route", "face_keysync"]}.get(p.get("face", "auto"), ["face_gen", "face_comp"])
    if p["subs"]["enabled"] or p["watermark"]["enabled"]: pl.append("subs")
    return pl + ["mux"]


def progress(e2e: pathlib.Path, vmin: float, plan: list[str] | None = None) -> tuple[str, float, float | None]:
    """(current step, percent, ETA s) from <e2e>/work/stages.tsv: finished steps count fully, the running one by elapsed / expected (capped 95 %)"""
    p = e2e / "work" / "stages.tsv"
    rows = [l.split("\t") for l in p.read_text().splitlines() if "\t" in l] if p.exists() else []
    if not rows: return "starting", 0.0, None
    names = [r[0] for r in rows]; cur = names[-1]
    exp = lambda n: STEPS.get(n, (n, 0, 30, 0))[2] + STEPS.get(n, (n, 0, 30, 0))[3] * max(vmin, 0.05)
    wt = lambda n: STEPS.get(n, (n, 0, 0, 0))[1]
    plan = plan or ["precheck", "audio", "sep", "face_gen", "face_comp", "mux"]
    seen = list(dict.fromkeys(names)); tot = sum(wt(n) for n in dict.fromkeys(plan + seen))
    el = time.time() - float(rows[-1][1]); frac = min(0.95, el / max(5.0, exp(cur)))
    done = sum(wt(n) for n in seen[:-1]); pct = round(100.0 * min(1.0, (done + wt(cur) * frac) / max(tot, 1)), 1)
    rest = [n for n in plan if n not in seen]; eta = round(max(0.0, exp(cur) - el) + sum(exp(n) for n in rest))
    return cur, pct, eta


def detected_lang(jid: str) -> str | None:
    d = WORK / "artifacts" / "r4a" / f"r9_{jid}" / jid
    for m in sorted(d.glob("*.manifest.json")) if d.is_dir() else []:
        try:
            v = json.loads(m.read_text()).get("source_lang")
            if v: return str(v)
        except Exception: pass
    return None


def metrics(p: dict, man: dict, vmin: float, wall: float, rss: float, e2e: pathlib.Path, out: pathlib.Path) -> dict:
    tm = man.get("timing") or {}; g = {}
    try: g = json.loads((e2e / "gpu_time.json").read_text()).get("total", {})
    except Exception: pass
    vs = ((man.get("spec") or {}).get("video_status") or {})
    return {"schema": INTERFACE, "job_id": p["job_id"], "mode": p["mode"], "target_lang": p["target_lang"], "status": man.get("status"), "flags": man.get("flags", []),
            "video_minutes": round(vmin, 3), "wall_s": round(wall, 1), "wall_s_per_video_min": round(wall / vmin, 1) if vmin else None,
            "gpu_busy_s": round((tm.get("gpu_busy_total_s_per_video_min") or 0) * vmin, 1) if vmin else None,
            "gpu_busy_s_per_video_min": tm.get("gpu_busy_total_s_per_video_min"), "gpu_nvml": g or None,
            "stage_wall_s": tm.get("stage_wall_s"), "gpu_busy_s_by_stage": tm.get("gpu_busy_s"), "vram_peak_mib": tm.get("vram_peak_mib"),
            "vram_peak_mib_by_stage": tm.get("vram_peak_mib_by_stage"), "ram_peak_rss_mb": round(rss),
            "lipsync": {k: vs.get(k) for k in ("eligible_lipsync_frames", "latentsync_failed_frames", "held_landmark_frames", "degraded_ratio", "voice_only")} if vs else None,
            "output_format": (man.get("outputs") or {}), "source": man.get("source"), "speakers": (man.get("spec") or {}).get("speakers"),
            "disk_out_dir_mb": round(_du([out]) / 1e6, 1), "versions": versions(p.get("debug_env"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", nargs="?", help="job.json"); ap.add_argument("--validate", action="store_true"); ap.add_argument("--input"); ap.add_argument("--out")
    ap.add_argument("--job-id"); ap.add_argument("--mode", default="quality"); ap.add_argument("--target-lang", default="en"); ap.add_argument("--source-lang", default="auto")
    ap.add_argument("--subs", help="style: white|yellow|trendy|outline (on)"); ap.add_argument("--subs-lang", default="dub")
    ap.add_argument("--watermark", help="PNG path or text:<mark>"); ap.add_argument("--lipsync", default="auto"); ap.add_argument("--face", default=None, help="auto (default: KeySync, LatentSync for a beard) | keysync | latentsync"); ap.add_argument("--keep-intermediate", action="store_true")
    a = ap.parse_args()
    if a.job: job = json.loads(pathlib.Path(a.job).read_text())
    else:
        job = {"input": a.input, "out_dir": a.out, "job_id": a.job_id, "mode": a.mode, "target_lang": a.target_lang, "source_lang": a.source_lang,
               "lipsync": a.lipsync, "face": a.face, "output": {"keep_intermediate": a.keep_intermediate}}
        if a.subs: job["subs"] = {"enabled": True, "style": a.subs, "lang": a.subs_lang}
        if a.watermark: job["watermark"] = {"enabled": True, **({"text": a.watermark[5:]} if a.watermark.startswith("text:") else {"image": a.watermark})}
        job = {k: v for k, v in job.items() if v is not None}
    if a.validate:
        p, e = validate(job); print(json.dumps({"valid": not e, "errors": e, "job": p}, indent=1, ensure_ascii=False)); return 0 if not e else 2
    st = run_job(job)
    print(json.dumps({k: st.get(k) for k in ("job_id", "status", "error", "flags", "output", "video_minutes", "gpu_busy_s", "wall_s")}, ensure_ascii=False))
    return 0 if st["status"] in ("SUCCESS", "PARTIAL_SUCCESS") else (2 if (st.get("error") or {}).get("code") == "E_BAD_REQUEST" else 1)


if __name__ == "__main__":
    sys.exit(main())
