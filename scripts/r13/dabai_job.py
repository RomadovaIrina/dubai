#!/usr/bin/env python3
"""DabAI pipeline interface for the backend (r13 spec, INTERFACE.md): ONE job = mp4 + params in, mp4 + a JSON status out.

    python scripts/r13/dabai_job.py job.json            (or: --input X.mp4 --out DIR [--target-lang en] [--subs yellow] ...)
    from dabai_job import run_job; status = run_job({...})   # never raises; the returned dict = the final status.json

The job runs scripts/r9/e2e.sh (the one-command pipeline) in a child process. While it runs, <out>/status.json is rewritten every
few seconds (status RUNNING, the current step, progress %); at the end it holds SUCCESS / PARTIAL_SUCCESS / ERROR, the flags with a
user message for each, the output paths and the minutes. A crash of any stage, or of the child itself, ends as a status, never as an
exception in the caller (spec 11: the worker must survive every job)."""
from __future__ import annotations
import argparse, json, math, os, pathlib, re, shlex, subprocess, sys, threading, time, traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
VERSION = "dabai-job/1"
# progress: the share of the job each e2e stage stands for (wall-time shares of the r12 E2E on old03 / m18), and its expected s per video-minute
STEPS = [("precheck", "Checking the video", 1, 8), ("audio", "Recognising, translating and voicing the speech", 30, 140), ("sep", "Separating the background", 2, 10),
         ("face_gen", "Generating the lip movements", 45, 400), ("face_comp", "Compositing the face", 15, 150), ("subs", "Burning subtitles / watermark", 4, 25),
         ("mux", "Assembling the final video", 2, 5), ("end", "Done", 1, 1)]
MESSAGES = {   # flag -> (user message EN, RU); only flags the user should see
    "no_face": ("No face was found in the video: it is dubbed with the translated voice only, without lip-sync.",
                "В видео не найдено лицо: озвучили переведённым голосом, без синхронизации губ."),
    "face_skipped_no_face": None,
    "no_lipsync_head_pose": ("The head is turned away too often (more than 30 % of the frames): the video is dubbed with the translated voice, without lip-sync.",
                             "Голова слишком часто повёрнута (больше 30 % кадров): видео озвучено переведённым голосом, без синхронизации губ."),
    "voice_only_pose_precheck": None,
    "many_speakers_one_voice": ("More than two speakers were detected: all of them are dubbed with one voice.",
                                "В видео больше двух говорящих: все озвучены одним голосом."),
    "frames_substituted_over_2pct": ("Lip-sync failed on some frames (more than 2 %); the original picture was kept there.",
                                     "На части кадров (больше 2 %) синхронизация губ не удалась, там оставлена исходная картинка."),
    "face_gen_failed": ("Lip-sync failed: the video is dubbed with the translated voice and the original picture.",
                        "Синхронизация губ не удалась: видео озвучено переведённым голосом с исходной картинкой."),
    "face_comp_failed": ("Lip-sync failed: the video is dubbed with the translated voice and the original picture.",
                         "Синхронизация губ не удалась: видео озвучено переведённым голосом с исходной картинкой."),
    "subs_failed": ("Subtitles could not be added; the dubbed video is delivered without them.", "Субтитры добавить не удалось; видео отдано без них."),
    "no_usable_input": ("The file has no readable video or no audio track.", "В файле нет читаемого видео или звуковой дорожки."),
    "audio_failed": ("The speech could not be dubbed.", "Не удалось озвучить речь."),
    "no_output": ("The final video could not be assembled.", "Не удалось собрать итоговое видео."),
    "htdemucs_stems_failed": None, "htdemucs_mix_failed": None,
}


def _env(p: dict) -> dict:
    """job params -> the e2e.sh environment (every switch default = the accepted pipeline)"""
    e = {"ID": p["job_id"], "E2E_METRICS": "off", "E2E_FRAME_GATE": "off", "E2E_SYNC": str(p.get("sync", "1")), "DUB_MAX_VOICES": str(p.get("max_voices", 2))}
    tl = p.get("target_lang", "en")
    if tl != "en": e["DUB_TGT_LANG"] = tl
    if p.get("polyglot"): e["DUB_POLYGLOT"] = json.dumps(p["polyglot"], ensure_ascii=False)
    s = p.get("subs") or {}
    if s.get("enabled"):
        e.update(E2E_SUBS="on", E2E_SUBS_LANG=s.get("lang", "dub"), E2E_SUBS_STYLE=s.get("style", "yellow"), E2E_SUBS_KARAOKE="on" if s.get("karaoke", True) else "off")
    w = p.get("watermark") or {}
    if w.get("enabled"):
        e.update(E2E_WATERMARK=w.get("image") or f"text:{w.get('text', 'DabAI')}", E2E_WM_CORNER=w.get("corner", "br"), E2E_WM_OPACITY=str(w.get("opacity", 0.6)))
    if p.get("lipsync") == "off": e["E2E_LIPSYNC"] = "off"
    return {k: v for k, v in e.items() if v is not None}


def _progress(out: pathlib.Path, vmin: float) -> tuple[str, float]:
    """(current step, percent) from <out>/work/stages.tsv: finished steps count fully, the running one by elapsed / expected time (capped 95 %)"""
    p = out / "work" / "stages.tsv"
    rows = [l.split("\t") for l in p.read_text().splitlines() if "\t" in l] if p.exists() else []
    if not rows: return "queued", 0.0
    names = [r[0] for r in rows]; cur = names[-1]; w = {n: wt for n, _, wt, _ in STEPS}; exp = {n: s for n, _, _, s in STEPS}
    tot = sum(w.values()); done = sum(w.get(n, 0) for n in set(names[:-1]))
    frac = min(0.95, (time.time() - float(rows[-1][1])) / max(5.0, exp.get(cur, 30) * max(vmin, 0.1)))
    return cur, round(100.0 * min(1.0, (done + w.get(cur, 0) * frac) / tot), 1)


def _write(path: pathlib.Path, st: dict) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False)); os.replace(tmp, path)


def run_job(p: dict) -> dict:
    out = pathlib.Path(p["out_dir"]).resolve(); out.mkdir(parents=True, exist_ok=True); sp = out / "status.json"
    p = {**p, "job_id": re.sub(r"[^A-Za-z0-9_-]", "_", str(p.get("job_id") or pathlib.Path(p["input"]).stem))}
    st = {"version": VERSION, "job_id": p["job_id"], "status": "RUNNING", "step": "queued", "step_label": "Queued", "progress_pct": 0.0, "params": p,
          "flags": [], "messages": [], "output": None, "outputs": {}, "subs_ass": None, "video_minutes": None, "gpu_busy_s": None, "refund_suggested": False,
          "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "finished_at": None, "errors": []}
    _write(sp, st); e2e = out / "e2e"
    try:
        vmin = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p["input"]], capture_output=True, text=True).stdout.strip() or 0) / 60
    except Exception:
        vmin = 0.0
    st["video_minutes"] = round(vmin, 3)
    try:
        env = {**os.environ, **_env(p)}; log = open(out / "job.log", "w")
        proc = subprocess.Popen(["bash", str(ROOT / "scripts" / "r9" / "e2e.sh"), p["input"], str(e2e)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        to = float(p.get("timeout_s") or max(1800.0, 1200.0 * max(vmin, 0.5)))   # a hung stage must not hold the worker forever
        t0 = time.time()
        while proc.poll() is None:
            cur, pct = _progress(e2e, vmin); st.update(step=cur, step_label=dict((n, l) for n, l, _, _ in STEPS).get(cur, cur), progress_pct=pct); _write(sp, st)
            if time.time() - t0 > to:
                os.killpg(proc.pid, 9); st["errors"].append(f"timeout after {to:.0f} s in step {cur}"); break
            time.sleep(3)
        proc.wait(); log.close()
        man = json.loads((e2e / "manifest.json").read_text()) if (e2e / "manifest.json").exists() else {}
        status = man.get("status") or "ERROR"
        if st["errors"]: status = "ERROR"
        flags = man.get("flags", []); st["errors"] += man.get("errors", [])
        bed = p.get("bed", "demucs"); outs = {n: str(e2e / f"{p['job_id']}_DabAI_r9_{n}.mp4") for n in ("demucs", "spleeter") if (e2e / f"{p['job_id']}_DabAI_r9_{n}.mp4").exists()}
        main = outs.get(bed) or next(iter(outs.values()), None)
        if main:
            final = out / f"{p['job_id']}.mp4"; final.unlink(missing_ok=True); os.link(main, final) if os.stat(main).st_dev == os.stat(out).st_dev else __import__("shutil").copy(main, final)
            st["output"] = str(final)
        elif status != "ERROR":
            status = "ERROR"; flags.append("no_output")
        ass = e2e / f"{p['job_id']}.ass"
        st.update(status=status, flags=flags, outputs=outs, subs_ass=str(ass) if ass.exists() else None, progress_pct=100.0, step="end", step_label="Done",
                  messages=[{"flag": f, "en": MESSAGES[f][0], "ru": MESSAGES[f][1]} for f in flags if MESSAGES.get(f)],
                  gpu_busy_s=round((man.get("timing", {}).get("gpu_busy_total_s_per_video_min") or 0) * vmin, 1) if vmin else None,
                  wall_s=(man.get("timing") or {}).get("total_wall_s"), spec=man.get("spec"), output_format=(man.get("outputs") or {}).get(bed),
                  refund_suggested=status != "SUCCESS")
    except Exception as ex:   # the interface itself failed: still a status, never an exception for the caller
        st.update(status="ERROR", errors=st["errors"] + [f"{type(ex).__name__}: {ex}", traceback.format_exc()[-1500:]], refund_suggested=True)
    st["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); _write(sp, st)
    return st


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("job", nargs="?", help="job.json"); ap.add_argument("--input"); ap.add_argument("--out")
    ap.add_argument("--job-id"); ap.add_argument("--target-lang", default="en"); ap.add_argument("--subs", help="style: white|yellow|trendy|outline (on)")
    ap.add_argument("--subs-lang", default="dub"); ap.add_argument("--watermark", help="PNG path or text:<mark>"); ap.add_argument("--lipsync", default="auto")
    a = ap.parse_args()
    if a.job: p = json.loads(pathlib.Path(a.job).read_text())
    else:
        p = {"input": a.input, "out_dir": a.out, "job_id": a.job_id, "target_lang": a.target_lang, "lipsync": a.lipsync}
        if a.subs: p["subs"] = {"enabled": True, "style": a.subs, "lang": a.subs_lang}
        if a.watermark: p["watermark"] = {"enabled": True, **({"text": a.watermark[5:]} if a.watermark.startswith("text:") else {"image": a.watermark})}
    st = run_job(p); print(json.dumps({k: st[k] for k in ("job_id", "status", "flags", "output", "video_minutes", "gpu_busy_s")}, ensure_ascii=False))
    sys.exit(0 if st["status"] != "ERROR" else 1)


if __name__ == "__main__":
    main()
