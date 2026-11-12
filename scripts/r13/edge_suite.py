#!/usr/bin/env python3
"""r13 round 3: the edge-case robustness suite. Every input of /workspace/dub/r13s/edge/in through the backend interface (dabai_job.run_job,
defaults), one at a time, in ONE process (the worker must survive all of them). Per case: the expected status / flags, the output format vs the
source as DISPLAYED (rotation applied): width, height, fps, duration.
    python scripts/r13/edge_suite.py --in DIR --out DIR [--cases a,b,...]"""
import argparse, json, pathlib, subprocess, sys, time
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "job"))   # r17: the CURRENT job interface (scripts/job), not the r13 copy
import dabai_job  # noqa: E402

# case -> (expected statuses, flags that must be present, what it tests)
EXPECT = {
    "short_3s": ({"SUCCESS", "PARTIAL_SUCCESS"}, [], "3-second clip"),
    "no_audio": ({"ERROR"}, ["no_usable_input"], "no audio track"),
    "silence": ({"ERROR"}, ["no_speech"], "an audio track with silence only"),
    "music_only": ({"ERROR"}, ["no_speech"], "music only, no voice"),
    "fps60_conv": ({"SUCCESS"}, ["high_fps"], "60 fps (converted)"),
    "fps60_real": ({"SUCCESS"}, ["high_fps"], "60 fps phone clip 1072x1904"),
    "odd_fps_29.58_real": ({"SUCCESS"}, [], "odd frame rate 355/12 (real phone clip)"),
    "vfr": ({"SUCCESS"}, ["variable_fps"], "variable frame rate"),
    "rotated_meta90": ({"SUCCESS"}, ["rotated_source"], "phone rotation metadata (coded landscape, displayed portrait)"),
    "horizontal_1920x1080": ({"SUCCESS"}, ["landscape"], "horizontal 1920x1080"),
    "square_720": ({"SUCCESS"}, [], "square 720x720"),
    "res4k_vertical": ({"SUCCESS"}, [], "4K vertical 2160x3840"),
    "hdr_hlg_10bit": ({"SUCCESS"}, ["hdr_or_10bit"], "10-bit HEVC, HLG transfer"),
    "three_faces": ({"SUCCESS", "PARTIAL_SUCCESS"}, ["two_faces"], "three people side by side"),
    "face_half_out": ({"SUCCESS", "PARTIAL_SUCCESS"}, [], "face half out of frame"),
    "very_dark": ({"SUCCESS", "PARTIAL_SUCCESS"}, [], "very dark picture"),
    "long_10min": ({"SUCCESS", "PARTIAL_SUCCESS"}, ["long_video"], "10.4-min concatenation"),
}


def disp(p):
    try:
        r = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)], capture_output=True, text=True).stdout)
        v = next(s for s in r["streams"] if s["codec_type"] == "video"); n, d = v["avg_frame_rate"].split("/")
        rot = next((int(round(float(sd["rotation"]))) for sd in v.get("side_data_list", []) if "rotation" in sd), 0)
        w, h = int(v["width"]), int(v["height"]); w, h = (h, w) if abs(rot) % 180 == 90 else (w, h)
        return {"w": w, "h": h, "fps": round(float(n) / float(d), 3) if float(d) else 0, "dur": round(float(r["format"]["duration"]), 2), "pix_fmt": v.get("pix_fmt"), "codec": v.get("codec_name")}
    except Exception as e:
        return {"error": str(e)[:100]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cases", default="")
    a = ap.parse_args(); I = pathlib.Path(a.inp); O = pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    cases = a.cases.split(",") if a.cases else list(EXPECT); res = json.loads((O / "edge.json").read_text()) if (O / "edge.json").exists() else {}
    for c in cases:
        src = next(iter(I.glob(c + ".*"))); want, need, what = EXPECT[c]; t0 = time.time()
        st = dabai_job.run_job({"input": str(src), "out_dir": str(O / c), "job_id": c.replace(".", "_"), "target_lang": "en"})
        s_in, s_out = disp(src), disp(st["output"]) if st.get("output") else None
        fmt_ok = None if not s_out else (s_out.get("w") == s_in.get("w") and s_out.get("h") == s_in.get("h") and abs(s_out.get("fps", 0) - s_in.get("fps", 0)) < 0.1
                                         and abs(s_out.get("dur", 0) - s_in.get("dur", 0)) < 0.25)
        ok = st["status"] in want and all(f in st["flags"] for f in need) and (fmt_ok is not False)
        res[c] = {"what": what, "status": st["status"], "expected": sorted(want), "flags": st["flags"], "need_flags": need, "ok": ok, "format_ok": fmt_ok,
                  "src": s_in, "out": s_out, "messages": [m.get("en") for m in st.get("messages") or []], "errors": (st.get("errors") or ([st["error"]] if st.get("error") else []))[:3], "wall_s": round(time.time() - t0, 1),
                  "video_minutes": st.get("video_minutes"), "gpu_busy_s": st.get("gpu_busy_s")}
        (O / "edge.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"== {c}: {st['status']} (want {sorted(want)}) flags {st['flags']} format_ok {fmt_ok} -> {'OK' if ok else 'FAIL'}  {res[c]['wall_s']} s", flush=True)
    print("EDGE_SUITE", "PASS" if all(r["ok"] for r in res.values()) else "FAIL", sum(r["ok"] for r in res.values()), "/", len(res))


if __name__ == "__main__":
    main()
