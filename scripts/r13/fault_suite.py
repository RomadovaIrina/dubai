#!/usr/bin/env python3
"""r13 spec 11.1-11.3: ONE worker process runs a queue of fault-injected jobs through the backend interface (dabai_job.run_job) and must
survive all of them, each ending with the right status. Jobs (same input; DUB_FAULT via scripts/r13/fault_gen.py, no generator file touched):
  f_one     calls:<k>         one LatentSync call raises inside the model  -> its segment passes the source through (11.1 / 11.2), SUCCESS if <= 2 %
  f_many    calls:<k..>       several calls fail                           -> > 2 % substituted -> PARTIAL_SUCCESS + frames_substituted_over_2pct (11.3)
  f_oom     oom:<k>           one CUDA OOM                                 -> the retry path, nothing lost, SUCCESS
  f_crash   crash             the generator process dies                   -> face_gen_failed, source picture + dub, PARTIAL_SUCCESS
  f_bad     (unreadable input) a truncated mp4                              -> ERROR, the worker goes on
    python scripts/r13/fault_suite.py --input 04.mp4 --out DIR [--jobs f_one,f_many,...]"""
import argparse, json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import dabai_job  # noqa: E402

GEN = str(HERE / "fault_gen.py")
JOBS = {"f_one": ({"DUB_FAULT": "calls:1"}, {"SUCCESS", "PARTIAL_SUCCESS"}), "f_many": ({"DUB_FAULT": "calls:0,1,2,3"}, {"PARTIAL_SUCCESS"}),
        "f_oom": ({"DUB_FAULT": "oom:1", "DUB_STAGE_LOG": "/workspace/dub/r13s/vram/stage_log_04.jsonl"}, {"SUCCESS"}), "f_crash": ({"DUB_FAULT": "crash"}, {"PARTIAL_SUCCESS"}), "f_bad": ({}, {"ERROR"})}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", default="f_one,f_many,f_oom,f_crash,f_bad"); a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bad = out / "bad.mp4"; bad.write_bytes(pathlib.Path(a.input).read_bytes()[:20000])   # a truncated upload
    res = {}
    for j in a.jobs.split(","):
        env, want = JOBS[j]; t0 = time.time()
        p = {"input": str(bad if j == "f_bad" else a.input), "out_dir": str(out / j), "job_id": j, "target_lang": "en",
             "debug_env": {"R10_GEN_PY": GEN, **env} if env else {}}
        st = dabai_job.run_job(p)   # never raises: the loop must reach the next job whatever happened
        gl = next(iter(pathlib.Path("/workspace/dub/artifacts/r4m").glob(f"r9g_{j}/{j}_r9g_{j}.log")), None)   # the generator log (r4_video.sh)
        faults = sum(1 for l in open(gl) if l.startswith("[fault] injected")) if gl and gl.exists() else 0
        gm = (st.get("spec") or {}).get("video_status") or {}
        res[j] = {"status": st["status"], "expected": sorted(want), "ok": st["status"] in want, "flags": st["flags"], "messages": [m["en"] for m in st["messages"]],
                  "output": st["output"], "faults_injected": faults, "degraded_ratio": gm.get("degraded_ratio"), "ls_failed_frames": gm.get("latentsync_failed_frames"),
                  "eligible": gm.get("eligible_lipsync_frames"), "wall_s": round(time.time() - t0, 1), "errors": st["errors"][:3]}
        print(f"== {j}: {st['status']} (want {sorted(want)}) flags {st['flags']} faults {faults} degraded {gm.get('degraded_ratio')} output {bool(st['output'])}", flush=True)
        (out / "fault_suite.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    res["PASS"] = all(r["ok"] for r in res.values() if isinstance(r, dict)); (out / "fault_suite.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("FAULT_SUITE", "PASS" if res["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
