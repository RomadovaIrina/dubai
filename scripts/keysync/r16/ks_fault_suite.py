#!/usr/bin/env python3
"""r16 TZ 11.1-11.3 on KeySync: ONE worker process (scripts/job/dabai_job.run_job, the live backend entry) runs a queue of fault-injected jobs
and must survive all of them, each ending with the right status and the substituted-frame counts in status.spec.video_status.
Faults (KS_FAULT via the job's debug_env; no code path is bypassed):
  k_ok     none                         the clean run (its dub is reused by the single-shot faults: DUB_AUDIO_FROM)   -> SUCCESS, counts 0..2 %
  k_crash  crash                        the KeySync generator dies on its first full render                           -> PARTIAL (face_gen_failed), source picture + dub
  k_few    guard:0.01                   1 % of the frames broken -> replaced (ks_guard)                                -> SUCCESS, ~1 % substituted in the status
  k_many   guard:0.05                   5 % broken                                                                    -> PARTIAL + frames_substituted_over_2pct
  k_shot   shot:1 (2-shot input)        one shot fails -> keeps its source frames, the other shot is lip-synced        -> PARTIAL (ks_shot_failed) + counts
  k_bad    (a truncated upload)         unreadable input                                                               -> ERROR, the worker goes on
    python scripts/keysync/r16/ks_fault_suite.py --single m18_20s.mp4 --shots two_shots.mp4 --out DIR [--jobs k_ok,k_crash,...]"""
import argparse, glob, json, pathlib, sys, time
R = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R / "scripts" / "job"))
import dabai_job  # noqa: E402

JOBS = {   # name: (input key, debug_env, expected statuses, a flag that must be present or None, reuse the single-shot dub)
    "k_ok": ("single", {}, {"SUCCESS"}, None, False),
    "k_crash": ("single", {"KS_FAULT": "crash"}, {"PARTIAL_SUCCESS"}, "face_gen_failed", True),
    "k_few": ("single", {"KS_FAULT": "guard:0.01"}, {"SUCCESS"}, None, True),
    "k_many": ("single", {"KS_FAULT": "guard:0.05"}, {"PARTIAL_SUCCESS"}, "frames_substituted_over_2pct", True),
    "k_shot": ("shots", {"KS_FAULT": "shot:1"}, {"PARTIAL_SUCCESS"}, "ks_shot_failed", False),
    "k_bad": ("bad", {}, {"ERROR"}, None, False),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--single", required=True); ap.add_argument("--shots", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", default="k_ok,k_crash,k_few,k_many,k_shot,k_bad"); a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bad = out / "bad.mp4"; bad.write_bytes(pathlib.Path(a.single).read_bytes()[:20000])   # a truncated upload
    ins = {"single": a.single, "shots": a.shots, "bad": str(bad)}; dub_dir = None; res = {}
    for j in a.jobs.split(","):
        key, env, want, need, reuse = JOBS[j]; env = dict(env)
        if reuse and dub_dir: env["DUB_AUDIO_FROM"] = dub_dir
        job = {"job_id": f"ksf_{j}", "input": ins[key], "out_dir": str(out / j), "target_lang": "en", "output": {"keep_intermediate": j == "k_ok"}, "debug_env": env}
        t0 = time.time()
        try: st = dabai_job.run_job(job)
        except Exception as e: st = {"status": "WORKER_CRASH", "error": repr(e)}   # must never happen: the worker has to survive every fault
        vs = ((st.get("spec") or {}).get("video_status")) or {}
        ok = st.get("status") in want and (need is None or need in (st.get("flags") or []))
        res[j] = {"status": st.get("status"), "want": sorted(want), "flags": st.get("flags"), "error": (st.get("error") or {}).get("code") if isinstance(st.get("error"), dict) else st.get("error"),
                  "warnings": [w.get("code") for w in st.get("warnings") or []], "refund_suggested": st.get("refund_suggested"),
                  "video_status": {k: vs.get(k) for k in ("eligible_lipsync_frames", "substituted_frames", "guard_replaced_frames", "failed_frames", "degraded_ratio", "status")},
                  "audio_reused": bool(env.get("DUB_AUDIO_FROM")), "wall_s": round(time.time() - t0, 1), "pass": ok}
        print(f"{'PASS' if ok else 'FAIL'} {j}: {json.dumps(res[j], ensure_ascii=False)}", flush=True)
        if j == "k_ok":   # the clean run's dub = the input of the single-shot faults (the same video: only the face stage differs)
            c = sorted(glob.glob(f"/workspace/dub/artifacts/r4a/*/ksf_k_ok/final_mix_aac.m4a")); dub_dir = str(pathlib.Path(c[0]).parent) if c else None
            print(f"   dub for the faults: {dub_dir}", flush=True)
    (out / "ks_fault_suite.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    n_ok = sum(r["pass"] for r in res.values()); print(f"KS_FAULT_SUITE {n_ok}/{len(res)} pass; the worker survived every job: {all(r['status'] != 'WORKER_CRASH' for r in res.values())}")
    return 0 if n_ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
