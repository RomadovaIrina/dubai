"""GPU-busy seconds for runs timed before face_only.sh recorded them: stage windows from face.log (start) + timing.json (stage seconds), samples from logs/nvsmi.csv."""
import datetime, json, pathlib, sys
for d in map(pathlib.Path, sys.argv[1:]):
    t = json.loads((d / "timing.json").read_text()); log = (d / "logs/face.log").read_text().split()
    t0 = datetime.datetime.fromisoformat(next(w for w in log if w.startswith("20") and "T" in w)).timestamp()
    t1 = t0 + t["seconds"]["face_gen"]; t2 = t1 + t["seconds"]["face_comp"]; vmin = t["duration_s"] / 60
    smp = []
    for l in open(d / "logs/nvsmi.csv"):
        f_ = [x.strip() for x in l.split(",")]
        if len(f_) < 3 or not f_[2].isdigit(): continue
        try: smp.append((datetime.datetime.strptime(f_[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), int(f_[2])))
        except ValueError: pass
    def busy(a_, b_):
        return sum(max(0.0, min(tb, b_) - max(ta, a_)) for (ta, ua), (tb, _) in zip(smp, smp[1:]) if ua > 10)
    gb = {"face_gen": round(busy(t0, t1), 1), "face_comp": round(busy(t1, t2), 1), "face": round(busy(t0, t2), 1)}
    t.update(gpu_busy_seconds=gb, gpu_busy_s_per_video_min={k: round(v / vmin, 1) for k, v in gb.items()}, gpu_samples=len(smp), gpu_busy_note="backfilled from 500 ms samples")
    (d / "timing.json").write_text(json.dumps(t, indent=1))
    print(d.name, "wall", t["s_per_video_min"], "gpu-busy", t["gpu_busy_s_per_video_min"])
