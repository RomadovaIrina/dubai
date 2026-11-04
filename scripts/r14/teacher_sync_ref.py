#!/usr/bin/env python3
"""r14: the reference level for the train sync term: the stable-SyncNet BCE loss of the TEACHER's x0 on the same 16-frame windows the student
trains on (and of the real frames for own-audio windows). If the student's sync loss climbs to the teacher's level, it is the blur bias going
away (the stable SyncNet scores blurry mouths high); if it goes above the teacher's level, the sync term is losing.
    python scripts/r14/teacher_sync_ref.py --t DIR/t --log DIR/train_log.jsonl [--max 300]"""
import argparse, glob, json, pathlib, random, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "r13" / "distill"))
import numpy as np, torch, torch.nn.functional as F
import distill as D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--t", required=True); ap.add_argument("--log"); ap.add_argument("--max", type=int, default=300); ap.add_argument("--json")
    a = ap.parse_args(); vae = D.load_vae(); sn, sc = D.load_syncnet()
    fs = sorted(glob.glob(str(pathlib.Path(a.t) / "*.pt"))); random.Random(0).shuffle(fs); res = {"teacher": [], "real": [], "teacher_own": [], "teacher_ext": []}
    with torch.no_grad():
        for f in fs:
            d = torch.load(f)
            if d.get("mel") is None or d["x0_teacher"].shape[1] != 16: continue
            bce = lambda s: F.binary_cross_entropy(s.clamp(1e-6, 1 - 1e-6), torch.ones_like(s)).item()
            lt = bce(D.sync_sim(sn, sc, vae, d["x0_teacher"].cuda()[None], d["mel"])); res["teacher"].append(lt)
            res["teacher_ext" if d.get("orig_audio") else "teacher_own"].append(lt)
            if d.get("orig_audio"): res["real"].append(bce(D.sync_sim(sn, sc, vae, d["cond"][5:9].cuda()[None], d["mel"])))
            if len(res["teacher"]) >= a.max: break
    out = {k: {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None} for k, v in res.items()}
    if a.log:
        rows = [json.loads(l) for l in open(a.log) if l.strip()]; s = [(r["step"], r["sync"]) for r in rows if r.get("sync") is not None]
        last = s[-1][0] if s else 0
        for lo in range(0, last, 250):
            v = [x for st, x in s if lo < st <= lo + 250]
            if v: out[f"student_train_steps_{lo + 1}-{lo + 250}"] = {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))}
    print(json.dumps(out, indent=1))
    if a.json: pathlib.Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
