#!/usr/bin/env python3
"""Face-region quality metrics + zoomed mouth strips for one pipeline output vs its 25 fps master (Phase 1/3 evidence).

Per LATENT_SYNC segment (bbox from facexlib RetinaFace on the master every 5 frames, held in between):
  mouth_sharp_ratio   Laplacian variance of the mouth crop, output / master  (<1 = blurrier mouth)
  upper_sharp_ratio   same for the upper face (eyes/nose): should stay ~1.0 (LatentSync must not touch it)
  flicker_ratio       mean |frame_t - frame_t-1| in the mouth crop, output / master (>1 = more temporal change than the source)
  mouth_psnr          PSNR output vs master in the mouth crop (low = mouth changed a lot: expected when lip-synced)
  upper_psnr          PSNR output vs master in the upper face (high = identity/texture preserved outside the mouth)
Per LS<->pass-through transition: mouth-crop frame jump at the boundary (|f - f-1|) for output vs master, over -3..+3.
Strips: <frames>/<id>_<tag>/mouth_seg<idx>.png (2x zoom mouth crops, row1 output row2 master) and mouth_transition_<k>.png.
    python scripts/quality/face_metrics.py --run-dir /tmp/dabai_quality/baseline --tag baseline --ids 04
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
FPS = 25; Q = pathlib.Path("/tmp/dabai_quality")


def crops(fr, bb):
    x1, y1, x2, y2 = bb; w, h = x2 - x1, y2 - y1; H, W = fr.shape[:2]
    def c(a, b, c_, d): return fr[max(0, b):min(H, d), max(0, a):min(W, c_)]
    mouth = c(x1 + int(0.15 * w), y1 + int(0.55 * h), x2 - int(0.15 * w), y2 + int(0.1 * h))
    upper = c(x1, y1, x2, y1 + int(0.5 * h))
    return mouth, upper


def sharp(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) if g.size else 0.0


def psnr(a, b):
    if a.shape != b.shape or a.size == 0: return None
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)); return 99.0 if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 2)


def strip(tiles, path, cols, scale=2.0, title=""):
    import cv2
    ts = []
    for lab, im in tiles:
        if im is None or im.size == 0: continue
        t = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC); ts.append((lab, t))
    if not ts: return None
    h = max(t.shape[0] for _, t in ts); w = max(t.shape[1] for _, t in ts); rows = math.ceil(len(ts) / cols)
    canvas = np.zeros((rows * (h + 20) + 24, cols * w, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for k, (lab, t) in enumerate(ts):
        r, c = divmod(k, cols); y = 24 + r * (h + 20); x = c * w
        canvas[y + 20:y + 20 + t.shape[0], x:x + t.shape[1]] = t
        cv2.putText(canvas, lab, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), canvas); return str(path)


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work")); ap.add_argument("--tag", default="baseline")
    ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--out", default=str(Q / "manifests")); ap.add_argument("--frames", default=str(Q / "frames"))
    ap.add_argument("--max-segments", type=int, default=8)
    a = ap.parse_args()
    from retinaface_router import RetinaFaceRouter
    router = RetinaFaceRouter(50, 80, 0.8)
    for vid in a.ids:
        run = pathlib.Path(a.run_dir); man = json.loads((run / f"{vid}_{a.tag}.manifest.json").read_text())
        segs = sorted(man["lipsync"]["segments"], key=lambda s: s["start_frame"]); ls_segs = [s for s in segs if s["action"] == "LATENT_SYNC"]
        master = pathlib.Path(a.work_dir) / f"work_{vid}" / "master_25fps.mp4"; out = run / f"{vid}_{a.tag}.mp4"
        trs = [(b["start_frame"], a_["action"], b["action"]) for a_, b in zip(segs, segs[1:]) if (a_["action"] == "LATENT_SYNC") != (b["action"] == "LATENT_SYNC")]
        seg_of = {}
        for s in ls_segs: 
            for f in range(s["start_frame"], s["end_frame"]): seg_of[f] = s["index"]
        tr_frames = {f + d: (k, d) for k, (f, _, _) in enumerate(trs) for d in range(-3, 4)}
        cm, co = cv2.VideoCapture(str(master)), cv2.VideoCapture(str(out))
        acc: dict[int, dict] = {s["index"]: {"n": 0, "sm": 0.0, "so": 0.0, "um": 0.0, "uo": 0.0, "fm": 0.0, "fo": 0.0, "nf": 0, "pm": [], "pu": []} for s in ls_segs}
        jumps: dict[int, dict] = {}
        bb = None; prev_m = prev_o = None; prev_bb = None; i = 0; keep_m = {}; keep_o = {}
        want = set()
        for s in ls_segs[:a.max_segments]:
            n_ = s["end_frame"] - s["start_frame"]; want.update(s["start_frame"] + int(n_ * f) for f in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95))
        want.update(tr_frames.keys())
        while True:
            okm, fm = cm.read(); oko, fo = co.read()
            if not (okm and oko): break
            if i % 5 == 0 or bb is None:
                r = router.classify_frame(fm)
                if r["bbox"] and r["cls"] != "NO_FACE": bb = (r["bbox"]["x1"], r["bbox"]["y1"], r["bbox"]["x2"], r["bbox"]["y2"])
            if bb is not None and (i in seg_of or i in tr_frames):
                mm, um = crops(fm, bb); mo, uo = crops(fo, bb)
                if i in seg_of and mm.size and mo.size and mm.shape == mo.shape:
                    d = acc[seg_of[i]]; d["n"] += 1; d["sm"] += sharp(mm); d["so"] += sharp(mo); d["um"] += sharp(um); d["uo"] += sharp(uo)
                    p1, p2 = psnr(mo, mm), psnr(uo, um)
                    if p1 is not None: d["pm"].append(p1)
                    if p2 is not None: d["pu"].append(p2)
                    if prev_m is not None and prev_bb == bb and (i - 1) in seg_of:
                        pmm, _ = crops(prev_m, bb); pmo, _ = crops(prev_o, bb)
                        if pmm.shape == mm.shape:
                            d["fm"] += float(np.mean(np.abs(mm.astype(np.float32) - pmm.astype(np.float32)))); d["fo"] += float(np.mean(np.abs(mo.astype(np.float32) - pmo.astype(np.float32)))); d["nf"] += 1
                if i in tr_frames and prev_m is not None and prev_bb == bb:
                    k, dlt = tr_frames[i]; pmm, _ = crops(prev_m, bb); pmo, _ = crops(prev_o, bb)
                    if pmm.shape == mm.shape and mm.size:
                        jumps.setdefault(k, {"frame": trs[k][0], "t": round(trs[k][0] / FPS, 2), "from": trs[k][1], "to": trs[k][2], "master": {}, "output": {}})
                        jumps[k]["master"][dlt] = round(float(np.mean(np.abs(mm.astype(np.float32) - pmm.astype(np.float32)))), 2)
                        jumps[k]["output"][dlt] = round(float(np.mean(np.abs(mo.astype(np.float32) - pmo.astype(np.float32)))), 2)
                if i in want: keep_m[i] = mm.copy(); keep_o[i] = mo.copy()
            prev_m, prev_o, prev_bb = fm, fo, bb; i += 1
        cm.release(); co.release()
        res = {"video": vid, "tag": a.tag, "frames_compared": i, "segments": [], "transitions": []}
        for s in ls_segs:
            d = acc[s["index"]]
            if d["n"] == 0: continue
            res["segments"].append({"index": s["index"], "start_s": s["start_s"], "end_s": s["end_s"], "frames": s["frames"], "measured": d["n"],
                                    "mouth_sharp_master": round(d["sm"] / d["n"], 1), "mouth_sharp_out": round(d["so"] / d["n"], 1), "mouth_sharp_ratio": round(d["so"] / max(d["sm"], 1e-6), 3),
                                    "upper_sharp_ratio": round(d["uo"] / max(d["um"], 1e-6), 3),
                                    "flicker_master": round(d["fm"] / max(d["nf"], 1), 2), "flicker_out": round(d["fo"] / max(d["nf"], 1), 2), "flicker_ratio": round(d["fo"] / max(d["fm"], 1e-6), 3),
                                    "mouth_psnr": round(float(np.mean(d["pm"])), 2) if d["pm"] else None, "upper_psnr": round(float(np.mean(d["pu"])), 2) if d["pu"] else None})
        for k in sorted(jumps):
            j = jumps[k]; bo = j["output"].get(0); bm = j["master"].get(0)
            base_o = np.mean([v for dd, v in j["output"].items() if dd != 0] or [1]); base_m = np.mean([v for dd, v in j["master"].items() if dd != 0] or [1])
            j["boundary_jump_out"] = bo; j["boundary_jump_master"] = bm; j["jump_vs_neighbours_out"] = round(bo / max(base_o, 1e-6), 2) if bo is not None else None
            j["jump_vs_neighbours_master"] = round(bm / max(base_m, 1e-6), 2) if bm is not None else None
            j["master"] = {str(kk): v for kk, v in j["master"].items()}; j["output"] = {str(kk): v for kk, v in j["output"].items()}
            res["transitions"].append(j)
        fd = pathlib.Path(a.frames) / f"{vid}_{a.tag}"; sheets = []
        for s in ls_segs[:a.max_segments]:
            n_ = s["end_frame"] - s["start_frame"]; fr_idx = [s["start_frame"] + int(n_ * f) for f in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
            p = strip([(f"O f{f}", keep_o.get(f)) for f in fr_idx] + [(f"M f{f}", keep_m.get(f)) for f in fr_idx], fd / f"mouth_seg{s['index']:03d}.png", cols=len(fr_idx),
                      title=f"{vid} {a.tag} mouth zoom LS seg {s['index']} {s['start_s']}-{s['end_s']}s: row1 output, row2 master")
            if p: sheets.append(p)
        for k, (f, fa, fb) in enumerate(trs):
            fr_idx = [f + d for d in range(-3, 4)]
            p = strip([(f"O f{x}", keep_o.get(x)) for x in fr_idx] + [(f"M f{x}", keep_m.get(x)) for x in fr_idx], fd / f"mouth_transition_{k:02d}_f{f}.png", cols=7,
                      title=f"{vid} {a.tag} mouth zoom transition {f/FPS:.2f}s {fa}->{fb} (-3..+3): row1 output, row2 master")
            if p: sheets.append(p)
        res["sheets"] = sheets
        op = pathlib.Path(a.out) / f"{vid}_{a.tag}.face_metrics.json"; op.parent.mkdir(parents=True, exist_ok=True); op.write_text(json.dumps(res, indent=1))
        print(f"== {vid} {a.tag}: {len(res['segments'])} LS segments, {len(res['transitions'])} transitions, {len(sheets)} sheets -> {op}")
        for s in res["segments"]:
            print(f"  seg {s['index']:3d} {s['start_s']:6.2f}-{s['end_s']:6.2f}s  mouth sharp out/master {s['mouth_sharp_out']:.0f}/{s['mouth_sharp_master']:.0f} = {s['mouth_sharp_ratio']:.2f} | upper sharp ratio {s['upper_sharp_ratio']:.2f} | "
                  f"flicker out/master {s['flicker_out']:.2f}/{s['flicker_master']:.2f} = {s['flicker_ratio']:.2f} | psnr mouth {s['mouth_psnr']} upper {s['upper_psnr']}")
        for j in res["transitions"]:
            print(f"  transition {j['t']:6.2f}s {j['from'][:12]}->{j['to'][:12]} boundary jump out {j['boundary_jump_out']} (x{j['jump_vs_neighbours_out']} vs neighbours) master {j['boundary_jump_master']} (x{j['jump_vs_neighbours_master']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
