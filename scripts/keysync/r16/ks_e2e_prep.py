#!/usr/bin/env python3
"""r16 T: the KeySync face stage's speaker -> face plan + the per-face KeySync inputs.
Input: the MediaPipe pass of the 25 fps master (scripts/pilot/mp_landmarks.py --num-faces N) + the dub (16 kHz) + optionally Q's route.json
(scripts/r16q/face_route.py, reports/experiments/r16-qa/ROUTING.md: SyncNet active speaker on the SOURCE).
  - route.json (the default when e2e made one): every route face is matched to a MediaPipe track (most frames inside its present_s, then the nearest
    box centre). mode "lipsync" faces: KeySync for their WHOLE presence (Irina 07.11: no lip-sync <-> source switch on a visible face); the dub fed to
    KeySync is SILENT over the face's silence_ranges (speech that is not its own), so the mouth stays closed and continuous. Route faces on the same
    MediaPipe track (one person split by S3FD) are one KeySync job. mode "source" faces: never lip-synced, subtracted from every paste mask (--others).
    keep_source_all -> plan mode "keep_source" (no face work).
  - no route.json: the r16 T fallback = the dubbed units mapped to faces by mouth-motion correlation (clean_face_stack.assign_speakers).
  - no-face rule (Q / K2: q05 product-shot PiP, q06 VHS near-profile): a speaking track with median IOD < KS_MIN_IOD px (default 22) or present in
    < 10 % of the frames is NOT lip-synced (kept as source) - KeySync's lower face turns to garbage on small / inset faces.
  - a speaking face's landmarks are NaN outside its presence: ks_plain then leaves those frames to the source (--absent skip).
Writes OUT/faces.json + per face: lm_<tag>.npz, others_<tag>.npz, dub_<tag>.wav (16 kHz mono, NOT delayed: the delay probe sets that)."""
import argparse, glob, json, os, sys, wave
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../pilot"))
import clean_face_stack as cfs

ap = argparse.ArgumentParser()
ap.add_argument("--mp", required=True); ap.add_argument("--dub", required=True); ap.add_argument("--manifest", default="")
ap.add_argument("--route", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--pad", type=float, default=0.08, help="s kept around a face's own units when muting the others (no-route fallback)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
FPS = 25.0; MIN_IOD = float(os.environ.get("KS_MIN_IOD", "22"))
z = np.load(a.mp); LM1 = z["lm"].astype(np.float32); T = len(LM1)
LA = z["lm_all"].astype(np.float32) if "lm_all" in z.files else LM1[:, None]


def hold(lm):   # = clean_face_stack: landmark gaps <= HOLD_GAP_FRAMES inside a face run -> the previous valid frame
    lm = lm.copy(); ok = np.isfinite(lm[:, 0, 0]); i = 0
    while i < T:
        if not ok[i]:
            j = i
            while j < T and not ok[j]: j += 1
            if 0 < i and j < T and j - i <= cfs.HOLD_GAP_FRAMES: lm[i:j] = lm[i - 1]
            i = j
        else: i += 1
    return lm


def bridge(lm, keep=None, max_gap=int(os.environ.get("KS_BRIDGE_FRAMES", "50"))):
    """after hold(): landmark gaps up to max_gap frames (2 s) INSIDE the face's presence are linearly interpolated, so the paste mask follows
    the face through a short head turn / occlusion; longer gaps (a cutaway, a title card) and frames outside `keep` stay NaN = the source
    (ks_plain --absent skip)."""
    lm = lm.copy(); ok = np.isfinite(lm[:, 0, 0]); i = 0
    while i < T:
        if not ok[i]:
            j = i
            while j < T and not ok[j]: j += 1
            if 0 < i and j < T and j - i <= max_gap:
                w_ = (np.arange(1, j - i + 1) / (j - i + 1))[:, None, None]; lm[i:j] = lm[i - 1] * (1 - w_) + lm[j] * w_
            i = j
        else: i += 1
    if keep is not None: lm[~keep] = np.nan
    return lm


with wave.open(a.dub) as w:
    sr = w.getframerate(); assert w.getnchannels() == 1 and w.getsampwidth() == 2, "dub must be 16-bit mono"
    pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).copy()
units = []
man = a.manifest or next(iter(sorted(glob.glob(os.path.join(os.path.dirname(a.dub), "*.manifest.json")))), "")
if man and os.path.exists(man):
    units = [{"speaker": u["speaker"], "start": float(u["start"]), "end": float(u["end"])} for u in (json.load(open(man)).get("units") or [])
             if u.get("speaker") is not None and float(u.get("end", 0)) > float(u.get("start", 0))]
tr, _ = cfs.track_faces(LA)
pres = np.isfinite(tr[:, :, 0, 0])
present = [j for j in range(tr.shape[1]) if pres[:, j].mean() >= 0.02]
both = int((np.isfinite(LA[:, 0, 0, 0]) & np.isfinite(LA[:, 1, 0, 0])).sum()) if LA.shape[1] >= 2 else 0
co = both / max(int(np.isfinite(LA[:, :, 0, 0]).any(1).sum()), 1)
iod = lambda lm: float(np.nanmedian(np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1))) if np.isfinite(lm[:, 0, 0]).any() else 0.0
plan = {"frames": T, "tracks": len(present), "co_occurrence": round(co, 4), "units": len(units), "manifest": man, "route": a.route or None, "faces": [], "skipped": []}


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.astype(np.int16).tobytes())


def others_npz(path, idx):
    o = np.full((T, max(2, len(idx)), 478, 2), np.nan, np.float32)
    for q, j in enumerate(idx): o[:, q] = tr[:, j]
    np.savez_compressed(path, others=o)


def silence(x, ranges, keep=()):
    x = x.copy(); kp = np.zeros(len(x), bool); muted = 0.0
    for s0, s1 in keep: kp[max(0, int(s0 * sr)):int(s1 * sr)] = True
    for s0, s1 in ranges:
        i0, i1 = max(0, int(s0 * sr)), min(len(x), int(s1 * sr))
        if i1 > i0: seg = ~kp[i0:i1]; x[i0:i1][seg] = 0; muted += seg.sum() / sr
    return x, round(muted, 2)


def dub_vad(x, thr_rel=0.03, gap=0.25, min_len=0.1):
    """r16 T (Q audit O6): speech segments of the DUB (20 ms RMS > thr_rel x the clip's p99, gaps < gap s merged). The route's speak / silence ranges
    come from the SOURCE diarization; an EN phrase can run past its RU turn, so a face keeps every dub speech segment that overlaps its own turns."""
    hop = sr // 50; nf = len(x) // hop
    if nf == 0: return []
    e = np.sqrt((x[:nf * hop].astype(np.float32).reshape(nf, hop) ** 2).mean(1)); on = e > thr_rel * max(float(np.percentile(e, 99)), 1.0)
    segs, k = [], 0
    while k < nf:
        if on[k]:
            j = k
            while j < nf and on[j]: j += 1
            if segs and k / 50 - segs[-1][1] < gap: segs[-1][1] = j / 50
            else: segs.append([k / 50, j / 50])
            k = j
        else: k += 1
    return [g for g in segs if g[1] - g[0] >= min_len]


def extend_keep(keep, vad):   # every dub speech segment overlapping an own turn is kept whole
    out = list(keep)
    for v0, v1 in vad:
        if any(v0 < k1 and v1 > k0 for k0, k1 in keep): out.append((v0, v1))
    return out


def add_face(j, lm, x, info):
    tag = f"f{len(plan['faces'])}"; v = iod(lm); pr = float(np.isfinite(lm[:, 0, 0]).mean())
    if v < MIN_IOD or pr < 0.10:
        plan["skipped"].append({"track": int(j), "reason": f"no-face rule: median IOD {v:.0f} px (< {MIN_IOD:.0f}) or presence {pr:.2f} (< 0.10)", **info}); return None
    np.savez_compressed(f"{a.out}/lm_{tag}.npz", lm=lm); write_wav(f"{a.out}/dub_{tag}.wav", x)
    plan["faces"].append({"tag": tag, "track": j if isinstance(j, str) else int(j), "iod_px": round(v, 1), "presence": round(pr, 3), **info}); return tag


route = json.load(open(a.route)) if a.route and os.path.exists(a.route) else None
VAD = dub_vad(pcm) if route is not None else []
if route is not None:
    plan["mode"] = "route"
    if route.get("keep_source_all"):
        plan["mode"] = "keep_source"
    else:
        groups, src_tracks = {}, set()
        for rf in route.get("faces", []):
            t0, t1 = rf.get("present_s", [0, T / FPS]); k0, k1 = max(0, int(t0 * FPS)), min(T, int(np.ceil(t1 * FPS)))
            SNAP = int(os.environ.get("KS_EDGE_SNAP", "13"))   # r16c T: a route span ending < 0.5 s before the video end (m03: 40.44 of 40.52 s) reaches the end, else the last frames jump to the source
            k0, k1 = (0 if k0 <= SNAP else k0), (T if T - k1 <= SNAP else k1)
            if k1 <= k0 or not present: continue
            cx, cy = rf.get("box_center_xy", [np.nan, np.nan])
            def score(j):
                inside = pres[k0:k1, j].mean()
                c = np.nanmedian(tr[k0:k1, j].mean(1), axis=0) if inside > 0 else np.array([np.inf, np.inf])
                return (round(inside, 1), -float(np.hypot(c[0] - cx, c[1] - cy)) if np.isfinite(cx) else 0.0)
            j = max(present, key=score)
            if pres[k0:k1, j].mean() < 0.3:   # MediaPipe has no landmarks for this S3FD face (q07 interviewee): nothing to lip-sync / protect
                plan["skipped"].append({"route_track": rf.get("track"), "reason": "no MediaPipe track for this route face", "present_s": [t0, t1]}); continue
            if rf.get("mode") == "lipsync": groups.setdefault(j, []).append((rf, k0, k1))
            else: src_tracks.add(j)
        src_tracks -= set(groups)
        for j, rfs in sorted(groups.items(), key=lambda kv: -sum(k1 - k0 for _, k0, k1 in kv[1])):
            lm = np.full_like(tr[:, j], np.nan); keep = np.zeros(T, bool)
            for _, k0, k1 in rfs: keep[k0:k1] = True
            lm[keep] = tr[keep, j]; lm = bridge(hold(lm), keep)
            own = [r for rf, _, _ in rfs for r in rf.get("speak_ranges", [])]; keep = extend_keep(own, VAD)   # (O6: the own turns grown to the dub's speech)
            x, muted = silence(pcm, [r for rf, _, _ in rfs for r in rf.get("silence_ranges", [])], keep=keep)
            tag = add_face(j, lm, x, {"route_faces": [rf.get("track") for rf, _, _ in rfs], "muted_s": muted})
            if tag: others_npz(f"{a.out}/others_{tag}.npz", sorted(src_tracks | (set(groups) - {j})))
        plan["source_tracks"] = sorted(int(j) for j in src_tracks)
elif len(present) >= 2 and co >= 0.10 and units:   # no route: the dubbed units -> faces by mouth-motion correlation (r16 T v1)
    asg = cfs.assign_speakers(tr, units); mp_ = asg["mapping"]; plan["mode"] = "multi_face"; plan["assignment"] = asg
    speaking = sorted({j for j in mp_.values()}, key=lambda j: -sum(u["end"] - u["start"] for u in units if mp_.get(u["speaker"]) == j))
    for j in speaking:
        own = [u for u in units if mp_.get(u["speaker"]) == j]
        x, muted = silence(pcm, [(u["start"], u["end"]) for u in units if mp_.get(u["speaker"]) != j], keep=[(u["start"] - a.pad, u["end"] + a.pad) for u in own])
        tag = add_face(j, bridge(hold(tr[:, j])), x, {"speakers": sorted(s for s, k in mp_.items() if k == j), "units": len(own), "muted_s": muted})
        if tag: others_npz(f"{a.out}/others_{tag}.npz", [k for k in present if k != j])
elif len(present) >= 2 and co >= 0.10:   # two faces, no units: the largest face gets the dub
    j = max(present, key=lambda k: float(np.nanmean(np.ptp(tr[:, k, :, 0], axis=1))))
    plan["mode"] = "multi_face"; tag = add_face(j, bridge(hold(tr[:, j])), pcm, {"speakers": "all"})
    if tag: others_npz(f"{a.out}/others_{tag}.npz", [k for k in present if k != j])
else:
    plan["mode"] = "single_face"; add_face("largest", bridge(hold(LM1)), pcm, {"speakers": "all"})
json.dump(plan, open(f"{a.out}/faces.json", "w"), indent=1, default=str)
print("KS_FACES " + json.dumps({k: v for k, v in plan.items() if k != "assignment"} | {"scores": (plan.get("assignment") or {}).get("scores")}, default=str))
