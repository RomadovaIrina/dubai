#!/usr/bin/env python3
"""r16 T: the KeySync face stage's speaker -> face plan (checklist #14) + the per-face KeySync inputs.
From the 2-face MediaPipe pass of the 25 fps master (scripts/pilot/mp_landmarks.py --num-faces 2) and the dub manifest's units:
  - one face (or faces that never co-occur = one person split across tracks): KeySync on the largest face, the whole dub drives it (= the r15 K recipe);
  - >= 2 co-occurring faces: the dubbed units (LS_FACE_UNITS logic, r15 F: a speaker with no dubbed line never gets a mouth) are mapped to face tracks
    by mouth-motion correlation (clean_face_stack.assign_speakers). KeySync runs ONLY on faces that got a speaker; each one hears its own units,
    every other unit muted (so it keeps a closed mouth while someone else talks in the dub). Faces with no speaker are left untouched and are
    subtracted from the paste mask ('others', ks_plain --others) of every face that is pasted.
  - KS_DUB_DELAY_MS (default 0): the K1 hook - the dub fed to KeySync is delayed by N ms (zeros in front, same length); the final mix is never touched.
Writes OUT/faces.json + per face: lm_<tag>.npz (lm, pixels, short gaps held), others_<tag>.npz, dub_<tag>.wav (16 kHz mono)."""
import argparse, glob, json, os, sys, wave
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../pilot"))
import clean_face_stack as cfs

ap = argparse.ArgumentParser()
ap.add_argument("--mp", required=True, help="mp_landmarks.py --num-faces 2 npz of the 25 fps master")
ap.add_argument("--dub", required=True, help="dubbed_16k.wav (the dub voice)"); ap.add_argument("--manifest", default="")
ap.add_argument("--out", required=True); ap.add_argument("--pad", type=float, default=0.08, help="s kept around a face's own units when muting the others")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
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


with wave.open(a.dub) as w:
    sr = w.getframerate(); assert w.getnchannels() == 1 and w.getsampwidth() == 2, "dub must be 16-bit mono"
    pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).copy()
units = []
man = a.manifest or next(iter(sorted(glob.glob(os.path.join(os.path.dirname(a.dub), "*.manifest.json")))), "")
if man and os.path.exists(man):
    m = json.load(open(man))
    units = [{"speaker": u["speaker"], "start": float(u["start"]), "end": float(u["end"])} for u in (m.get("units") or [])
             if u.get("speaker") is not None and float(u.get("end", 0)) > float(u.get("start", 0))]
tr, _ = cfs.track_faces(LA)
present = [j for j in range(tr.shape[1]) if np.isfinite(tr[:, j, 0, 0]).mean() >= 0.1]
both = int((np.isfinite(LA[:, 0, 0, 0]) & np.isfinite(LA[:, 1, 0, 0])).sum()) if LA.shape[1] >= 2 else 0
co = both / max(int(np.isfinite(LA[:, :, 0, 0]).any(1).sum()), 1)
plan = {"frames": T, "tracks": len(present), "co_occurrence": round(co, 4), "units": len(units), "manifest": man,
        "dub_delay_ms": int(os.environ.get("KS_DUB_DELAY_MS", "0") or 0), "faces": []}


def write_wav(path, x):
    d = plan["dub_delay_ms"] * sr // 1000
    if d > 0: x = np.concatenate([np.zeros(d, np.int16), x[:len(x) - d]])   # K1 hook: the generator input only
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.astype(np.int16).tobytes())


def others_npz(path, idx):
    o = np.full((T, 2, 478, 2), np.nan, np.float32)
    for q, j in enumerate(idx[:2]): o[:, q] = tr[:, j]
    np.savez_compressed(path, others=o)


if len(present) >= 2 and co >= 0.10 and units:
    asg = cfs.assign_speakers(tr, units); mp_ = asg["mapping"]; plan["mode"] = "multi_face"; plan["assignment"] = asg
    speaking = sorted({j for j in mp_.values()}, key=lambda j: -sum(u["end"] - u["start"] for u in units if mp_.get(u["speaker"]) == j))
    plan["unmapped_speakers"] = sorted({u["speaker"] for u in units} - set(mp_))   # e.g. an off-screen voice: muted for every face
    for q, j in enumerate(speaking):
        tag = f"f{q}"; own = [u for u in units if mp_.get(u["speaker"]) == j]; x = pcm.copy()
        keep = np.zeros(len(x), bool)
        for u in own: keep[max(0, int((u["start"] - a.pad) * sr)):int((u["end"] + a.pad) * sr)] = True
        muted = 0.0
        for u in units:
            if mp_.get(u["speaker"]) == j: continue
            s0, s1 = int(u["start"] * sr), int(u["end"] * sr); seg = ~keep[s0:s1]; x[s0:s1][seg] = 0; muted += seg.sum() / sr
        np.savez_compressed(f"{a.out}/lm_{tag}.npz", lm=hold(tr[:, j])); others_npz(f"{a.out}/others_{tag}.npz", [k for k in present if k != j])
        write_wav(f"{a.out}/dub_{tag}.wav", x)
        plan["faces"].append({"tag": tag, "track": int(j), "speakers": sorted(s for s, k in mp_.items() if k == j), "units": len(own),
                              "muted_s": round(muted, 2), "presence": round(float(np.isfinite(tr[:, j, 0, 0]).mean()), 3)})
    plan["untouched_tracks"] = [int(j) for j in present if j not in speaking]
else:
    plan["mode"] = "single_face"; plan["note"] = ("faces never co-occur: one person split across tracks" if len(present) >= 2 and co < 0.10
                                                  else "no dubbed units with speakers" if len(present) >= 2 else "one face")
    np.savez_compressed(f"{a.out}/lm_f0.npz", lm=hold(LM1)); write_wav(f"{a.out}/dub_f0.wav", pcm)
    plan["faces"].append({"tag": "f0", "track": "largest", "speakers": "all", "presence": round(float(np.isfinite(LM1[:, 0, 0]).mean()), 3)})
json.dump(plan, open(f"{a.out}/faces.json", "w"), indent=1, default=str)
print("KS_FACES " + json.dumps({k: v for k, v in plan.items() if k != "assignment"} | {"scores": (plan.get("assignment") or {}).get("scores")}, default=str))
