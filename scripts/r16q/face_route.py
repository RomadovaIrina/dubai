#!/usr/bin/env python3
"""r16 Q: DUB_FACE=auto routing = active speaker (r16q_active_speaker.py) + beard (r16q_beard.py) -> route.json.

  keep_source_ranges: [[t0, t1], ...] seconds where diarization has speech but NO visible face speaks it in the source
                      (off-screen voice, a silent listener on screen: q02). The paste must show the SOURCE face there
                      (no lip-sync); the dub audio still plays. Merged, padded by PAD s, minus the on-screen turns.
  keep_source_all:    no turn has a visible speaker -> skip the face stage entirely (voice-only result).
  lipsync_track:      the SyncNet / S3FD track that speaks most (box centre given, to match MediaPipe tracks by centre).
  face_engine:        latentsync if that face has a beard / stubble (KeySync is weak on beards), else keysync.
                      Override: DUB_FACE_ENGINE=keysync|latentsync. Unmeasured beard (dark / small / profile) -> keysync.
Usage: python3 face_route.py --active as.json --beard beard.json --out route.json"""
import argparse, json, os

PAD = float(os.environ.get("ROUTE_PAD", "0.12"))
ap = argparse.ArgumentParser(); ap.add_argument("--active", required=True); ap.add_argument("--beard", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
A = json.load(open(a.active)); B = json.load(open(a.beard))


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out


def minus(iv, cut):
    res = []
    for s, e in iv:
        parts = [[s, e]]
        for cs, ce in cut:
            nxt = []
            for ps, pe in parts:
                if ce <= ps or cs >= pe: nxt.append([ps, pe]); continue
                if cs > ps: nxt.append([ps, cs])
                if ce < pe: nxt.append([ce, pe])
            parts = nxt
        res += parts
    return [[round(s, 3), round(e, 3)] for s, e in res if e - s >= 0.2]


on = merge([[t["start"], t["end"]] for t in A["turns"] if t["on_screen"]])
off = merge([[max(0.0, t["start"] - PAD), t["end"] + PAD] for t in A["turns"] if not t["on_screen"]])
keep = minus(off, on)
talk = {}
for t in A["turns"]:
    if t["on_screen"] and t.get("track") is not None: talk[t["track"]] = talk.get(t["track"], 0.0) + t["end"] - t["start"]
main = max(talk, key=talk.get) if talk else None
tr = next((x for x in A["tracks"] if x["id"] == main), None)
face = None
if tr and tr.get("box_center_xy") and B.get("faces"):
    cx, cy = tr["box_center_xy"]
    face = min(B["faces"], key=lambda f: (f["center_xy"][0] - cx) ** 2 + (f["center_xy"][1] - cy) ** 2)
    if ((face["center_xy"][0] - cx) ** 2 + (face["center_xy"][1] - cy) ** 2) ** 0.5 > 2.5 * (tr.get("box_size") or 1e9): face = None
eng = os.environ.get("DUB_FACE_ENGINE") or (face["face_engine"] if face else "keysync")
route = {"keep_source_all": bool(A["hook"]["keep_source_all"] or main is None), "keep_source_ranges": keep,
         "keep_source_s": round(sum(e - s for s, e in keep), 2), "lipsync_track": main,
         "lipsync_box_center_xy": tr.get("box_center_xy") if tr else None, "lipsync_speech_s": round(talk.get(main, 0.0), 2),
         "face_engine": eng, "beard": None if face is None else {k: face.get(k) for k in ("beard", "speck", "dark", "texture_reliable")},
         "why": ("override" if os.environ.get("DUB_FACE_ENGINE") else "beard/stubble -> latentsync" if eng == "latentsync"
                 else "no beard" if face else "beard unmeasured -> keysync default")}
json.dump(route, open(a.out, "w"), indent=1)
print("ROUTE " + json.dumps(route))
