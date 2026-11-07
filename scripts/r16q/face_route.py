#!/usr/bin/env python3
"""r16 Q: DUB_FACE=auto routing = active speaker (r16q_active_speaker.py) + beard (r16q_beard.py) -> route.json.

  faces[]:            per S3FD track (match to MediaPipe tracks by box_center_xy):
                      mode "lipsync" = the face speaks some turn in the source -> lip-synced for its whole presence, its generator
                        input = the dub with SILENCE over silence_ranges (speech that is not its own, padded), so the mouth stays
                        closed and the face never switches between generated and source pixels (Irina: no slideshow morph);
                      mode "source" = never speaks in the shot -> no lip-sync, keep the source, subtract from the paste masks.
  offscreen_speech_ranges: info - speech no visible face speaks (q02 narrator 7.1-40.3 s).
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
# Irina 07.11: NO lip-sync <-> source pixel switches on a visible face (a fade reads as a slideshow morph).
#  - a face that speaks somewhere in the shot = "lipsync" for its WHOLE presence; its generator input is SILENCE outside its own
#    turns (silence_ranges), so it stays continuous with a closed mouth while the narrator / someone else talks;
#  - a face that never speaks = "source": no lip-sync at all, subtracted from the paste masks (m03 child, q07 reporters).
allspeech = merge([[t["start"], t["end"]] for t in A["turns"]])
TR = {x["id"]: x for x in A["tracks"]}
span = {i: (x["f0"] / 25.0, (x["f0"] + x["frames"] + 4) / 25.0) for i, x in TR.items()}
spk_of = {}
for t in A["turns"]:
    if t["on_screen"] and t.get("track") is not None: spk_of.setdefault(t["track"], set()).add(t["speaker"])


def clip(iv, t0, t1):
    return [[round(max(s, t0), 3), round(min(e, t1), 3)] for s, e in iv if e > t0 and s < t1 and min(e, t1) - max(s, t0) >= 0.1]


def same_person(i, j):
    """S3FD breaks one person into tracks at detection gaps (dark / profile / a cup): adjacent in time (< 0.5 s) and the box nearby."""
    (a0, a1), (b0, b1) = span[i], span[j]
    if not (abs(b0 - a1) < 0.5 or abs(a0 - b1) < 0.5): return False
    ci, cj = TR[i].get("box_center_xy"), TR[j].get("box_center_xy")
    if not ci or not cj: return False
    return ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5 < 1.0 * max(TR[i].get("box_size") or 0, TR[j].get("box_size") or 0)


# a non-speaking piece next to a speaking piece of the same person inherits that person's speakers (repeat: chains of pieces)
for _ in range(len(TR)):
    changed = False
    for i in TR:
        if i in spk_of: continue
        nb = [j for j in spk_of if same_person(i, j)]
        if nb: spk_of[i] = set().union(*(spk_of[j] for j in nb)); changed = True
    if not changed: break
faces = []
for x in A["tracks"]:
    t0, t1 = span[x["id"]]
    conc = {k for k in TR if k != x["id"] and min(span[k][1], t1) - max(span[k][0], t0) > 0.2}   # other faces visible at the same time
    own = merge([[t["start"], t["end"]] for t in A["turns"] if t["on_screen"] and (t.get("track") == x["id"] or (
        t["speaker"] in spk_of.get(x["id"], ()) and t.get("track") not in conc))])
    own = clip(own, t0, t1)
    if own:
        sil = minus(merge([[max(t0, s - PAD), min(t1, e + PAD)] for s, e in allspeech if e > t0 and s < t1]), own)
        faces.append({"track": x["id"], "mode": "lipsync", "present_s": [round(t0, 2), round(t1, 2)], "box_center_xy": x.get("box_center_xy"),
                      "speak_ranges": own, "silence_ranges": sil, "silence_s": round(sum(e - s for s, e in sil), 2)})
    else:
        faces.append({"track": x["id"], "mode": "source", "present_s": [round(t0, 2), round(t1, 2)], "box_center_xy": x.get("box_center_xy"),
                      "conf_whole": x.get("conf_whole")})
route = {"keep_source_all": bool(A["hook"]["keep_source_all"] or main is None), "faces": faces,
         "offscreen_speech_ranges": keep, "offscreen_speech_s": round(sum(e - s for s, e in keep), 2), "lipsync_track": main,
         "lipsync_box_center_xy": tr.get("box_center_xy") if tr else None, "lipsync_speech_s": round(talk.get(main, 0.0), 2),
         "face_engine": eng, "beard": None if face is None else {k: face.get(k) for k in ("beard", "speck", "dark", "texture_reliable")},
         "why": ("override" if os.environ.get("DUB_FACE_ENGINE") else "beard/stubble -> latentsync" if eng == "latentsync"
                 else "no beard" if face else "beard unmeasured -> keysync default")}
json.dump(route, open(a.out, "w"), indent=1)
print("ROUTE " + json.dumps(route))
