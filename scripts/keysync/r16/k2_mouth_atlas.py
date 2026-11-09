#!/usr/bin/env python3
"""r16c K2 analysis: mouth-content defects of a lip-synced video vs its source, per frame, with the generator INPUTS next to them.
Classes (brief_r16c_analysis taxonomy): black hole, soft mouth, lip shape (eaten / swollen). Same 25 fps grid for ours and the source; MediaPipe
landmarks of each (--lm-ours / --lm-src npz from mp_landmarks.py). Per frame:
  open      inner-lip gap 13-14 / mouth width 61-291        width     mouth width / IOD
  lips      upper (0-13) + lower (14-17) lip thickness / IOD (not / mouth width: a pucker or a spread vowel changes the width)
  black     open (> 0.15) and the inner-lip hull is mean L* < 15 OR >= 40 % near-black (L* < 15), with < 5 % teeth-bright (L* > 55) pixels
  sharp     Laplacian variance of the lower lip + chin box (14 -> 152, 61 -> 291) / the nose band (6 -> 2) of the same frame
  audio     from the video's own audio: RMS dB per frame, "open" = share of 300-1000 Hz (F1 band) energy in 80-4000 Hz (open vowels high)
  ref       the generator's identity frame = the first frame of each face run (ref 0 of each ks_face segment): its open, lips, teeth share
Events (runs, gaps <= 2 frames bridged):
  black hole  ours black, the source not black at the same frame (>= 3 frames)
  soft mouth  ours sharp / the source's sharp (median over +-12 frames) < --soft (0.5) for >= 10 frames
  lip eaten   ours lips < --eaten (0.75) x the speaker's median lips (source, open frames) while ours open > 0.08, >= 5 frames
  lip swollen ours lips > --swollen (1.35) x that median, >= 5 frames
Writes <out>.json (series + events) and prints the taxonomy rows.
    python k2_mouth_atlas.py --name m18 --ours e2e.mp4 --src src25.mp4 --lm-ours ours.npz --lm-src src.npz --out m18_atlas"""
import argparse, json, subprocess, wave
import numpy as np, cv2

INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ours", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--lm-ours", required=True, dest="lmo"); ap.add_argument("--lm-src", required=True, dest="lms"); ap.add_argument("--out", required=True)
ap.add_argument("--soft", type=float, default=0.5); ap.add_argument("--eaten", type=float, default=0.75); ap.add_argument("--swollen", type=float, default=1.35)
a = ap.parse_args()


def feats(vid, lmf):
    L = np.load(lmf)["lm"][..., :2].astype(np.float32); n = len(L); c = cv2.VideoCapture(vid)
    F = {k: np.full(n, np.nan) for k in ("open", "width", "lips", "up", "lo", "black", "dark", "teeth", "sharp", "innerL")}
    for k in range(n):
        ok, f = c.read()
        if not ok: break
        p = L[k]
        if not np.isfinite(p).all(): continue
        w = float(np.linalg.norm(p[61] - p[291])); iod = float(np.linalg.norm(p[33] - p[263]))
        if w < 4 or iod < 4: continue
        F["open"][k] = np.linalg.norm(p[13] - p[14]) / w; F["width"][k] = w / iod
        F["up"][k] = np.linalg.norm(p[0] - p[13]) / iod; F["lo"][k] = np.linalg.norm(p[14] - p[17]) / iod; F["lips"][k] = F["up"][k] + F["lo"][k]   # / IOD: a pucker / spread vowel changes the width, not the lips
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo = g[int(p[14, 1]):int(p[152, 1]), int(p[61, 0]):int(p[291, 0])]; up = g[int(p[6, 1]):int(p[2, 1]), int(p[61, 0]):int(p[291, 0])]
        if lo.size > 40 and up.size > 40: F["sharp"][k] = cv2.Laplacian(lo, cv2.CV_32F).var() / max(cv2.Laplacian(up, cv2.CV_32F).var(), 1e-3)
        if F["open"][k] > 0.04:
            m = cv2.fillPoly(np.zeros(g.shape, np.uint8), [cv2.convexHull(p[INNER])[:, 0].astype(np.int32)], 1)
            Lv = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0][m > 0].astype(np.float32) * 100 / 255
            if len(Lv) > 15:
                F["innerL"][k] = Lv.mean(); F["dark"][k] = (Lv < 15).mean(); F["teeth"][k] = (Lv > 55).mean()
                F["black"][k] = float(F["open"][k] > 0.15 and (Lv.mean() < 15 or (Lv < 15).mean() > 0.40) and (Lv > 55).mean() < 0.05)   # old03 66.2 s void: mean L ~20 with 50-60 % near-black
    return F, L


O, LO = feats(a.ours, a.lmo); S, LS = feats(a.src, a.lms); n = min(len(O["open"]), len(S["open"]))
# audio of OUR video (the dub that drove the generator, mixed with the bed)
wav = a.out + ".wav"; subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", a.ours, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
w_ = wave.open(wav); x = np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float32); hop = 640
db = np.full(n, -120.0); f1 = np.zeros(n)
fr = np.fft.rfftfreq(1024, 1 / 16000); b1 = (fr >= 300) & (fr < 1000); ball = (fr >= 80) & (fr < 4000)
for k in range(n):
    s = x[k * hop:k * hop + 1024]
    if len(s) < 1024: break
    db[k] = 20 * np.log10(np.sqrt((s ** 2).mean()) + 1e-6); P = np.abs(np.fft.rfft(s * np.hanning(1024))) ** 2; f1[k] = P[b1].sum() / max(P[ball].sum(), 1e-9)
speech = db > db.max() - 30
# reference frames = first frame of each source face run
pres = np.isfinite(S["open"][:n]); refs = [k for k in range(n) if pres[k] and (k == 0 or not pres[k - 1])]
med_lips = float(np.nanmedian(S["lips"][:n][S["open"][:n] > 0.05])) if np.isfinite(S["lips"]).any() else np.nan
REF = [{"frame": r, "open": round(float(S["open"][r]), 3), "lips_rel": round(float(S["lips"][r] / med_lips), 2), "teeth": round(float(np.nan_to_num(S["teeth"][r])), 3),
        "up": round(float(S["up"][r]), 3), "lo": round(float(S["lo"][r]), 3)} for r in refs]


def ref_of(k):
    rs = [r for r in REF if r["frame"] <= k]; return rs[-1] if rs else (REF[0] if REF else None)


def runs(mask, min_len, gap=2):
    out = []; i = 0
    while i < n:
        if not mask[i]: i += 1; continue
        j = i; last = i
        while j < n and (mask[j] or (j - last) <= gap):
            if mask[j]: last = j
            j += 1
        if last - i + 1 >= min_len: out.append((i, last + 1))
        i = j
    return out


sm = lambda v, r: np.array([np.nanmedian(v[max(0, k - r):k + r + 1]) if np.isfinite(v[max(0, k - r):k + r + 1]).any() else np.nan for k in range(len(v))])
Ssh = sm(S["sharp"][:n], 12); soft_r = O["sharp"][:n] / Ssh
EV = []
def add(cls, i, j, metric, mval, extra):
    r = ref_of(i); seg = slice(i, j)
    EV.append({"video": a.name, "t0": round(i / 25, 2), "t1": round(j / 25, 2), "frames": [i, j], "class": cls, "metric": metric, "value": round(float(mval), 3),
               "ours_open": round(float(np.nanmean(O["open"][seg])), 3), "src_open": round(float(np.nanmean(S["open"][seg])), 3),
               "ours_width": round(float(np.nanmean(O["width"][seg])), 3), "src_width": round(float(np.nanmean(S["width"][seg])), 3),
               "ours_lips_rel": round(float(np.nanmean(O["lips"][seg]) / med_lips), 2), "src_dark": round(float(np.nanmean(np.nan_to_num(S["dark"][seg]))), 3),
               "audio_db": round(float(db[seg].mean()), 1), "audio_speech": round(float(speech[seg].mean()), 2), "audio_f1": round(float(f1[seg].mean()), 3),
               "ref": r} | extra)
for i, j in runs((O["black"][:n] == 1) & (S["black"][:n] != 1), 3): add("black hole", i, j, "black frames share", float(np.mean(O["black"][i:j] == 1)), {"innerL": round(float(np.nanmean(O["innerL"][i:j])), 1)})
for i, j in runs(soft_r < a.soft, 10): add("soft mouth", i, j, "sharp ours/src", float(np.nanmedian(soft_r[i:j])), {})
lr = O["lips"][:n] / med_lips
for i, j in runs((lr < a.eaten) & (O["open"][:n] > 0.08), 5): add("lip shape (eaten)", i, j, "lips / speaker median", float(np.nanmedian(lr[i:j])), {})
for i, j in runs(lr > a.swollen, 5): add("lip shape (swollen)", i, j, "lips / speaker median", float(np.nanmedian(lr[i:j])), {})
for e in EV:
    d = e["t1"] - e["t0"]; e["severity"] = 3 if d >= 1.0 else (2 if d >= 0.4 else 1)
summ = {"video": a.name, "frames": n, "speaker_median_lips": round(med_lips, 3), "refs": REF,
        "ours_black_share_open": round(float(np.nanmean(O["black"][:n][O["open"][:n] > 0.15] == 1)) if (O["open"][:n] > 0.15).any() else 0, 3),
        "src_black_share_open": round(float(np.nanmean(S["black"][:n][S["open"][:n] > 0.15] == 1)) if (S["open"][:n] > 0.15).any() else 0, 3),
        "soft_ratio_median": round(float(np.nanmedian(soft_r)), 3), "lips_rel_median_open": round(float(np.nanmedian(lr[O["open"][:n] > 0.08])), 3),
        "open_mean_ours": round(float(np.nanmean(O["open"][:n])), 3), "open_mean_src": round(float(np.nanmean(S["open"][:n])), 3)}
json.dump({"summary": summ, "events": EV, "series": {"ours_" + k: np.round(np.nan_to_num(v[:n], nan=-1), 4).tolist() for k, v in O.items()} |
           {"src_" + k: np.round(np.nan_to_num(v[:n], nan=-1), 4).tolist() for k, v in S.items()} | {"db": np.round(db, 1).tolist(), "f1": np.round(f1, 3).tolist()}},
          open(a.out + ".json", "w"))
print("ATLAS_SUMMARY " + json.dumps(summ), flush=True)
for e in sorted(EV, key=lambda e: (-e["severity"], e["t0"])):
    print(f"ROW {e['video']} | {e['t0']:.2f}-{e['t1']:.2f} | {e['class']} | {e['severity']} | {e['metric']} {e['value']} | open ours/src {e['ours_open']}/{e['src_open']}, width {e['ours_width']}/{e['src_width']}, lips_rel {e['ours_lips_rel']}, audio {e['audio_db']} dB f1 {e['audio_f1']} | ref {e['ref']}", flush=True)
