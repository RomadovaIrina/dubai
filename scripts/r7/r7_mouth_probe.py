#!/usr/bin/env python3
"""r7 F: two user complaints on the m8 WATCH, measured per variant and video.
 (1) 'glued / pressed lips' where the dub is silent: over dub-silent frames (energy VAD on the variant's dubbed track) with a face, the mouth
     height (outer lips 0-17 / IOD), the lip thickness (0-13 + 14-17 / IOD) and the inner opening (13-14 / IOD) from the variant's OWN
     output landmarks (<id>_<tag>_out_mp.npz), against the source's resting closed mouth (source frames with opening < 0.02 IOD).
 (2) 'black mouth inside': over open-mouth frames (opening > 0.08 IOD), the mean luma inside the inner-lip polygon and the teeth fraction
     (luma > 140), source vs variants.
    python scripts/r7/r7_mouth_probe.py --ids 03 01 05 02 04 --variants m5:AUDIO6 m6:AUDIO6 m7:AUDIO6 m8:AUDIO7 --sheet-dir DIR
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

B = pathlib.Path("/workspace/dub/artifacts/r4m"); SRC_LM = pathlib.Path("/workspace/dub/artifacts/r7f/lm")
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def silent_spans(wav, fps, n, thr_db=-45.0, min_s=0.3):
    import soundfile as sf
    w, sr = sf.read(wav);
    if w.ndim > 1: w = w.mean(1)
    hop = int(sr * 0.02); e = np.array([np.sqrt(np.mean(w[i:i + hop] ** 2) + 1e-12) for i in range(0, len(w) - hop, hop)]); db = 20 * np.log10(e + 1e-9)
    sil = db < thr_db; spans = []; i = 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]: j += 1
            if (j - i) * 0.02 >= min_s: spans.append((i * 0.02, j * 0.02))
            i = j
        else: i += 1
    mask = np.zeros(n, bool)
    for a, b in spans: mask[int(a * fps):min(n, int(b * fps) + 1)] = True
    return spans, mask


def geo(lm):
    iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1); iod[iod < 1] = np.nan
    opening = np.linalg.norm(lm[:, 13] - lm[:, 14], axis=1) / iod; height = np.linalg.norm(lm[:, 0] - lm[:, 17], axis=1) / iod
    thick = (np.linalg.norm(lm[:, 0] - lm[:, 13], axis=1) + np.linalg.norm(lm[:, 14] - lm[:, 17], axis=1)) / iod
    width = np.linalg.norm(lm[:, 61] - lm[:, 291], axis=1) / iod
    return opening, height, thick, width


def interior(frame, p):
    import cv2
    q = p[LIPS_INNER].astype(np.int32); m = np.zeros(frame.shape[:2], np.uint8); cv2.fillPoly(m, [q], 1)
    if m.sum() < 12: return None
    y = frame[..., 2] * 0.299 + frame[..., 1] * 0.587 + frame[..., 0] * 0.114; v = y[m > 0]
    return float(v.mean()), float((v > 140).mean()), int(m.sum())


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=["03", "01", "05", "02", "04"]); ap.add_argument("--variants", nargs="+", required=True, help="tag:audio_handoff_dir")
    ap.add_argument("--sheet-dir", default="/workspace/dub/artifacts/r7f/mouth_probe"); ap.add_argument("--out", default="/workspace/dub/artifacts/r7f/mouth_probe/mouth_probe.json")
    a = ap.parse_args(); sd = pathlib.Path(a.sheet_dir); sd.mkdir(parents=True, exist_ok=True); R = {}
    for id_ in a.ids:
        src_lm = np.load(SRC_LM / f"{id_}_src_mp.npz")["lm"]; n = len(src_lm); so, sh, st, sw = geo(src_lm)
        m8 = json.loads((B / f"m8/{id_}_m8.manifest.json").read_text()); src = m8["input"]; fps = n / m8["source"]["duration_s"]
        rest = np.isfinite(so) & (so < 0.02); R[id_] = {"source_rest_frames": int(rest.sum()), "source_rest_height": float(np.nanmedian(sh[rest])) if rest.any() else None,
                                                       "source_rest_thick": float(np.nanmedian(st[rest])) if rest.any() else None, "source_rest_width": float(np.nanmedian(sw[rest])) if rest.any() else None}
        # source interior on its own open frames
        def sample_interior(video, lm, frames):
            c = cv2.VideoCapture(str(video)); vals = []; crops = []
            for k in frames:
                c.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = c.read()
                if not ok or not np.isfinite(lm[k][LIPS_INNER + [33, 263]]).all(): continue
                r = interior(f, lm[k])
                if r is None: continue
                vals.append(r[:2]); p = lm[k]; cx, cy = int(p[LIPS_OUTER, 0].mean()), int(p[LIPS_OUTER, 1].mean()); rr = int(max(24, 0.9 * np.linalg.norm(p[33] - p[263])))
                if len(crops) < 12: crops.append(cv2.resize(f[max(0, cy - rr):cy + rr, max(0, cx - rr):cx + rr], (120, 120), interpolation=cv2.INTER_CUBIC))
            c.release(); v = np.array(vals) if vals else np.zeros((0, 2))
            return (float(v[:, 0].mean()) if len(v) else None, float(v[:, 1].mean()) if len(v) else None, len(v)), crops
        open_src = np.flatnonzero(np.isfinite(so) & (so > 0.08))[::5][:80]
        (lum, teeth, nn), crops_src = sample_interior(src, src_lm, open_src); R[id_]["source"] = {"interior_luma": lum, "teeth_frac": teeth, "n_open": nn}
        sheet_rows = [("src", crops_src)]
        for var in a.variants:
            tag, aud = var.split(":"); d = B / tag; vid = d / f"{id_}_{tag}.mp4"; lmp = d / f"{id_}_{tag}_out_mp.npz"
            if not vid.exists() or not lmp.exists(): R[id_][tag] = None; continue
            lm = np.load(lmp)["lm"]; o, h, t, w = geo(lm); wav = pathlib.Path(aud) / id_ / "dubbed_16k.wav"
            spans, sil = silent_spans(wav, fps, len(lm)); face = np.isfinite(o)
            vis = sil & face; src_talk = vis & np.isfinite(so) & (so > 0.05)   # the user's case: the source talks, the dub is silent
            r = {"silent_spans": [[round(x, 2), round(y, 2)] for x, y in spans], "silent_face_frames": int(vis.sum()), "silent_src_talking_frames": int(src_talk.sum())}
            for lab, mk in (("silent", vis), ("silent_src_talking", src_talk)):
                if mk.any():
                    r[lab] = {"height_med": float(np.nanmedian(h[mk])), "thick_med": float(np.nanmedian(t[mk])), "opening_med": float(np.nanmedian(o[mk])), "width_med": float(np.nanmedian(w[mk])),
                              "thick_p10": float(np.nanpercentile(t[mk], 10)), "closed_frac": float(np.mean(o[mk] < 0.02)),
                              "height_vs_rest": float(np.nanmedian(h[mk]) / R[id_]["source_rest_height"]) if R[id_]["source_rest_height"] else None,
                              "thick_vs_rest": float(np.nanmedian(t[mk]) / R[id_]["source_rest_thick"]) if R[id_]["source_rest_thick"] else None}
            open_v = np.flatnonzero(np.isfinite(o) & (o > 0.08))[::5][:80]; (lum, teeth, nn), crops = sample_interior(vid, lm, open_v)
            r["interior_luma"] = lum; r["teeth_frac"] = teeth; r["n_open"] = nn; R[id_][tag] = r; sheet_rows.append((tag, crops))
            print(f"== {id_} {tag}: silent face frames {vis.sum()} (src talking {src_talk.sum()}), thick_vs_rest {r.get('silent', {}).get('thick_vs_rest')}, height_vs_rest {r.get('silent', {}).get('height_vs_rest')}, "
                  f"interior luma {lum} (src {R[id_]['source']['interior_luma']}), teeth {teeth} (src {R[id_]['source']['teeth_frac']})", flush=True)
        rows = []
        for lab, crops in sheet_rows:
            if not crops: continue
            row = np.hstack(crops + [np.zeros((120, 120, 3), np.uint8)] * (12 - len(crops))); cv2.putText(row, lab, (4, 16), 0, 0.5, (0, 255, 255), 1); rows.append(row)
        if rows: cv2.imwrite(str(sd / f"{id_}_open_mouth_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    pathlib.Path(a.out).write_text(json.dumps(R, indent=1)); print("== written", a.out)


if __name__ == "__main__":
    main()
