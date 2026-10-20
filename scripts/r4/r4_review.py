#!/usr/bin/env python3
"""Round 4 review sheet: outbox/<id>_review.md, one row per 10 s, so the user / Max only confirm or refute. Per window, NEW vs PREV (same frame
grid): where the face changed (mean |new - prev| in the lower face, grey levels; ~0.5 = encoder noise), lip-flash candidates of each, the worst
half-face colour shift and the below-jaw band (chin) of each, and the audio (identical / changed RMS dB).
    python scripts/r4/r4_review.py --id 04 --new NEW.mp4 --prev PREV.mp4 --new-score N.score.json --prev-score P.score.json --lm SRC_lm.npz
        --title "r4_video_v1 vs v4" --out outbox/04_review.md"""
from __future__ import annotations
import argparse, json, pathlib, subprocess
import numpy as np

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]


def audio_rms(p, sr=16000):
    x = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(x, np.float32)


def main():
    ap = argparse.ArgumentParser()
    for k in ("id", "new", "prev", "new-score", "prev-score", "lm", "title", "out"): ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    import av, cv2
    lm = np.load(a.lm)["lm"]; cn, cp = av.open(a.new), av.open(a.prev); fps = float(cn.streams.video[0].average_rate)
    diff = []
    for k, (fn, fp) in enumerate(zip(cn.decode(video=0), cp.decode(video=0))):
        if k >= len(lm) or not np.isfinite(lm[k, :, :2]).all(): diff.append(np.nan); continue
        n, p = fn.to_ndarray(format="gray").astype(np.float32), fp.to_ndarray(format="gray").astype(np.float32); q = lm[k, :, :2]
        m = np.zeros(n.shape, np.uint8); cv2.fillPoly(m, [np.round(q[FACE_OVAL]).astype(np.int32)], 1); m[: int(q[1, 1])] = 0
        diff.append(float(np.abs(n - p)[m.astype(bool)].mean()) if m.any() else np.nan)
    cn.close(); cp.close(); diff = np.array(diff)
    def flashes(score):
        f = pathlib.Path(score).with_name(pathlib.Path(score).name.replace(".score.json", ".flash.json"))
        return [e["frame"] for e in json.loads(f.read_text())["flash_list"] if (e.get("mc_ratio") or 0) >= 2 and (e.get("mc_vs_src") or 0) >= 2] if f.exists() else []
    def face(score):
        f = pathlib.Path(score).with_name(pathlib.Path(score).name.replace(".score.json", ".r4face.json")); return json.loads(f.read_text()) if f.exists() else {}
    fN, fP = flashes(a.new_score), flashes(a.prev_score)
    an, ap_ = audio_rms(a.new), audio_rms(a.prev); same_audio = len(an) == len(ap_) and np.allclose(an, ap_, atol=1e-4)
    T = len(diff) / fps; L = [f"# {a.id} review: {a.title}\n", "Rows of 10 s. **face Δ** = mean |NEW − PREV| on the lower face (grey levels, ≈0.5 = encoder noise; "
         "> 2 = visibly different mouth / skin there). Flash = strong single-frame lip spikes (lower is better). Fill the last columns: ✓ / ✗ / comment.\n",
         f"Audio: {'identical to PREV' if same_audio else 'CHANGED vs PREV (see the audio agent sheet)'}.\n",
         "| t (s) | face Δ mean / max | flashes NEW / PREV | look at | voice | background | pauses | mouth | face | other |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t0 in np.arange(0, T, 10.0):
        a0, a1 = int(t0 * fps), int(min(T, t0 + 10) * fps); d = diff[a0:a1]
        nN = [k for k in fN if a0 <= k < a1]; nP = [k for k in fP if a0 <= k < a1]
        look = []
        if np.isfinite(d).any() and np.nanmax(d) > 2: look.append(f"mouth differs most at {(a0 + int(np.nanargmax(d))) / fps:.1f} s")
        if len(nN) > len(nP): look.append("NEW flashes at " + ", ".join(f"{k / fps:.2f}" for k in nN[:4]))
        if len(nN) < len(nP): look.append("PREV flashes gone at " + ", ".join(f"{k / fps:.2f}" for k in nP[:4]))
        dm = f"{np.nanmean(d):.2f} / {np.nanmax(d):.2f}" if np.isfinite(d).any() else "no face"
        L.append(f"| {t0:.0f}–{min(T, t0 + 10):.0f} | {dm} | {len(nN)} / {len(nP)} | {'; '.join(look) or '–'} |  |  |  |  |  |  |")
    fn_, fp_ = face(a.new_score), face(a.prev_score)
    if fn_ and fp_:
        L += ["", f"Whole video: worst 1 s half-face colour shift NEW {fn_.get('colour_worst_1s_abs_dL')} / PREV {fp_.get('colour_worst_1s_abs_dL')}; "
              f"jaw jitter (out / source) NEW {(fn_.get('jaw_jitter') or {}).get('ratio_out_over_src')} / PREV {(fp_.get('jaw_jitter') or {}).get('ratio_out_over_src')}; "
              f"below-jaw band NEW {fn_['below_jaw']['mean']} / PREV {fp_['below_jaw']['mean']}."]
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print(a.out)


if __name__ == "__main__":
    main()
