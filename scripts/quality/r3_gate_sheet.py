#!/usr/bin/env python3
"""Round 3 D: acceptance-gate contact sheet: the top-N flash-detector frames (r3_flash.py JSON, by mc_ratio) plus any --frames, each as a
2-row tile (source on top, output below) of frames k-1 / k / k+1, mouth crop scaled 2x.
    python scripts/quality/r3_gate_sheet.py --video OUT.mp4 --src SRC.mp4 --lm SRC_lm.npz --flash OUT.flash.json --out sheet.png [--top 30] [--frames 710 774]"""
import argparse, json, sys, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import r3_flash as rf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--lm", required=True)
    ap.add_argument("--flash", required=True); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=30); ap.add_argument("--frames", type=int, nargs="*", default=[])
    a = ap.parse_args(); lm = np.load(a.lm)["lm"]; fl = json.loads(pathlib.Path(a.flash).read_text())["flash_list"]
    ks = [e["frame"] for e in sorted(fl, key=lambda e: -(e.get("mc_ratio") or 0))[:a.top]]
    ks = sorted(set(ks) | set(a.frames))
    need = sorted({x for k in ks for x in (k - 1, k, k + 1)})
    import av
    def grab(p):
        c = av.open(p); out = {}
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i > need[-1]: break
            if i in need: out[i] = f.to_ndarray(format="bgr24")
        c.close(); return out
    O, S = grab(a.video), grab(a.src); tiles = []
    for k in ks:
        if not np.isfinite(lm[k][rf.LIPS]).all(): continue
        q = lm[k][rf.LIPS, :2]; c = q.mean(0); s = int((q[:, 0].max() - q[:, 0].min()) * 0.9)
        rows = []
        for src in (S, O):
            t = []
            for x in (k - 1, k, k + 1):
                f = src.get(x)
                if f is None: t.append(np.zeros((84, 120, 3), np.uint8)); continue
                cr = cv2.resize(f[max(0, int(c[1] - s * 0.7)):int(c[1] + s * 0.7), max(0, int(c[0] - s)):int(c[0] + s)], (120, 84))
                t.append(cr)
            rows.append(np.hstack(t))
        tile = np.vstack(rows); cv2.putText(tile, str(k), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    cols = 6; rows_ = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    cv2.imwrite(a.out, np.vstack(rows_)); print(a.out, len(tiles), "tiles")


if __name__ == "__main__":
    main()
