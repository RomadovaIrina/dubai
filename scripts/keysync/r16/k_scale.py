"""r16 K1: face-scale breathing per stage. Scale = mean distance of stable + lower-face landmarks to their centroid (eye corners, nose bridge,
cheeks, jaw), final / source on the master grid. Breathing = std of the band-passed ratio (remove the slow trend > 1 s), in %."""
import numpy as np, sys, json
v = sys.argv[1]; R = "/workspace/dub/keysync/r16/"; P = f"/workspace/dub/r16in/HANDOFF_K/paste_inputs/{v}/"
z = np.load(P + "source_lm_F.npz"); L = z["lm"][..., :2]; J = z["j"]; S = L[J]
SETS = {"eyes": [33, 263, 133, 362, 168, 6], "temples": [10, 338, 297, 109, 67, 103, 54, 21, 162, 127, 356, 389, 251, 284, 332], "cheeks": [234, 454, 93, 323, 132, 361, 50, 280, 205, 425], "jaw": [58, 288, 172, 397, 150, 379, 152, 148, 377], "lipsout": [61, 291, 0, 17]}
def scale(X, ids):
    p = X[:, ids]; c = p.mean(1, keepdims=True); return np.linalg.norm(p - c, axis=2).mean(1)
def band(x, w=25):
    k = np.ones(w) / w; tr = np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, "valid")[: len(x)]; return x - tr
arms = sys.argv[2:]; out = {}
for a in arms:
    F = np.load(R + a)["lm"][..., :2][: len(S)]; n = len(F)
    for nm, ids in SETS.items():
        r = scale(F, ids) / scale(S[:n], ids); r = np.where(np.isfinite(r), r, 1.0); b = band(r)
        out[f"{a.split('/')[-1].replace('.mp.npz','')}:{nm}"] = {"breath_std_pct": round(float(b.std() * 100), 3), "p95_pct": round(float(np.percentile(np.abs(b), 95) * 100), 3), "mean": round(float(r.mean()), 4)}
        np.save(R + f"scale_{v}_{a.split('/')[-1].replace('.mp.npz','')}_{nm}.npy", r)
for k, x in out.items(): print(k, x)
