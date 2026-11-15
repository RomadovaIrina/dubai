import sys, numpy as np
# jaw.py <job> ... : drawn-vs-source jaw contour offset (px in the 512 crop, mediapipe jaw points), per second around the turn and overall
JAW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
for j in sys.argv[1:]:
    D = "/r18/data/jobs/%s/e2e/work/ks/" % j; s = np.load(D + "crop_lm_f0.npz")["lm"][..., :2]; r = np.load(D + "post_f0/raw_lm.npz")["lm"][..., :2]; n = min(len(s), len(r), 1348)
    d = np.linalg.norm(r[:n][:, JAW] - s[:n][:, JAW], axis=2).mean(1); dl = np.linalg.norm(s[:n, 1] - s[:n, 234], axis=1); dr = np.linalg.norm(s[:n, 454] - s[:n, 1], axis=1); yaw = (dl - dr) / (dl + dr)
    ok = np.isfinite(d); t = yaw > 0.4
    print("%-6s jaw offset px: all med %.1f p95 %.1f | yaw > 0.4 (%d fr) med %.1f p95 %.1f | by second 26..31: %s" % (j, np.median(d[ok]), np.percentile(d[ok], 95), int((t & ok).sum()), np.median(d[t & ok]), np.percentile(d[t & ok], 95), [round(float(np.nanmedian(d[k * 25:(k + 1) * 25])), 1) for k in range(26, 32)]))
