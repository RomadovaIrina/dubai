import sys, numpy as np
# per-frame inner-lip opening (gap 13-14 / width 61-291) and outer-lip height (0-17 / width) from a MediaPipe npz
for f in sys.argv[1:]:
    L = np.load(f)["lm"][..., :2]; w = np.linalg.norm(L[:, 61] - L[:, 291], axis=1)
    gap = np.linalg.norm(L[:, 13] - L[:, 14], axis=1) / w; outer = np.linalg.norm(L[:, 0] - L[:, 17], axis=1) / w
    k0, k1 = 237, 263   # 9.5-10.5 s
    print(f"{f:28s} open 9.5-10.5s mean {np.nanmean(gap[k0:k1]):.3f} max {np.nanmax(gap[k0:k1]):.3f} | outer {np.nanmean(outer[k0:k1]):.3f} | whole: open mean {np.nanmean(gap):.3f} p90 {np.nanpercentile(gap,90):.3f}")
