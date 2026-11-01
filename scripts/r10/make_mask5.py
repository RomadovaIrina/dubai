#!/usr/bin/env python3
# r11 FACE: writes latentsync/utils/mask5.png (LS_MASK=mask5.png) = the production mask.png cut at the chin bottom (crop row 435) and the jaw sides (cols 78-435):
# the source chin / jaw outline stay, the top (row 190) is unchanged. third_party/ is not in git: run once per box.
#   python scripts/r10/make_mask5.py third_party/latentsync/latentsync/utils
import cv2, numpy as np, sys
d = sys.argv[1]
m = cv2.imread(f"{d}/mask.png", 0); h, w = m.shape; s = 512 / h
gen = m < 128                                  # regenerated = dark
yy, xx = np.mgrid[0:h, 0:w]
keep_chin = (yy * s > 435) | (xx * s < 78) | (xx * s > 435)   # mask3's bottom / sides: the source chin and jaw sides stay
gen5 = gen & ~keep_chin
out = np.where(gen5, 0, 255).astype(np.uint8); cv2.imwrite(f"{d}/mask5.png", out)
r = cv2.resize(out, (512, 512)) < 128; ys, xs = np.nonzero(r); print("mask5 regenerated rows", ys.min(), ys.max(), "cols", xs.min(), xs.max(), "frac %.3f" % r.mean())
