#!/usr/bin/env python3
"""r15 Z: LS_CFG_TSMOOTH=a (default 0 = off) in ls_stable._cfg and bon._ls_cfg: the CFG AMPLIFICATION term (g - 1)(na - nu) smoothed across
neighbouring frames [a, 1 - 2a, a] (edges replicated), the conditional prediction kept exact: out = na + (g - 1) * smooth(na - nu).
A frame-to-frame snap (12.5 Hz at a = 0.25) is removed from the guidance, syllable-rate mouth motion (3-5 Hz) mostly kept. Idempotent."""
import pathlib, sys
R = pathlib.Path(__file__).resolve().parents[2]
for path, fn in (("scripts/quality/ls_stable.py", "_cfg"), ("scripts/r15/bon.py", "_ls_cfg")):
    p = R / path; s = p.read_text()
    if "LS_CFG_TSMOOTH" in s: print(f"  = {path}: already"); continue
    a1 = '_CFG_RESCALE = float(os.environ.get("LS_CFG_RESCALE", "0") or 0)\n'
    a2 = "    out = nu + g * (na - nu)\n    if _CFG_RESCALE <= 0: return out\n"
    if s.count(a1) != 1 or s.count(a2) != 1: sys.exit(f"{path}: anchors {s.count(a1)} {s.count(a2)}")
    s = s.replace(a1, a1 + '_CFG_TSMOOTH = float(os.environ.get("LS_CFG_TSMOOTH", "0") or 0)   # r15 Z: temporal smoothing of the CFG amplification term\n')
    s = s.replace(a2, "    out = nu + g * (na - nu)\n"
                      "    if _CFG_TSMOOTH > 0 and out.ndim == 5 and out.shape[2] >= 3:   # (b, c, f, h, w): smooth (na - nu) over f, keep na exact\n"
                      "        d = (na - nu).float(); dp = torch.cat([d[:, :, :1], d[:, :, :-1]], 2); dn = torch.cat([d[:, :, 1:], d[:, :, -1:]], 2)\n"
                      "        out = (na.float() + (g - 1) * ((1 - 2 * _CFG_TSMOOTH) * d + _CFG_TSMOOTH * (dp + dn))).to(out.dtype)\n"
                      "    if _CFG_RESCALE <= 0: return out\n")
    p.write_text(s); print(f"  + {path}")
print("PATCH_TSMOOTH_OK")
