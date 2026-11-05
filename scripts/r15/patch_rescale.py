#!/usr/bin/env python3
"""r15 Z: LS_CFG_RESCALE=phi (default 0 = off) at every classifier-free-guidance combine of the teacher (ls_stable windows / plain / bridges,
r15 bon.py continue paths): Lin et al. 2023 guidance rescale, per frame. Idempotent."""
import pathlib, sys
R = pathlib.Path(__file__).resolve().parents[2]


def sub(path, old, new, count=1):
    p = R / path; s = p.read_text()
    if new in s: print(f"  = {path}: already"); return
    n = s.count(old)
    if n != count: sys.exit(f"{path}: anchor {n}x (expected {count}): {old[:70]!r}")
    p.write_text(s.replace(old, new)); print(f"  + {path} ({n}x)")


HELPER = ('\n\n_CFG_RESCALE = float(os.environ.get("LS_CFG_RESCALE", "0") or 0)\n\n\n'
          'def _cfg(nu, na, g):\n'
          '    """classifier-free guidance nu + g (na - nu). r15 Z LS_CFG_RESCALE=phi (default 0 = off): Lin et al. 2023 guidance rescale PER FRAME:\n'
          '    the guided prediction\'s std over (C, H, W) of each frame is matched to the conditional prediction\'s and blended by phi. The guidance\n'
          '    direction (mouth shape / sync) is kept; its per-frame brightness / colour overshoot (the lip "blinks", which CFG drives) is not."""\n'
          '    out = nu + g * (na - nu)\n'
          '    if _CFG_RESCALE <= 0: return out\n'
          '    dims = (1, 3, 4) if out.ndim == 5 else tuple(range(1, out.ndim)); o = out.float()\n'
          '    s_c = na.float().std(dim=dims, keepdim=True); s_o = o.std(dim=dims, keepdim=True)\n'
          '    return (_CFG_RESCALE * o * (s_c / s_o.clamp_min(1e-6)) + (1 - _CFG_RESCALE) * o).to(out.dtype)\n\n\n'
          'def _dc_half(dc):\n')
sub("scripts/quality/ls_stable.py", "\n\ndef _dc_half(dc):\n", HELPER)
sub("scripts/quality/ls_stable.py", "nu, na = npred.chunk(2); dl_ = na - nu; npred = nu + self.guidance * dl_", "nu, na = npred.chunk(2); dl_ = na - nu; npred = _cfg(nu, na, self.guidance)")
sub("scripts/quality/ls_stable.py", "nu, na = npred.chunk(2); npred = nu + self.guidance * (na - nu)", "nu, na = npred.chunk(2); npred = _cfg(nu, na, self.guidance)")
sub("scripts/quality/ls_stable.py", "nu, na = npred.chunk(2); dlb_ = na - nu; npred = nu + self.guidance * dlb_", "nu, na = npred.chunk(2); dlb_ = na - nu; npred = _cfg(nu, na, self.guidance)")
b = (R / "scripts/r15/bon.py").read_text()
if "_cfg(nu, na, acc.guidance)" not in b:
    n = b.count("nu, na = npred.chunk(2); npred = nu + acc.guidance * (na - nu)")
    if n != 3: sys.exit(f"bon.py: {n} CFG sites (expected 3)")
    b = b.replace("nu, na = npred.chunk(2); npred = nu + acc.guidance * (na - nu)", "nu, na = npred.chunk(2); npred = _ls_cfg(nu, na, acc.guidance)")
    b = b.replace("\nimport ", "\nfrom ls_stable import _cfg as _ls_cfg   # r15 Z LS_CFG_RESCALE\nimport ", 1) if "_ls_cfg" not in b.split("def ")[0] else b
    (R / "scripts/r15/bon.py").write_text(b); print("  + scripts/r15/bon.py (3x)")
print("PATCH_RESCALE_OK")
