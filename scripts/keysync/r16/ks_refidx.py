#!/usr/bin/env python3
"""r16 K2's KeySync edit (scripts/keysync/r16/keysync_r16_refidx.patch is a bare hunk): the identity / reference frame from KS_REF_IDX
(unset = 0 = the authors' behaviour). Idempotent; run from the keysync code dir: python3 ks_refidx.py"""
import pathlib
p = pathlib.Path("scripts/sampling/dubbing_pipeline_raw.py"); s = p.read_text(); old = "    random_cond_idx = 0\n"
if "KS_REF_IDX" not in s:
    assert s.count(old) == 1, "unexpected dubbing_pipeline_raw.py"
    s = s.replace(old, '    random_cond_idx = int(__import__("os").environ.get("KS_REF_IDX", 0))   # r16 K2: the identity / reference frame\n')
    p.write_text(s)
print("REFIDX_OK")
