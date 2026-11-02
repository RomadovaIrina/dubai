import re
p = "sgm/modules/diffusionmodules/video_model.py"; s = open(p).read()
# 1) decide full / cached step right before the input blocks
a = "        h = x\n\n        if self.encode_landmarks:"
assert s.count(a) == 1
s = s.replace(a, '''        h = x
        # keysync speed: DeepCache (KS_DC=N[:B]): every N-th denoising step runs the whole UNet; the others run only input blocks 0..B and
        # output blocks L-1-B..L-1 and reuse the deep features (the input to output block L-1-B) cached on the last full step.
        # A new sampling run = the timestep goes up again (sigma falls within a run) or the batch shape changes -> full step.
        _dc = os.environ.get("KS_DC", "")
        _dcn, _dcb = (int(_dc.split(":")[0]), int(_dc.split(":")[1]) if ":" in _dc else 0) if _dc else (0, 0)
        _full = True
        if _dcn > 1:
            st = getattr(self, "_ks_dc", None); tcur = float(timesteps.flatten()[0])
            if st is None or st["shape"] != tuple(x.shape) or tcur >= st["t"]:
                st = {"n": 0}
            else:
                st["n"] += 1
            st["t"] = tcur; st["shape"] = tuple(x.shape); self._ks_dc = st
            _full = st["n"] % _dcn == 0 or "h" not in st
        _L = len(self.output_blocks)

        if self.encode_landmarks:''')
# 2) input blocks: stop after block B on a cached step
a = "        for i, module in enumerate(self.input_blocks):\n"
assert s.count(a) == 1
s = s.replace(a, "        for i, module in enumerate(self.input_blocks):\n            if not _full and i > _dcb:\n                break\n")
# 3) middle block only on full steps
a = "        h, is_attention = self.middle_block("
assert s.count(a) == 1
s = s.replace(a, "        if _full:\n          h, is_attention = self.middle_block(")
# indent the middle-block call's continuation lines + the context increment: find the block between the middle call and the output loop
i0 = s.index("        if _full:\n          h, is_attention = self.middle_block(")
i1 = s.index("        for i, module in enumerate(self.output_blocks):", i0)
seg = s[i0:i1].split("\n")
out = seg[:2] + [("  " + l if l.strip() else l) for l in seg[2:]]
s = s[:i0] + "\n".join(out) + s[i1:]
# 4) output blocks: skip the deep ones on a cached step, cache h + context index on a full step
a = "        for i, module in enumerate(self.output_blocks):\n            skip_x = hs.pop()\n"
assert s.count(a) == 1
s = s.replace(a, '''        for i, module in enumerate(self.output_blocks):
            if _dcn > 1 and i == _L - 1 - _dcb:
                if _full:
                    self._ks_dc["h"], self._ks_dc["ci"] = h, curr_context_idx
                else:
                    h, curr_context_idx = self._ks_dc["h"], self._ks_dc["ci"]
            elif not _full and i < _L - 1 - _dcb:
                continue
            skip_x = hs.pop()
''')
if "\nimport os\n" not in s: s = "import os\n" + s
open(p, "w").write(s)
print("patched")
