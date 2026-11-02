p = "sgm/modules/diffusionmodules/video_model.py"; s = open(p).read()
old = '''            st = getattr(self, "_ks_dc", None); tcur = float(timesteps.flatten()[0])
            if st is None or st["shape"] != tuple(x.shape) or tcur >= st["t"]:
                st = {"n": 0}
            else:
                st["n"] += 1
            st["t"] = tcur; st["shape"] = tuple(x.shape); self._ks_dc = st
            _full = st["n"] % _dcn == 0 or "h" not in st'''
new = '''            # the denoiser may call the UNet several times per step (chunk_network: chunks of the segment batch) -> one cache per call k
            st = getattr(self, "_ks_dc", None); tcur = float(timesteps.flatten()[0])
            if st is None or tcur > st["t"]:
                st = {"n": 0, "k": 0, "c": {}}
            elif tcur == st["t"]:
                st["k"] += 1
            else:
                st["n"] += 1; st["k"] = 0
            st["t"] = tcur; self._ks_dc = st
            _ck = st["c"].get(st["k"])
            _full = st["n"] % _dcn == 0 or _ck is None or _ck["shape"] != tuple(x.shape)'''
assert old in s; s = s.replace(old, new)
old = '''                if _full:
                    self._ks_dc["h"], self._ks_dc["ci"] = h, curr_context_idx
                else:
                    h, curr_context_idx = self._ks_dc["h"], self._ks_dc["ci"]'''
new = '''                if _full:
                    self._ks_dc["c"][self._ks_dc["k"]] = {"h": h, "ci": curr_context_idx, "shape": tuple(x.shape)}
                else:
                    h, curr_context_idx = _ck["h"], _ck["ci"]'''
assert old in s; s = s.replace(old, new); open(p, "w").write(s)
p = "scripts/sampling/dubbing_pipeline_raw.py"; s = open(p).read()
old = "            if chunk_size is not None:\n                chunk_size = chunk_size * num_frames\n"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "            if chunk_size is not None:  # keysync speed: KS_ICHUNK = segments per interpolation UNet call (authors: 2)\n                chunk_size = int(os.environ.get(\"KS_ICHUNK\", chunk_size)) * num_frames\n")
open(p, "w").write(s); print("ok")
