p = "/workspace/dub/keysync/ks_fast.py"; s = open(p).read()
old = "    for line in open(a.jobs):\n"
new = '''    ig = os.environ.get("KS_IGUIDE", "")   # interpolation guidance cost: cond | trunc:K | 2way  (keyframes keep the full 3-way CFG)
    if ig:
        g = model.sampler.guider; prep0, call0 = g.prepare_inputs, g.__call__; gs = {"n": 0, "s": None, "mode": "full"}
        def mode_now():
            if ig == "cond": return "cond"
            if ig == "2way": return "2way"
            if ig.startswith("trunc:"): return "full" if gs["n"] < int(ig.split(":")[1]) else "cond"
            raise ValueError(ig)
        def prep(x, sig, c, uc):
            sv = float(sig.flatten()[0])
            gs["n"] = 0 if gs["s"] is None or sv > gs["s"] else gs["n"] + 1; gs["s"] = sv; gs["mode"] = mode_now()
            xi, si, co = prep0(x, sig, c, uc)
            if gs["mode"] == "full": return xi, si, co
            B = x.shape[0]; keep = slice(2 * B, 3 * B) if gs["mode"] == "cond" else slice(B, 3 * B)   # rows: [uncond | ref | audio+ref]
            co = {k: (v[keep] if isinstance(v, torch.Tensor) and v.shape[0] == 3 * B else v) for k, v in co.items()}
            return xi[keep], si[keep], co
        def call(x, sig):
            if gs["mode"] == "full": return call0(x, sig)
            if gs["mode"] == "cond": return x
            e_ref, c_ar = x.chunk(2)   # 2-way: drop the uncond row = the ref term at ratio 1: e = e_ref + audio_ratio * (c - e_ref)
            return e_ref + g.audio_ratio * (c_ar - e_ref)
        g.prepare_inputs = prep; g.__call__ = call; model.sampler.guider = g
        type(g).__call__ = lambda self, x, sig: call(x, sig) if self is g else call0.__func__(self, x, sig)
    for line in open(a.jobs):
'''
assert old in s; open(p, "w").write(s.replace(old, new, 1)); print("ok")
