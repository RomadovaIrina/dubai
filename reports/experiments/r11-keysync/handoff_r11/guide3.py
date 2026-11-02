p = "/workspace/dub/keysync/ks_fast.py"; s = open(p).read()
old = '''            gs["n"] = 0 if gs["s"] is None or sv > gs["s"] else gs["n"] + 1; gs["s"] = sv; gs["mode"] = mode_now()
'''
new = '''            gs["n"] = 0 if gs["s"] is None or sv > gs["s"] else gs["n"] + 1; gs["s"] = sv; prev = gs["mode"]; gs["mode"] = mode_now()
            if gs["mode"] != prev:   # the rows per UNet chunk change -> DeepCache's per-chunk features no longer match: start a new cache
                dmo = getattr(model.model.diffusion_model, "_orig_mod", model.model.diffusion_model); dmo._ks_dc = None
'''
assert old in s; s = s.replace(old, new, 1); open(p, "w").write(s); print("ok")
