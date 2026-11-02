p = "/workspace/dub/keysync/ks_fast.py"; s = open(p).read()
old = "    for line in open(a.jobs):\n"
new = '''    if os.environ.get("KS_PROFILE"):   # profile UNet calls 40..60 of the interpolation model, print the top CUDA ops
        from torch.profiler import profile, ProfilerActivity
        dm = model.model.diffusion_model; fwd = dm.forward; st = {"n": 0, "p": None}
        def pf(*x, **k):
            st["n"] += 1
            if st["n"] == 40: st["p"] = profile(activities=[ProfilerActivity.CUDA]); st["p"].__enter__()
            r = fwd(*x, **k)
            if st["n"] == 60:
                torch.cuda.synchronize(); st["p"].__exit__(None, None, None)
                print(st["p"].key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
            return r
        dm.forward = pf
    for line in open(a.jobs):
'''
assert old in s; open(p, "w").write(s.replace(old, new, 1)); print("ok")
