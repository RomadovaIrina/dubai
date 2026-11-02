p = "/workspace/dub/keysync/code/sgm/modules/attention.py"; s = open(p).read()
# CrossAttention + MemoryEfficientCrossAttention: when no sample skips attention (always at inference), skip the boolean-index copy in and the scatter out
old_in = "        x_non_skip = x[non_skip_indices]\n"
n = s.count(old_in); assert n >= 1, n
s = s.replace(old_in, "        _ks_all = not bool(skip_indices.any())  # keysync speed: nothing skipped -> no index copy / scatter\n        x_non_skip = x if _ks_all else x[non_skip_indices]\n")
old_out = "        combined_out = torch.zeros(\n            (x.shape[0], out.shape[1], out.shape[2]), dtype=out.dtype, device=out.device\n        )\n        combined_out[non_skip_indices] = out\n"
m = s.count(old_out); assert m >= 1, m
s = s.replace(old_out, "        if _ks_all:\n            combined_out = out\n        else:\n          combined_out = torch.zeros(\n            (x.shape[0], out.shape[1], out.shape[2]), dtype=out.dtype, device=out.device\n          )\n          combined_out[non_skip_indices] = out\n")
open(p, "w").write(s); print("replaced", n, m)
