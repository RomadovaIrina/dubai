p = "/workspace/dub/keysync/code/sgm/modules/attention.py"; s = open(p).read()
old = "            out = F.scaled_dot_product_attention(*(t.view(b, self.heads, t.shape[1], self.dim_head) for t in (q, k, v))).reshape(b * self.heads, q.shape[1], self.dim_head)\n"
new = "            out = _ks_sdpa(*(t.view(b, self.heads, t.shape[1], self.dim_head) for t in (q, k, v))).reshape(b * self.heads, q.shape[1], self.dim_head)\n"
assert s.count(old) == 1; s = s.replace(old, new)
old2 = "            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)\n"
new2 = "            out = _ks_sdpa(q, k, v) if mask is None else F.scaled_dot_product_attention(q, k, v, attn_mask=mask)\n"
assert s.count(old2) == 1; s = s.replace(old2, new2)
helper = '''

# r13 speed: KS_SAGE=1 -> SageAttention 1.0.6 (triton, INT8 QK) for the unmasked attention with half inputs and head dim 64 / 96 / 128; else torch SDPA
_KS_SAGE = os.environ.get("KS_SAGE") == "1"
if _KS_SAGE:
    from sageattention import sageattn as _sageattn


def _ks_sdpa(q, k, v):
    if _KS_SAGE and q.dtype in (torch.float16, torch.bfloat16) and q.shape[-1] in (64, 96, 128) and k.dtype == q.dtype and v.dtype == q.dtype:
        return _sageattn(q.contiguous(), k.contiguous(), v.contiguous(), tensor_layout="HND")
    return F.scaled_dot_product_attention(q, k, v)
'''
i = s.index("\nclass "); s = s[:i] + helper + s[i:]
if "\nimport os" not in s and not s.startswith("import os"): s = "import os\n" + s
open(p, "w").write(s); print("ok")
