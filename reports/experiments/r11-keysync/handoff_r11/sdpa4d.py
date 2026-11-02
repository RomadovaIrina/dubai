p = "sgm/modules/attention.py"; s = open(p).read()
old = "            out = F.scaled_dot_product_attention(q, k, v)\n"
assert s.count(old) == 1
new = ("            # 4-D (batch, heads, tokens, dim): 3-D input makes SDPA fall back to the MATH kernel (22x slower at 4096 tokens on sm_120)\n"
       "            out = F.scaled_dot_product_attention(*(t.view(b, self.heads, t.shape[1], self.dim_head) for t in (q, k, v))).reshape(b * self.heads, q.shape[1], self.dim_head)\n")
open(p, "w").write(s.replace(old, new, 1)); print("ok")
