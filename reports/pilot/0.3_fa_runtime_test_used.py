import time, torch, flash_attn
from flash_attn import flash_attn_func
import torch.nn.functional as F
print("flash_attn", flash_attn.__version__, flash_attn.__file__)
ok = True
for (B,L,Lk,H,D,causal) in [(1,256,256,8,64,False),(1,4096,4096,8,64,False),(16,1024,1024,8,64,False),(1,1024,1024,16,64,True),(2,2048,2048,16,128,True)]:
    q=torch.randn(B,L,H,D,device="cuda",dtype=torch.float16); k=torch.randn(B,Lk,H,D,device="cuda",dtype=torch.float16); v=torch.randn_like(k)
    ref=F.scaled_dot_product_attention(q.transpose(1,2).float(),k.transpose(1,2).float(),v.transpose(1,2).float(),is_causal=causal).transpose(1,2)
    for _ in range(3): y=flash_attn_func(q,k,v,causal=causal)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(20): y=flash_attn_func(q,k,v,causal=causal)
    torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/20*1000
    err=(y.float()-ref).abs().max().item(); fin=bool(torch.isfinite(y).all())
    # backward too (JIT / kernel coverage)
    q.requires_grad_(True); flash_attn_func(q,k,v,causal=causal).sum().backward(); torch.cuda.synchronize()
    good = fin and err < 5e-3
    ok &= good
    print(f"B={B} L={L} kv={Lk} H={H} D={D} causal={causal}: {ms:.3f} ms  max|d| vs fp32 {err:.2e} finite={fin} bwd=ok -> {'PASS' if good else 'FAIL'}")
print("RUNTIME", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
