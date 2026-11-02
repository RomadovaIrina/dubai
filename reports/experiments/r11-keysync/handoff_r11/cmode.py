p = "/workspace/dub/keysync/ks_fast.py"; s = open(p).read()
s = s.replace('    ap.add_argument("--vae-half"', '    ap.add_argument("--compile-mode", default="max-autotune-no-cudagraphs", help="torch.compile mode for the UNets (max-autotune = + CUDA graphs)")\n    ap.add_argument("--vae-half"', 1)
s = s.replace('torch.compile(m.model.diffusion_model, mode="max-autotune-no-cudagraphs", dynamic=False)', 'torch.compile(m.model.diffusion_model, mode=a.compile_mode, dynamic=False)', 1)
open(p, "w").write(s); print(s.count("a.compile_mode"))
