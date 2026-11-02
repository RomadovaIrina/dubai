p = "/workspace/dub/keysync/ks_fast.py"; s = open(p).read()
s = s.replace('    ap.add_argument("--vae-half"', '    ap.add_argument("--cudnn-bench", action="store_true"); ap.add_argument("--vae-cl", action="store_true", help="VAE decoder in channels_last")\n    ap.add_argument("--vae-half"', 1)
old = "    if a.vae_half:\n"
new = '''    if a.cudnn_bench: torch.backends.cudnn.benchmark = True
    if a.vae_cl:
        for m in (model, model_kf): m.first_stage_model.decoder.to(memory_format=torch.channels_last)
    if a.vae_half:
'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('"vae_half": a.vae_half,', '"vae_half": a.vae_half, "cudnn_bench": a.cudnn_bench, "vae_cl": a.vae_cl,', 1)
open(p, "w").write(s); print("ok")
