p = "/workspace/dub/keysync/ks_fast.py"; s = open(p).read()
s = s.replace('    ap.add_argument("--vae-half"', '    ap.add_argument("--fp8", action="store_true", help="torchao float8 dynamic activation + weight on the UNets\' linear layers")\n    ap.add_argument("--vae-half"', 1)
old = "    if a.compile == \"all\":"
new = '''    if a.fp8:
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, PerRow
        for m in (model, model_kf): quantize_(m.model.diffusion_model, float8_dynamic_activation_float8_weight(granularity=PerRow()))
    if a.compile == "all":'''
assert old in s; s = s.replace(old, new, 1); s = s.replace('"cudnn_bench": a.cudnn_bench,', '"cudnn_bench": a.cudnn_bench, "fp8": a.fp8,', 1)
open(p, "w").write(s); print("ok")
