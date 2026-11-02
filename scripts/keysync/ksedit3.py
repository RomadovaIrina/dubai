p = "/workspace/dub/keysync/ks_fast.py"
s = open(p).read()
a = """        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, PerRow
        for m in (model, model_kf): quantize_(m.model.diffusion_model, float8_dynamic_activation_float8_weight(granularity=PerRow()))"""
b = """        # PerRow needs bf16 weights; the UNets are fp16 -> PerTensor; only the Linear layers with in / out % 16 == 0 (float8 GEMM shapes)
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, PerTensor
        f16 = lambda mod, fqn: isinstance(mod, torch.nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        for m in (model, model_kf): quantize_(m.model.diffusion_model, float8_dynamic_activation_float8_weight(granularity=PerTensor()), filter_fn=f16)"""
assert a in s; s = s.replace(a, b)
a = "        for m in (model, model_kf): m.first_stage_model.decoder.to(memory_format=torch.channels_last)"
b = """        for m in (model, model_kf):   # the temporal decoder has 5-D (Conv3d) weights: channels_last on the 2-D convs only
            for mod in m.first_stage_model.decoder.modules():
                if isinstance(mod, torch.nn.Conv2d): mod.to(memory_format=torch.channels_last)"""
assert a in s; s = s.replace(a, b)
open(p, "w").write(s); print("EDITED3")
