import sys, os, shutil, torch
# scale_lora.py <src dir> <dst dir> <s>: the same keyframe LoRA at strength s (up weights x s; merge_lora math unchanged)
s_, d_, s = sys.argv[1], sys.argv[2], float(sys.argv[3]); os.makedirs(d_, exist_ok=True)
sd = torch.load(f"{s_}/kf_lora.pt", map_location="cpu"); sd = {k: (v * s if k.endswith(".up.weight") else v) for k, v in sd.items()}
torch.save(sd, f"{d_}/kf_lora.pt"); shutil.copy(f"{s_}/lora_meta.json", f"{d_}/lora_meta.json"); print("scaled", s_, "->", d_, s, len(sd))
