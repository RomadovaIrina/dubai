import json, sys
# mkjob.py <arm> <video> [KEY=VAL ...]: the current chain (coordinator 10.11) on the ACCEPTED dub (paste_inputs) + the arm env, keep_intermediate
arm, v, *kv = sys.argv[1:]
IN = {"m03": "/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV", "old03": "/workspace/dub/inputs/test_videos_raw/03.mp4"}[v]
env = {"DUB_AUDIO_FROM": f"/workspace/dub/r16in/HANDOFF_K/paste_inputs/{v}", "KS_MASK": "tight", "KS_OCCL": "on", "KS_TAIL_SRC": "on", "KS_CAP": "1.0", "KS_PAUSE": "warp", "KS_REF": "0"}
env.update(dict(x.split("=", 1) for x in kv)); out = f"/workspace/dub/r16l/d4/{arm}_{v}"
json.dump({"job_id": f"l4_{arm}_{v}", "input": IN, "out_dir": out, "target_lang": "en", "timeout_s": 10800, "output": {"keep_intermediate": True}, "debug_env": env},
          open(f"/workspace/dub/r16l/d4/j_{arm}_{v}.json", "w"), indent=1)
print(f"/workspace/dub/r16l/d4/j_{arm}_{v}.json")
