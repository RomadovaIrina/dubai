import json
G = "/workspace/dub/r16t/golden"; P = "/workspace/dub/r16in/HANDOFF_K/paste_inputs"; Q = "/workspace/dub/r16in/day1/HANDOFF_Q"
RT = "/workspace/dub/dubai/reports/experiments/r16-qa/routing"
src = {"m03": "/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV", "m18": "/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4",
       "old03": "/workspace/dub/inputs/test_videos_raw/03.mp4", "q02": Q + "/clips/q02_vera.mp4"}
for v, s in src.items():
    de = {"DUB_AUDIO_FROM": Q + "/mixes/q02_vera" if v == "q02" else P + "/" + v}
    if v == "m03": de["DUB_ROUTE_JSON"] = RT + "/m03.json"
    elif v == "q02": de["DUB_ROUTE_JSON"] = RT + "/q02_vera.json"
    else: de["DUB_ROUTE"] = "off"   # one face, no route needed
    json.dump({"job_id": "gold_" + v, "input": s, "out_dir": G + "/jobs/gold_" + v, "target_lang": "en", "timeout_s": 14400, "debug_env": de,
               "output": {"keep_intermediate": True}}, open(G + "/gold_" + v + ".json", "w"), indent=1)
    print(v, de)
