#!/bin/bash
# Phase 2 (speed, r7 F): one lever arm = generator (r7_e2e.sh, SKIP_SCORE) with VIDEO_EXTRA flags + the m8 composite stack (r7_m7_comp.sh:
# layer SR + fps_composite + score + twitch + whole-frame gate) + the quality-baseline gate. Output in artifacts/r4m/<ARM>raw (layers) and
# artifacts/r4m/<ARM> (candidate) + artifacts/r4m/<ARM>/gate_vs_baseline.md + speed.json (per-stage s/video-min).
#   ARM=s10 VIDEO_EXTRA="--steps 10" AUDIO=/workspace/dub/handoff/in/audio-r7-v3a1 bash scripts/r7/p2_arm.sh 05 [04 ...]
#   env: SR_EXTRA (r7_layer_sr.py flags, e.g. "--temporal 0"), FIX30_EXTRA / EXTRA (composite flags), SR (layer suffix, default srt1)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
ARM=${ARM:?}; AUDIO=${AUDIO:-/workspace/dub/handoff/in/audio-r7-v3a1}; B=/workspace/dub/artifacts/r4m; IDS=${*:-05}; RAW=${ARM}raw
echo "== p2_arm $ARM VIDEO_EXTRA='${VIDEO_EXTRA:-}' SR_EXTRA='${SR_EXTRA:-}' EXTRA='${EXTRA:-}' $(date +%T)"
AUDIO=$AUDIO TAG=$RAW SKIP_SCORE=1 bash scripts/r7/r7_e2e.sh $IDS 2>&1 | grep "^==\|FAIL"
TAG=$ARM SRC_TAG=$RAW SR=${SR:-srt1} SR_EXTRA="${SR_EXTRA:-}" FIX30_THREADS=${FIX30_THREADS:-16} EXTRA="${EXTRA:---tex-tau 1000 --tex-sigma 2.5}" bash scripts/r7/r7_m7_comp.sh $IDS 2>&1 | grep "^==\|FAIL"
python scripts/r7/r7_baseline.py gate --tag $ARM --ids $IDS --audio $AUDIO --baseline reports/experiments/r7-face/quality_baseline.json --out $B/$ARM/gate_vs_baseline.md | grep "FAIL\|Verdict"
python - $ARM $RAW $IDS <<'PY'
import json, pathlib, sys
arm, raw, ids = sys.argv[1], sys.argv[2], sys.argv[3:]; B = pathlib.Path("/workspace/dub/artifacts/r4m"); out = {}
for id_ in ids:
    try:
        m25 = json.loads((B / raw / f"work_{id_}_{raw}" / f"{id_}_{raw}_25fps.manifest.json").read_text()); st = m25.get("stage_seconds") or {}
        mc = json.loads((B / arm / f"{id_}_{arm}.manifest.json").read_text()); vmin = mc["source"]["duration_s"] / 60; r = mc.get("r4v", {}).get("stage_wall_s", {})
        ls = m25["lipsync"]; calls = ls.get("runner_calls", [])
        unet = sum(c["stages"].get("unet_denoise", 0) for c in calls); vae = sum(c["stages"].get("vae_and_cond", 0) for c in calls); bridge = sum(c["stages"].get("bridge", 0) for c in calls)
        gen_total = st.get("total", 0)
        out[id_] = {"s_per_video_min": {"generator_stage": round(gen_total / vmin, 1), "latentsync": round(st.get("latentsync", 0) / vmin, 1), "unet": round(unet / vmin, 1), "vae_cond": round(vae / vmin, 1), "bridge": round(bridge / vmin, 1),
                    "mediapipe": round((st.get("face_stack_mediapipe", 0) + st.get("prep_mediapipe", 0)) / vmin, 1), "ls_load": round(st.get("latentsync_load", 0) / vmin, 1),
                    "layer_sr_and_composite": round(r.get("layer_sr_and_composite", 0) / vmin, 1), "face_total": round((gen_total + r.get("layer_sr_and_composite", 0)) / vmin, 1)},
                    "ls_frames": ls["by_action"].get("LATENT_SYNC", {}).get("frames"), "windows": sum(c["windows"] for c in calls), "unet_calls": sum(c["unet_calls"] for c in calls), "runner": ls.get("runner"),
                    "peak_vram_mib": mc.get("r4v", {}).get("peak_vram_mib")}
    except Exception as e: out[id_] = {"error": str(e)}
(B / arm / "speed.json").write_text(json.dumps(out, indent=1)); print("== speed", arm, json.dumps({k: v.get("s_per_video_min") for k, v in out.items()}))
PY
echo "== p2_arm $ARM done $(date +%T)"
