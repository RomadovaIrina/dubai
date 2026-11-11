#!/bin/bash
# DabAI image (r16 TZ 5.1 / 0.10 / 10.2): build + fill the models volume + self-test + COLD and WARM smoke on ONE Docker VM, with every
# number recorded (image size, build time, volume size, fetch time, first-job time incl. the TensorRT build, warm job time, API time-to-healthy).
# Pushes NOTHING. Run from the repo root on a Docker host with an RTX 5090 (see infra/docker/VM_REQUIREMENTS.md).
#     export DUB_PROFILE=clean HF_TOKEN=... KS_KIT_DIR=/path/kit [STUDENT_LORA_FILE=/path/latentsync_student_r2_s4_step01500_ema.pt]
#     bash infra/docker/build_smoke.sh /path/to/smoke_clip.mp4 [out_dir]
#   KS_KIT_DIR = lora_pick.pt + lora_meta.json + kf.pt + interp.pt (laptop: C:\repos\chkpts\keysync_student_r15_lcm_r1 + keysync_trt_r15/keysync/trt/cap)
set -uo pipefail
cd "$(dirname "$0")/../.."
CLIP=$(readlink -f "${1:?usage: build_smoke.sh <smoke clip.mp4> [out dir]}"); OUT=$(mkdir -p "${2:-build_smoke_out}" && readlink -f "${2:-build_smoke_out}")
VOL=${VOL:-dabai-models}; IMAGE=${IMAGE:-dabai-worker}; R=$OUT/report.json; LOG=$OUT/build_smoke.log; : > "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }
num(){ python3 - "$R" "$1" "$2" <<'PY'
import json, sys, os
p, k, v = sys.argv[1:]; d = json.load(open(p)) if os.path.exists(p) else {}
try: v = float(v) if "." in v or v.isdigit() else v
except ValueError: pass
d[k] = v; json.dump(d, open(p, "w"), indent=1)
PY
}
t(){ date +%s.%N; }; dt(){ python3 -c "print(round($(t) - $1, 1))"; }
# ---- 0. prerequisites
case "${DUB_PROFILE:-}" in clean|current) ;; *) echo "set DUB_PROFILE=clean|current" >&2; exit 2;; esac
[ -n "${HF_TOKEN:-}" ] || { echo "set HF_TOKEN (pyannote 3.1 + the SVD VAE terms accepted)" >&2; exit 2; }
for f in lora_pick.pt lora_meta.json kf.pt interp.pt; do [ -s "${KS_KIT_DIR:-/nonexistent}/$f" ] || { echo "KS_KIT_DIR must hold $f" >&2; exit 2; }; done
docker buildx version >/dev/null 2>&1 || { echo "docker buildx missing" >&2; exit 3; }
docker info 2>/dev/null | grep -qi "nvidia" || log "WARNING: no nvidia runtime listed by docker info (nvidia-container-toolkit?)"
FREE=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc 0-9); log "free disk under /var/lib/docker: ${FREE:-?} GB (need ~150)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$LOG"
num gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; num driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
num profile "$DUB_PROFILE"; num git_sha "$(git rev-parse --short=12 HEAD)"
# ---- 1. build
t0=$(t); log "1/6 build"
DUB_PROFILE=$DUB_PROFILE IMAGE=$IMAGE bash infra/docker/build.sh >> "$LOG" 2>&1 || { log "BUILD FAILED"; exit 4; }
TAG=$IMAGE:$(git rev-parse --short=12 HEAD)-$DUB_PROFILE
num build_s "$(dt $t0)"; num image "$TAG"; num image_bytes "$(docker image inspect "$TAG" --format '{{.Size}}')"; log "built $TAG"
# ---- 2. models volume
t0=$(t); log "2/6 fetch-models (~37 GB: 23.6 GB pipeline + 13.2 GB KeySync)"
docker volume create "$VOL" >/dev/null
SL=(); [ -n "${STUDENT_LORA_FILE:-}" ] && SL=(-v "$(dirname "$STUDENT_LORA_FILE")":/in_s:ro -e STUDENT_LORA_FILE=/in_s/$(basename "$STUDENT_LORA_FILE"))
docker run --rm --gpus all -e HF_TOKEN -v "$VOL":/workspace/dub/dubai/models -v "$KS_KIT_DIR":/in_k:ro -e KS_KIT_DIR=/in_k "${SL[@]}" "$TAG" fetch-models >> "$LOG" 2>&1 \
  || { log "FETCH FAILED"; exit 5; }
num fetch_s "$(dt $t0)"; num volume_gb "$(docker run --rm -v "$VOL":/m --entrypoint du "$TAG" -s -B1G /m | cut -f1)"
# ---- 3. self-test
log "3/6 check"; docker run --rm --gpus all -v "$VOL":/workspace/dub/dubai/models "$TAG" check --deep > "$OUT/check.log" 2>&1; num check "$(tail -1 "$OUT/check.log")"
# ---- 4. cold smoke (first job on this GPU / driver: builds the TensorRT plans + the inductor cache into the volume)
mkdir -p "$OUT/jobs"; log "4/6 cold smoke on $(basename "$CLIP")"
t0=$(t); docker run --rm --gpus all --shm-size 16g -v "$VOL":/workspace/dub/dubai/models -v "$OUT/jobs":/data/jobs -v "$CLIP":/in.mp4:ro "$TAG" smoke /in.mp4 > "$OUT/smoke_cold.log" 2>&1
num smoke_cold_s "$(dt $t0)"; cp "$OUT/jobs/smoke/status.json" "$OUT/smoke_cold.status.json" 2>/dev/null; num smoke_cold_status "$(python3 -c "import json; print(json.load(open('$OUT/smoke_cold.status.json'))['status'])" 2>/dev/null || echo none)"
# ---- 5. warm smoke (plans cached)
log "5/6 warm smoke"; rm -rf "$OUT/jobs/smoke"
t0=$(t); docker run --rm --gpus all --shm-size 16g -v "$VOL":/workspace/dub/dubai/models -v "$OUT/jobs":/data/jobs -v "$CLIP":/in.mp4:ro "$TAG" smoke /in.mp4 > "$OUT/smoke_warm.log" 2>&1
num smoke_warm_s "$(dt $t0)"; cp "$OUT/jobs/smoke/status.json" "$OUT/smoke_warm.status.json" 2>/dev/null; num smoke_warm_status "$(python3 -c "import json; print(json.load(open('$OUT/smoke_warm.status.json'))['status'])" 2>/dev/null || echo none)"
num trt_cache_gb "$(docker run --rm -v "$VOL":/m --entrypoint du "$TAG" -s -B1G /m/keysync/trt_cache 2>/dev/null | cut -f1)"
# ---- 6. API time-to-healthy (container start -> GET /health 200)
log "6/6 API cold start"; t0=$(t)
CID=$(docker run -d --rm --gpus all --shm-size 16g -p 18000:8000 -e DABAI_API_TOKEN=smoke -v "$VOL":/workspace/dub/dubai/models -v "$OUT/jobs":/data/jobs "$TAG")
for i in $(seq 300); do curl -sf http://127.0.0.1:18000/health > "$OUT/health.json" && break; sleep 1; done
num api_healthy_s "$(dt $t0)"; docker stop "$CID" >/dev/null 2>&1
log "done: $(cat "$R")"
echo "NEXT (yours, with registry credentials): docker tag $TAG <registry>/<repo>:<tag> && docker push <registry>/<repo>:<tag>"
