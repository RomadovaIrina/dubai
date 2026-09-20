#!/usr/bin/env bash
# Local smoke test of a built worker image WITHOUT model weights or GPU:
#   1. mock volume (1-byte stand-ins, generated from handler.py's own mandatory list);
#   2. entrypoint prepare twice (idempotent symlinks);
#   3. handler --check --no-cuda -> must be ready (paths/env/symlinks) ; handler --check -> ready=false ONLY because of CUDA;
#   4. handler --import-smoke; 5. RunPod SDK local run of the 'info' action via --test_input.
# Usage: infra/serverless/smoke.sh <image> [--gpu]   (--gpu adds --gpus all and expects the CUDA checks to pass; sm mismatch is
#        still reported unless DABAI_REQUIRE_SM=any is exported — this is the RTX-2000/preparation-Pod case)
set -euo pipefail
IMAGE="${1:?image}"; GPU="${2:-}"
MOCK="${DABAI_MOCK_VOLUME:-$(mktemp -d /tmp/dabai-mock-volume.XXXX)}"
RUN=(docker run --rm -v "${MOCK}:/runpod-volume" -e "DABAI_REQUIRE_SM=${DABAI_REQUIRE_SM:-sm_120}")
[ "${GPU}" = "--gpu" ] && RUN+=(--gpus all)
echo "== 1. mock volume -> ${MOCK}"
docker run --rm -v "${MOCK}:/runpod-volume" --entrypoint python "${IMAGE}" /opt/dabai/infra/serverless/handler.py --mock-volume /runpod-volume
echo "== 2. entrypoint prepare (x2, idempotent)"; "${RUN[@]}" "${IMAGE}" prepare; "${RUN[@]}" "${IMAGE}" prepare
echo "== 3a. readiness without CUDA (paths/env/symlinks) -> must be ready"
"${RUN[@]}" --entrypoint /opt/dabai/infra/serverless/entrypoint.sh "${IMAGE}" python /opt/dabai/infra/serverless/handler.py --check --no-cuda | tee /tmp/dabai-smoke-check.json | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["ready"], d["missing"]; print("ready (no-cuda):", d["ready"])'
echo "== 3b. full readiness (CUDA) -> informational on hosts without the target GPU"
"${RUN[@]}" "${IMAGE}" check | python3 -c 'import json,sys; d=json.load(sys.stdin); print("ready:", d["ready"], "| gpu:", d.get("gpu"), d.get("sm"), "| missing:", d["missing"])' || true
echo "== 4. import smoke"; "${RUN[@]}" --entrypoint python "${IMAGE}" /opt/dabai/infra/serverless/handler.py --import-smoke | tail -3
echo "== 5. RunPod SDK local job: info"
"${RUN[@]}" "${IMAGE}" serve --test_input '{"input": {"action": "info"}}' 2>&1 | tail -15
echo "== 6. no-download guard: entrypoint without a volume must fail fast (exit 64)"
set +e; docker run --rm "${IMAGE}" prepare; rc=$?; set -e
if [ "${rc}" = "64" ]; then echo "fail-fast OK (exit ${rc})"; else echo "expected exit 64, got ${rc}"; exit 1; fi
echo "== smoke done (mock volume kept at ${MOCK})"
