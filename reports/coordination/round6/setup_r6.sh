#!/bin/bash
# usage (on server): bash -s <BRANCH> [BASE] < setup_r6.sh — HTTPS clone (SSH push), new branch from origin/experiment/r4-merged-m4, setup in background
BR=$1; BASE=${2:-origin/experiment/r4-merged-m4}
mkdir -p /workspace/dub && cd /workspace/dub
[ -d dubai/.git ] || { for t in 1 2 3; do git clone -q https://github.com/RomadovaIrina/dubai.git dubai && break; rm -rf dubai; sleep 5; done; }
cd dubai || { echo CLONE_FAILED; exit 1; }
git remote set-url --push origin git@github.com:RomadovaIrina/dubai.git
git fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
git checkout -q -B "$BR" $BASE && echo "branch $BR @ $(git rev-parse --short HEAD)"
ln -sfn /workspace/dub/inputs/test_videos_raw test_videos; grep -q '^test_videos' .git/info/exclude || echo test_videos >> .git/info/exclude
if ! pgrep -f setup_dabai.sh >/dev/null && ! grep -q SETUP_DONE /workspace/dub/setup.log 2>/dev/null; then
  nohup bash -c 'cd /workspace/dub/dubai && export DUB_ROOT=/workspace/dub/dubai && for t in 1 2 3; do (scripts/setup_dabai.sh && scripts/download_models.sh && scripts/setup_mediapipe.sh) && { echo SETUP_DONE; break; }; echo "SETUP attempt $t failed"; sleep 10; done; [ -f scripts/setup_llama_cuda.sh ] && JOBS=16 bash scripts/setup_llama_cuda.sh && echo LLAMA_CUDA_DONE' > /workspace/dub/setup.log 2>&1 &
  echo "setup started"
fi
