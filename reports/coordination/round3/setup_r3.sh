#!/bin/bash
# usage (on server): bash -s <BRANCH> <BASE> < setup_r3.sh
# Clones the repo, creates the round-3 branch from BASE, starts the environment setup in the background (log: /workspace/dub/setup.log).
BR=$1; BASE=$2
mkdir -p /workspace/dub && cd /workspace/dub
if [ ! -d dubai/.git ]; then
  for t in 1 2 3; do git clone -q git@github.com:RomadovaIrina/dubai.git dubai && break; rm -rf dubai; sleep 5; done
fi
cd dubai || { echo "CLONE FAILED"; exit 1; }
git fetch -q origin
git checkout -q -B "$BR" "origin/$BASE" && echo "branch $BR @ $(git rev-parse --short HEAD) from $BASE"
git push -q -u origin "$BR" 2>&1 | tail -1; echo "push test rc=$?"
ln -sfn /workspace/dub/inputs/test_videos_raw test_videos
grep -q '^test_videos' .git/info/exclude || echo test_videos >> .git/info/exclude
if ! pgrep -f setup_dabai.sh >/dev/null && ! grep -q SETUP_DONE /workspace/dub/setup.log 2>/dev/null; then
  nohup bash -c 'cd /workspace/dub/dubai && export DUB_ROOT=/workspace/dub/dubai && for t in 1 2 3; do (scripts/setup_dabai.sh && scripts/download_models.sh && scripts/setup_mediapipe.sh) && { echo SETUP_DONE; break; }; echo "SETUP attempt $t failed, retrying"; sleep 10; done' > /workspace/dub/setup.log 2>&1 &
  echo "setup started"
fi
