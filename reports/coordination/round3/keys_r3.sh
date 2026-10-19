#!/bin/bash
# usage: bash keys_r3.sh <A|B|C|D>  — deploy key + git identity, prints PUBKEY (no secrets needed)
S=$1
test -f ~/.ssh/id_ed25519_dubai || ssh-keygen -q -t ed25519 -N "" -C "dubai-r3-server$S-2026-10-19" -f ~/.ssh/id_ed25519_dubai
grep -q id_ed25519_dubai ~/.ssh/config 2>/dev/null || printf "Host github.com\n  HostName github.com\n  User git\n  IdentityFile ~/.ssh/id_ed25519_dubai\n  IdentitiesOnly yes\n" >> ~/.ssh/config
chmod 600 ~/.ssh/config
ssh-keyscan -t ed25519,rsa,ecdsa github.com 2>/dev/null >> ~/.ssh/known_hosts
git config --global user.name "DabAI agent $S"
git config --global user.email "dabai-agent-$S@users.noreply.github.com"
git config --global pull.ff only
echo "HOST $(hostname) GPU $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) DISK $(df -h /workspace 2>/dev/null | tail -1 | awk '{print $2, $4}') CPUS $(nproc)"
for u in https://github.com https://huggingface.co https://pypi.org https://drive.usercontent.google.com; do curl -s -o /dev/null -m 10 -w "$u %{http_code}\n" $u || echo "$u FAIL"; done
echo "PUBKEY: $(cat ~/.ssh/id_ed25519_dubai.pub)"
