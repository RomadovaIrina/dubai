#!/bin/bash
# every 20 min: git bundle of each round-4 clone -> Documents\DabAI_AB_backup\r5_git\
D=/c/Users/irina/Documents/DabAI_AB_backup/r5_git; mkdir -p $D
while true; do
  for h in r5-a r5-g; do
    timeout 300 ssh -o BatchMode=yes -o ConnectTimeout=20 $h 'cd /workspace/dub/dubai && BR=$(git branch --show-current) && git bundle create -q /tmp/r4.bundle "$BR" ^8ebb299 2>/dev/null && echo "$BR $(git rev-parse --short HEAD)"' 2>/dev/null | grep -vE "^(Welcome|Have fun|AI agents)" > $D/$h.head.tmp && \
    timeout 300 scp -q -o BatchMode=yes $h:/tmp/r4.bundle $D/$h.bundle.part 2>/dev/null && mv -f $D/$h.bundle.part $D/$h.bundle && mv -f $D/$h.head.tmp $D/$h.head && echo "$(date +%H:%M) $h $(cat $D/$h.head)"
  done
  sleep 1200
done
