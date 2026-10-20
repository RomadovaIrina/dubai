#!/bin/bash
# Final save before the vast balance runs out: git bundles (all local commits incl. WIP snapshot), push attempt,
# every outbox mp4 + review/regression md, reports/ of each clone -> laptop.
cd "$(dirname "$0")"; TS=$(date -u +%m%d-%H%M)
G=/c/Users/irina/Documents/DabAI_AB_backup/r4_git; V=/c/Users/irina/Videos/DabAI_AB/r4; B=/c/Users/irina/Documents/DabAI_AB_backup/r4_final; mkdir -p $G $V $B
for h in r4-a r4-v; do
  timeout 300 ssh -o BatchMode=yes -o ConnectTimeout=20 $h 'cd /workspace/dub/dubai && BR=$(git branch --show-current); export GIT_INDEX_FILE=/tmp/fidx; cp .git/index $GIT_INDEX_FILE; git add -A; T=$(git write-tree); if [ "$T" != "$(git rev-parse HEAD^{tree})" ]; then C=$(git commit-tree $T -p HEAD -m "WIP snapshot by coordinator before shutdown"); git update-ref refs/heads/wip-'$TS' $C; fi; rm -f $GIT_INDEX_FILE; git push -q origin "$BR" 2>&1 | tail -1; git bundle create -q /tmp/final.bundle "$BR" $(git for-each-ref --format="%(refname)" refs/heads/wip-*) ^904c5d2; echo "$BR $(git rev-parse --short HEAD)"' 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)" | sed "s/^/$h: /"
  for t in 1 2 3; do timeout 600 scp -q -o BatchMode=yes $h:/tmp/final.bundle $G/$h.final.bundle && break; done
  S=$([ $h = r4-a ] && echo A || echo V)
  timeout 120 ssh -o BatchMode=yes $h 'cd /workspace/dub/outbox 2>/dev/null && find . -type f -printf "%s|%P\n"' 2>/dev/null | grep -E '^[0-9]+\|' | while IFS='|' read -r s p; do o="$V/${S}__$(echo $p | tr / _)"; [ -f "$o" ] && [ "$(stat -c %s "$o")" = "$s" ] && continue; ( for t in 1 2 3; do timeout 900 scp -q -o BatchMode=yes "$h:/workspace/dub/outbox/$p" "$o.part" </dev/null && mv -f "$o.part" "$o" && break; done ) & while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 1; done; done; wait
  timeout 600 ssh -o BatchMode=yes $h 'cd /workspace/dub/dubai && tar cf - reports 2>/dev/null' > $B/${S}_reports.tar 2>/dev/null
  timeout 600 ssh -o BatchMode=yes $h 'cd /workspace/dub && find handoff/out -type f -size -150M 2>/dev/null | tar cf - -T - 2>/dev/null' > $B/${S}_handoff.tar 2>/dev/null
done
echo "FINAL_SAVE_DONE $(date +%H:%M): bundles $(ls $G/*.final.bundle 2>/dev/null | wc -l), videos $(ls $V/*.mp4 2>/dev/null | wc -l), $(du -sh $B | cut -f1) reports/handoff"
