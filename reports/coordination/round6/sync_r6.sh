#!/bin/bash
# Round 3 laptop loop (every 2 min):
#  1) outbox mp4 of every server -> C:\Users\irina\Videos\DabAI_AB\r3\<S>__<file>
#  2) handoff relay: <S>:/workspace/dub/handoff/out/<T>/<name>/READY -> laptop copy (kept as backup) -> <T>:/workspace/dub/handoff/in/<S>/<name>/
# Agents: "S=host" or "S=host@ROOT" in $SERVERS (ROOT default /workspace/dub; it holds outbox/ and handoff/{in,out}/), e.g. "A=r3-a B=r3-b C=r3-a@/workspace/dub/c D=r3-d"
cd "$(dirname "$0")"
VID=/c/Users/irina/Videos/DabAI_AB/r6
REL=/c/Users/irina/Documents/DabAI_AB_backup/r6_handoff
mkdir -p "$VID" "$REL"
SSH="timeout 900 ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
filt() { grep -vE "^(Welcome|Have fun|AI agents)"; }
ent() { for p in $SERVERS; do [ "${p%%=*}" = "$1" ] && echo "${p#*=}"; done; }
host() { e=$(ent "$1"); echo "${e%%@*}"; }
root() { e=$(ent "$1"); case "$e" in *@*) echo "${e#*@}";; *) echo /workspace/dub;; esac; }
while true; do
  for p in $SERVERS; do S=${p%%=*}; [ -n "$ONLY" ] && [ "$S" != "$ONLY" ] && continue; h=$(host $S); R=$(root $S)
    # 1) videos
    $SSH $h "cd $R/outbox 2>/dev/null && find . -type f -name '*.mp4' -size +20k -mmin +0.5 -printf '%s|%P\n'" </dev/null 2>/dev/null | filt | grep -E '^[0-9]+\|' |
    { while IFS='|' read -r size path; do
      out="$VID/${S}__$(echo "$path" | tr '/' '_')"
      [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ] && continue
      ( for t in 1 2 3; do timeout 1200 scp -q -o BatchMode=yes -o ConnectTimeout=20 "$h:$R/outbox/$path" "$out.part" </dev/null 2>/dev/null && mv -f "$out.part" "$out" && echo "$(date +%H:%M) video $S $path" && break; done ) &
      while [ $(jobs -rp | wc -l) -ge ${PAR:-4} ]; do sleep 2; done
    done; wait; }
    # 2) handoff relay
    $SSH $h "cd $R/handoff/out 2>/dev/null && for r in */*/READY; do [ -f \"\$r\" ] && d=\$(dirname \"\$r\") && [ ! -f \"\$d/.relayed\" ] && echo \"\$d\"; done" </dev/null 2>/dev/null | filt | grep -E '^[A-Z]/' |
    while read -r d; do T=${d%%/*}; name=${d#*/}; th=$(host "$T"); TR=$(root "$T"); [ -z "$th" ] && continue
      key="$REL/${S}__$(echo "$name" | tr '/' '_').tar"
      rs=$($SSH $h "cd $R/handoff/out/$T && find '$name' -type f ! -name .relayed -printf '%s %P\n' | sort | md5sum | cut -c1-32" </dev/null 2>/dev/null | filt | tail -1)
      if [ -n "$rs" ] && [ -f "$key" ] && [ "$(cat "$key.src" 2>/dev/null)" = "$rs" ]; then tarf="$key"; ok=1
      else
        tarf="$key"; ok=""; for t in 1 2 3; do $SSH $h "cd $R/handoff/out/$T && tar cf - --exclude=.relayed '$name'" </dev/null 2>/dev/null > "$tarf.part" && mv -f "$tarf.part" "$tarf" && echo "$rs" > "$tarf.src" && ok=1 && break; done
      fi
      [ -z "$ok" ] && continue
      if [ "$(stat -c %s "$tarf")" -gt 60000000 ]; then
        r=$(bash chunked_upload.sh "$th" "$tarf" "$TR/handoff/in/$S" </dev/null)
      else
        r=""; for t in 1 2 3; do $SSH $th "mkdir -p $TR/handoff/in/$S && tar xf - -C $TR/handoff/in/$S" < "$tarf" 2>/dev/null && r="OK" && break; done
      fi
      if echo "$r" | grep -q OK; then
        $SSH $h "touch '$R/handoff/out/$T/$name/.relayed'" </dev/null >/dev/null 2>&1
        echo "$(date +%H:%M) relay $S -> $T $name ($(du -h "$tarf" | cut -f1))"
      else echo "$(date +%H:%M) relay FAIL $S -> $T $name"; fi
    done
  done
  sleep 120
done
