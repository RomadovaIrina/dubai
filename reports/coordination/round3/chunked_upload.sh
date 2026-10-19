#!/bin/bash
# usage: chunked_upload.sh HOST LOCAL_TAR REMOTE_EXTRACT_DIR
# Splits the tar into 5 MB parts, uploads each part with retries (resumable), then reassembles, checks sha256 and extracts.
H=$1; TAR=$2; DEST=$3
name=$(basename "$TAR"); work=$(mktemp -d); split -b 5M -d -a 4 "$TAR" "$work/$name.part."
sum=$(sha256sum "$TAR" | cut -d' ' -f1)
ssh -o BatchMode=yes "$H" "mkdir -p /tmp/up_$name" 2>/dev/null
for p in "$work"/*; do
  bn=$(basename "$p"); sz=$(stat -c %s "$p")
  for t in 1 2 3 4 5 6; do
    rs=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$H" "stat -c %s /tmp/up_$name/$bn 2>/dev/null" 2>/dev/null | grep -E '^[0-9]+$')
    [ "$rs" = "$sz" ] && break
    scp -q -o BatchMode=yes -o ConnectTimeout=20 "$p" "$H:/tmp/up_$name/$bn" 2>/dev/null
  done
done
ssh -o BatchMode=yes "$H" "cd /tmp/up_$name && cat $name.part.* > /tmp/$name && [ \$(sha256sum /tmp/$name | cut -d' ' -f1) = $sum ] && mkdir -p $DEST && tar xf /tmp/$name -C $DEST && rm -rf /tmp/up_$name /tmp/$name && echo OK $name || echo FAIL $name" 2>&1 | grep -E "^(OK|FAIL)"
rm -rf "$work"
