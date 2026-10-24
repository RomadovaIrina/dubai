#!/bin/bash
# usage: bash prep_r5.sh ALIAS IP PORT S BRANCH — ssh alias, deploy key, HF token, clone+setup, inputs
A=$1; IP=$2; P=$3; S=$4; BR=$5; cd "$(dirname "$0")"
sed -i "/^Host $A\$/,/^  ServerAliveInterval 15\$/d" ~/.ssh/config
printf "\nHost %s\n  HostName %s\n  User root\n  Port %s\n  IdentityFile ~/.ssh/id_ed25519\n  StrictHostKeyChecking accept-new\n  ServerAliveInterval 15\n" $A $IP $P >> ~/.ssh/config
for t in 1 2 3 4 5 6; do timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=25 $A 'bash -s '$S < keys_r3.sh 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)" > keys_$S.out && grep -q PUBKEY keys_$S.out && break; sleep 20; done; cat keys_$S.out
printf '%s\n' "$HF" | timeout 60 ssh -o BatchMode=yes $A "bash -c \"\$(cat <<'X'
$(cat hf_r3.sh)
X
)\"" 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)"
timeout 400 ssh -o BatchMode=yes $A "bash -s $BR" < setup_r5.sh 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)"
bash inputs_r5.sh $A > inputs_r5_$S.log 2>&1
echo "PREP_DONE $S"
