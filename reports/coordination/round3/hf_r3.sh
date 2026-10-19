read -r T
mkdir -p ~/.cache/huggingface /workspace/dub
printf '%s' "$T" > ~/.cache/huggingface/token
for f in /workspace/.env /workspace/dub/.env; do { grep -v '^HF_TOKEN=' "$f" 2>/dev/null; echo "HF_TOKEN=$T"; } > "$f.tmp" && mv "$f.tmp" "$f"; chmod 600 "$f"; done
chmod 600 ~/.cache/huggingface/token
echo "$(hostname) HF user: $(curl -s -m 15 -H "Authorization: Bearer $T" https://huggingface.co/api/whoami-v2 | grep -o '"name":"[^"]*"' | head -1)"
unset T
