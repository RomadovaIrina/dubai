#!/bin/bash
# Runs every component smoke test in pipeline order; writes reports/smoke.md and logs/check_<name>.log
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${DUB_ROOT}"; MD="${DUB_ROOT}/reports/smoke.md"
echo "# Smoke results $(date -Iseconds)" > "$MD"; echo >> "$MD"; echo "| component | result | time | log |" >> "$MD"; echo "|---|---|---|---|" >> "$MD"
TESTS=("chatterbox:python scripts/check_chatterbox.py" "faster-whisper:python scripts/check_whisper.py" "silero-vad:python scripts/check_vad.py"
       "pyannote:python scripts/check_pyannote.py" "librosa/atempo:python scripts/check_atempo.py" "llama.cpp:python scripts/check_llama.py"
       "retinaface:python scripts/check_retinaface.py" "latentsync:bash scripts/check_latentsync.sh" "codeformer:bash scripts/check_codeformer.sh"
       "ffmpeg-mux:bash scripts/check_ffmpeg_mux.sh")
if [ $# -gt 0 ]; then  # optional filter by component name(s), e.g. run_smoke.sh retinaface llama.cpp
  sel=" $* "; keep=()
  for t in "${TESTS[@]}"; do case "$sel" in *" ${t%%:*} "*) keep+=("$t");; esac; done
  TESTS=("${keep[@]}")
fi
for t in "${TESTS[@]}"; do
  name="${t%%:*}"; cmd="${t#*:}"; log="logs/check_${name//\//_}.log"; s=$(date +%s)
  if $cmd > "$log" 2>&1; then r=$(grep -q '^SKIP' "$log" && echo SKIP || echo PASS); else r=FAIL; fi
  echo "| $name | $r | $(( $(date +%s) - s ))s | $log |" >> "$MD"; echo "$name: $r"
done
cat "$MD"
