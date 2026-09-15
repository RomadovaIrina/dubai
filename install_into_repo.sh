#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-/workspace/dub}"
[ -d "$DST/scripts/pilot" ] || { echo "Not a DabAI repo: $DST/scripts/pilot missing" >&2; exit 1; }
cp -v "$SRC"/scripts/pilot/* "$DST/scripts/pilot/"
chmod +x "$DST"/scripts/pilot/*.py "$DST"/scripts/pilot/*.sh 2>/dev/null || true
echo
echo "Installed into $DST/scripts/pilot"
echo "Next: source $DST/scripts/env.sh && python $DST/scripts/pilot/pilot_status.py"
