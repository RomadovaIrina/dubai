#!/bin/bash
# r16 Q: e2e_audio.sh = scripts/r9/e2e.sh with the face block (LatentSync face_gen / face_comp) cut out, so the audio path stays byte-identical
# to the accepted e2e; the mux uses the source picture; metrics stage off. Regenerate after any e2e.sh change: bash scripts/r16q/mk_e2e_audio.sh
cd "$(dirname "$0")/../.."
python3 - <<"PY"
s = open("scripts/r9/e2e.sh").read().split("\n")
a = s.index("VID=\"\""); b = next(i for i, l in enumerate(s) if l.startswith("[ -n \"$VID\" ] || VID=$SRC"))
s[a:b + 1] = ["VID=$SRC; FL+=(\"face_skipped_r16q\")   # r16 Q: the face is made by KeySync fast2 + ks_plain afterwards (scripts/r16q/q_face.sh)"]
s = [l.replace("[ \"${E2E_METRICS:-on}\" = off ] && finish", "finish   # r16 Q: no gate rows") for l in s]
open("scripts/r16q/e2e_audio.sh", "w").write("\n".join(s))
PY
