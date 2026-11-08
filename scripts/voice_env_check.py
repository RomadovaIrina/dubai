#!/usr/bin/env python3
"""r16 T: the ACCEPTED-VOICE environment guard (08.11: two fresh r16 boxes came up with Chatterbox HEAD ab55923 instead of the accepted 5de7a54
and without noisereduce -> a different voice / a crashed remix). Run by setup_dabai.sh (stage 10) and by e2e.sh before the audio stage.
    voice_env_check.py [--python /venv/dabai/bin/python]      exit 0 = the accepted voice stack; 3 = mismatch (stderr says what)"""
import argparse, json, subprocess, sys
ACCEPTED_CHATTERBOX = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"   # user-accepted voice (30.10); infra/docker/requirements.dabai.lock.txt
ap = argparse.ArgumentParser(); ap.add_argument("--python", default="/venv/dabai/bin/python"); a = ap.parse_args()
code = r'''
import json, importlib.metadata as m
d = m.distribution("chatterbox-tts"); u = d.read_text("direct_url.json") or "{}"
c = json.loads(u).get("vcs_info", {}).get("commit_id", "")
try:
    import noisereduce; nr = noisereduce.__version__ if hasattr(noisereduce, "__version__") else m.version("noisereduce")
except Exception as e: nr = "MISSING: " + str(e)[:80]
print(json.dumps({"chatterbox_commit": c, "noisereduce": nr}))'''
r = subprocess.run([a.python, "-c", code], capture_output=True, text=True)
if r.returncode:
    print(f"VOICE_ENV_FAIL: cannot inspect {a.python}: {r.stderr[-300:]}", file=sys.stderr); sys.exit(3)
v = json.loads(r.stdout.strip().splitlines()[-1]); bad = []
if v["chatterbox_commit"] != ACCEPTED_CHATTERBOX: bad.append(f"chatterbox-tts commit {v['chatterbox_commit'][:7] or '?'} != accepted {ACCEPTED_CHATTERBOX[:7]}")
if str(v["noisereduce"]).startswith("MISSING"): bad.append(f"noisereduce {v['noisereduce']}")
if bad:
    print("VOICE_ENV_FAIL: " + "; ".join(bad) + "  -> fix: uv pip install --python " + a.python + " --no-deps -r infra/docker/requirements.dabai.lock.txt", file=sys.stderr)
    sys.exit(3)
print(f"VOICE_ENV_OK chatterbox {v['chatterbox_commit'][:7]} noisereduce {v['noisereduce']}")
