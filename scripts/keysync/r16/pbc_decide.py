#!/usr/bin/env python3
"""r17 KS_PROBE_BY_CHECK: the delay-probe decision taken from the post-render SyncNet CHECK instead of a separate 10 s probe render.
The render was made with the dub delayed by D0 ms (default 40); the check measures that render against the ORIGINAL dub, so the offset the probe
would have measured is  Op = Oc - D0 / 40  (m03: Oc 0, D0 40 -> Op -1 -> delay 40 = keep). Then the probe's own rules (ks_face.sh), in its order:
  no result -> delay 0 (flag ks_probe_failed) | conf < KS_OFFSCREEN_CONF (1.5) -> off-screen voice (source picture) |
  conf < KS_PROBE_MIN_CONF (3) -> delay 0 (flag ks_probe_low_conf) | |Op| > KS_PROBE_MAX (2) -> clamped (flag ks_probe_clamped);  delay = -Op * 40
Output (one line): "keep <D0> ok" | "rerender <delay> <flag or ok>" | "offscreen 0 ks_offscreen_voice".
The conf comes from the full render, not a 10 s window: whether that changes a decision is what scripts/keysync/r16/tz/probe_proof.sh measures.
    python3 pbc_decide.py --check-offset <int | x> --conf <float> --delay <D0 ms>"""
import argparse, os


def decide(oc, conf, d0, offscreen=None, min_conf=None, pmax=None):
    offscreen = float(os.environ.get("KS_OFFSCREEN_CONF", "1.5")) if offscreen is None else offscreen
    min_conf = float(os.environ.get("KS_PROBE_MIN_CONF", "3")) if min_conf is None else min_conf
    pmax = int(os.environ.get("KS_PROBE_MAX", "2")) if pmax is None else pmax
    if oc in (None, "x", ""):
        delay, flag = 0, "ks_probe_failed"
    elif conf < offscreen:
        return "offscreen", 0, "ks_offscreen_voice"
    elif conf < min_conf:
        delay, flag = 0, "ks_probe_low_conf"
    else:
        op = int(oc) - round(d0 / 40); flag = "ok"
        if abs(op) > pmax: op = pmax if op > 0 else -pmax; flag = "ks_probe_clamped"
        delay = -op * 40
    return ("keep", d0, "ok") if delay == d0 and flag == "ok" else ("rerender", delay, flag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check-offset", required=True); ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--delay", type=int, default=40); a = ap.parse_args()
    print(*decide(a.check_offset, a.conf, a.delay))
