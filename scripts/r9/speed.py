#!/usr/bin/env python3
"""r9: the speed BASELINE of the integrated E2E (production only, one video at a time per GPU), finer than the e2e stages:
  audio     = base pipeline (ASR / diarization / Qwen 14B / Chatterbox, r4 stage table) + the one-process retake / clause / burst-align stage + the rest
              (close_inner / soften / lip placement / hush / Spleeter remix);
  sep       = htdemucs stems (GPU) + the htdemucs remix;
  face_gen  = LatentSync (+ CodeFormer, MediaPipe prep) + the generator run's OWN composite (r4_video.sh: output discarded by the m19b comp);
  face_comp = cf detail + layer SR + fallback + the final composite.
    python scripts/r9/speed.py --out reports/experiments/r9-e2e/speed runs/*/manifest.json"""
from __future__ import annotations
import argparse, json, pathlib, re

A = pathlib.Path("/workspace/dub/artifacts/r4a"); B = pathlib.Path("/workspace/dub/artifacts/r4m")


def jl(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except Exception: return {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--server", default=""); ap.add_argument("manifests", nargs="+"); a = ap.parse_args()
    rows = []
    for p in a.manifests:
        m = jl(p); i = m["id"]; t = m["timing"]; st = t["stage_wall_s"]; vmin = t["video_min"]
        base = {}; lp = A / f"r9_{i}pre" / "runs" / f"{i}.log"
        if lp.exists():
            sec = lp.read_text().split("== stage wall times (s)")[-1]
            for k, v in re.findall(r"^\s+(\w+)\s+([\d.]+)$", sec, re.M): base[k] = float(v)
        tt = {}; tp = A / f"r9_{i}src" / "times.txt"
        if tp.exists():
            for k, v in re.findall(r"^(\w+) ([\d.]+) s", tp.read_text(), re.M): tt[k] = float(v)
        g = (jl(B / f"r9g_{i}" / f"{i}_r9g_{i}.manifest.json").get("r4v") or {}).get("stage_wall_s") or {}
        c = (jl(B / f"r9f_{i}" / f"{i}_r9f_{i}.manifest.json").get("r4v") or {}).get("stage_wall_s") or {}
        stem = jl(next(iter(sorted(pathlib.Path("/workspace/dub/r8a/stems_htdemucs").glob(f"{i}/sep.json"))), "")) if pathlib.Path(f"/workspace/dub/r8a/stems_htdemucs/{i}/sep.json").exists() else {}
        aud = st.get("audio", 0); bt = base.get("total"); one = tt.get("ONEPROC_DONE")
        r = {"id": i, "server": a.server, "dur_s": round(t["duration_s"], 1),
             "audio": aud, "audio.base_pipeline": bt, "audio.base.tts": base.get("tts"), "audio.base.asr+diar": round(base.get("asr", 0) + base.get("diarization", 0), 1) if base else None,
             "audio.base.loads": round(sum(base.get(k, 0) for k in ("translation_load", "tts_load", "qa_load")), 1) if base else None,
             "audio.oneproc(retake/clause/align)": one, "audio.rest(place/hush/soften/remix)": round(aud - (tt.get("BASE_DONE") or 0) - (one or 0), 1) if tt else None,
             "sep": st.get("sep"), "sep.htdemucs_stems": stem.get("separate_s"),
             "face_gen": st.get("face_gen"), "face_gen.latentsync": g.get("latentsync_25fps"), "face_gen.own_composite(discarded)": g.get("composite_source_fps"),
             "face_comp": st.get("face_comp"), "face_comp.sr+detail+fallback+composite": c.get("layer_sr_and_composite"),
             "production": round(sum(st.get(k, 0) for k in ("precheck", "upright", "audio", "sep", "face_gen", "face_comp", "mux")), 1)}
        rows.append({**r, **{k + " /vmin": (round(v / vmin, 1) if isinstance(v, (int, float)) and vmin else None) for k, v in r.items() if k not in ("id", "server", "dur_s")}})
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    keys = ["id", "server", "dur_s"] + [k for k in rows[0] if k.endswith(" /vmin")]
    tot = {k: sum((r.get(k[:-6]) or 0) for r in rows) for k in keys[3:]}; vm = sum(r["dur_s"] for r in rows) / 60
    fmt = lambda v: "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(k.replace(" /vmin", "") for k in keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    md += ["| **all (time-weighted)** | | " + f"{vm * 60:.0f}" + " | " + " | ".join(f"**{v / vm:.0f}**" for v in tot.values()) + " |"]
    out.with_suffix(".md").write_text("s/video-min, production only (scoring excluded).\n\n" + "\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
