#!/usr/bin/env python3
"""r5 audio gate axis "faithfulness": the SAME Qwen judge as dur_translate.judge (back-translation + meaning score 1-5, missing / added items) on the
final English text of every unit (the words actually synthesised and placed), so pb1 and every r5 candidate are scored identically. Adds
"faith" = {mean, lt4, rows} to <tag>/<id>.metrics.json.
    python scripts/r5/r5_faith.py <tag> [ids]"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT / "scripts" / "pilot"), str(ROOT / "scripts" / "quality")]
A = pathlib.Path("/workspace/dub/artifacts/r4a")


def main():
    tag = sys.argv[1]; ids = sys.argv[2:] or ["01", "02", "03", "04", "05"]
    import run_clean_pipeline_05 as rc
    import dur_translate as dt
    import os
    os.environ["DUB_QWEN_SIZE"] = os.environ.get("DUB_FAITH_QWEN", "14b")   # the gate judge: the stronger model, the same for every version scored
    rc.QWEN_GPU_LAYERS = -1; llm, _ = rc.load_llm({})
    for id_ in ids:
        mf = A / tag / id_ / f"{id_}_r4b.manifest.json"; mj = A / tag / f"{id_}.metrics.json"
        if not mf.exists() or not mj.exists(): print(f"skip {id_}: no manifest / metrics"); continue
        rows = []
        for u in json.loads(mf.read_text())["units"]:
            en = " ".join(t["text"] for t in u.get("tts_parts") or []) or u.get("translation_fitted") or u.get("translation", "")
            j = dt.judge(llm, "Russian", "English", u["text"], en); rows.append({"id": u["id"], "start": u["start"], "source": u["text"], "english": en, **j})
        sc = [r["score"] for r in rows]; m = json.loads(mj.read_text())
        m["faith"] = {"mean": round(sum(sc) / max(1, len(sc)), 3), "lt4": sum(1 for x in sc if x < 4), "rows": rows}
        mj.write_text(json.dumps(m, ensure_ascii=False, indent=1)); print(f"{tag} {id_} faith mean {m['faith']['mean']} lt4 {m['faith']['lt4']}", flush=True)


if __name__ == "__main__":
    main()
