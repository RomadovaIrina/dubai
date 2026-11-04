#!/usr/bin/env python3
"""r14: write reports/experiments/r14-distill/DATA_SOURCES.md from the download records (data/nasa/selected.json, GRID speakers used).
    python3 scripts/r14/mk_sources.py --data /workspace/dub/r14d/data --out reports/experiments/r14-distill/DATA_SOURCES.md [--grid s1 ...]"""
import argparse, json, pathlib


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--grid", nargs="*", default=["s1"])
    a = ap.parse_args(); D = pathlib.Path(a.data)
    sel = json.loads((D / "nasa" / "selected.json").read_text()) if (D / "nasa" / "selected.json").exists() else []
    gb = lambda p: sum(f.stat().st_size for f in p.glob("*") if f.is_file()) / 2**30 if p.exists() else 0.0
    L = ["# Training data sources: student round 2 (r14)", "",
         "Approved by the user on 04.11.2026 (\"разрешаю использовать чистые датасеты на обучение\"). Used only to train the LatentSync student LoRA;",
         "no clip is republished. The held-out test clips old03 / m03 / m18 are never used.", "",
         "| id | source | URL | licence | attribution |", "|---|---|---|---|---|",
         "| own_* | our own pipeline inputs (the 11 non-test clips) | – | own | – |"]
    for s in a.grid:
        L.append(f"| grid_{s} | GRID Audio-Visual Speech Corpus, speaker {s} | https://zenodo.org/records/3625687 | CC BY 4.0 | M. Cooke, J. Barker, S. Cunningham, X. Shao, "
                 f"\"The Grid Audio-Visual Speech Corpus\" (2006), Zenodo, doi:10.5281/zenodo.3625687, CC BY 4.0 |")
    for i, o in enumerate(sel, 1):
        L.append(f"| x_n{i:02d} | NASA Image and Video Library: {o['title'][:90]} | {o['page']} | US public domain (NASA media usage guidelines) | {o['attribution'][:170]} |")
    L += ["", "Notes:",
          "- **NASA:** the material is not copyrighted in the US. The NASA media guidelines ask not to imply endorsement and to respect the likeness of identifiable people: training use only. NASA IDs and file URLs are in `data/nasa/selected.json` on the server.",
          "- **EC Audiovisual Service:** approved (CC BY 4.0 / © European Union only), **not used yet**. The site serves the licence notice and the files through a JS API, so each item needs its own licence check before download.",
          f"- **Downloaded so far:** NASA {gb(D / 'nasa'):.1f} GB, GRID {gb(D / 'grid'):.1f} GB (limit ~50 GB in total)."]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print(f"wrote {a.out}: {len(sel)} NASA + {len(a.grid)} GRID")


if __name__ == "__main__":
    main()
