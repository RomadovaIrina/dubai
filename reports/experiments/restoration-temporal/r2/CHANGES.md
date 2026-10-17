# Round 2, track B (composite side): changelog

Branch `experiment/r2-composite-position` (local, from 1a244cf). Server dub-B3 (RTX 5090). One entry per change: what / why / files / result, commit.

| # | what | why | files | result | commit |
|---|---|---|---|---|---|
| 1 | Common slide metric + its helpers copied in unchanged (track A: `slide_metric.py`, `anchor_filters.py`, `anchor_study.py`); new `composite_register.py`: register LatentSync's generated lower face onto the source frame before blending (DIS flow on nose base / cheeks / nasolabial bands, translation or RANSAC similarity), optional flow refinement on the paste ring, `nosecut` mask (current-frame landmarks, cut below the nose base, explicit 10-90 % edge); `--variant clean0` reproduces decompose S3nc bit for bit | both servers must measure slide the same way; LatentSync's mask starts at the nose anchor, so alae / nostrils / cheeks are generated in crop coordinates and slide with the crop | `scripts/quality/{slide_metric,anchor_study,composite_register}.py`, `scripts/pilot/anchor_filters.py` | pending (inputs being regenerated) | (this) |
