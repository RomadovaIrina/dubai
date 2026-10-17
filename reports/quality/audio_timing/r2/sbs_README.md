# Agent C r2 listening clips (2026-10-17)

Every file: ORIGINAL video frames + one audio variant. **Lips are NOT re-synced** (LatentSync not re-run): the mouth is the original Russian one, judge the AUDIO (timing against the speaker, pauses, foreign sounds, bed).

B_old = cand5 timing + HDEMUCS bed with the old duck = what was heard before (HDEMUCS is a non-commercial reference, never shipped).
C_new = the r2 delivery (`/workspace/dub/artifacts/audio-r2/<id>/final_mix_aac.m4a`): r2 timing + accepted refit units + Spleeter bed, residue duck, -4 dB / 250 ms duck, declick.

| files | variants | what to listen for |
|---|---|---|
| `03_024.00-026.00s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | reviewed window: 24.5–25.0 s gap (dub silent while the speaker talks) 0.5 → 0.1 s |
| `03_034.00-037.00s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | reviewed window: 34.3–35.5 s gap 1.2 → 0.9 s (u04 translation length; layer 3 found no faithful rewrite) |
| `01_003.40-006.90s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 4.9–5.4 s (0.5 s, 13 LS frames, inner): source says "застройщиком." |
| `01_047.80-051.90s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 49.3–50.4 s (1.1 s, 0 LS frames, tail): source says "финишным покрытием." |
| `02_007.50-011.40s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 9.0–9.9 s (0.9 s, 0 LS frames, inner): source says "буквально уже тут" |
| `02_079.70-083.60s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 81.2–82.1 s (0.9 s, 0 LS frames, inner): source says "бане все" |
| `03_065.30-069.60s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 66.8–68.1 s (1.3 s, 33 LS frames, inner): source says "работать наша система" |
| `03_033.10-037.00s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 34.6–35.5 s (0.9 s, 23 LS frames, tail): source says "нам всем тот" |
| `04_017.60-021.20s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 19.1–19.7 s (0.6 s, 16 LS frames, tail): source says "и" |
| `04_025.20-028.80s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 26.7–27.3 s (0.6 s, 16 LS frames, inner): source says "приложения" |
| `05_019.10-022.70s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 20.6–21.2 s (0.6 s, 15 LS frames, inner): source says "возможностей" |
| `05_027.10-030.70s_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | worst remaining gap 28.6–29.2 s (0.6 s, 15 LS frames, tail): source says "формате" |
| `01_085.66-090.42s_D_r2_without_refit` / `_E_r2_with_refit` | before / after | layer-3 refit u23: "Waterproof paints can offer up to 10 years of service." |
| `04_047.20-053.34s_D_r2_without_refit` / `_E_r2_with_refit` | before / after | layer-3 refit u05: "The weather is really quite nice, isn't it? That's great." |
| `01_040.78-043.78s_F_without_declick` / `_G_with_declick` | before / after | click at 42.28 s (01 42.28: atempo splice; others: inside the raw Chatterbox take) before / after declick |
| `01_012.82-015.82s_F_without_declick` / `_G_with_declick` | before / after | click at 14.32 s (01 42.28: atempo splice; others: inside the raw Chatterbox take) before / after declick |
| `05_040.49-043.49s_F_without_declick` / `_G_with_declick` | before / after | click at 41.99 s (01 42.28: atempo splice; others: inside the raw Chatterbox take) before / after declick |
| `02_037.50-041.50s_H_without_residue_duck` / `_I_with_residue_duck` | before / after | Spleeter left a vocal sound in the bed at 39.0–39.4 s (dub silent): without / with the residue duck |
| `03_006.30-010.30s_J_old_duck` / `_K_new_duck` | before / after | bed step before a dub phrase at 8.3 s: old duck (-6 dB, 80 ms attack) vs new (-4 dB, 250 ms) |
| `03_012.30-016.30s_J_old_duck` / `_K_new_duck` | before / after | bed step before a dub phrase at 14.3 s: old duck (-6 dB, 80 ms attack) vs new (-4 dB, 250 ms) |
| `04_006.50-010.50s_J_old_duck` / `_K_new_duck` | before / after | bed step before a dub phrase at 8.5 s: old duck (-6 dB, 80 ms attack) vs new (-4 dB, 250 ms) |
