# Clean-Room Project-Fit Racing Simulator (`self_jev/sim`)

Original PyTorch-vectorized racing simulator for `self_jev`, calibrated against the user's own telemetry logs and official `agx-racing-kit` mapdata.

## Telemetry Schema & Observed Ranges (Step 1 Audit)

Inspected from:
- `1st/training-61bb13c7-8b8f-47e4-aca9-70920619993a.jsonl` (13,942 race rows, Parkland, `track_wetness = 1.0`, 4 laps)
- `data/training-ebe0e4b9-c62f-415c-8000-4aa51f9f662a.jsonl` (10,301 race rows, Parkland, `track_wetness = 0.25`)

| Signal | Units / Semantics | Observed Range (`61bb13c7` wet / `ebe0e4b9` damp) |
|---|---|---|
| Decision `dt` | seconds (`hold_ticks = 1`, 30 Hz) | `0.03333 s` (2 × 60 Hz physics substeps) |
| `speed_mps`, `vx` | m/s | `0.0 .. 92.80` (wet) / `0.0 .. 94.65` (damp) |
| `vy` | m/s (body lateral) | `-1.46 .. +1.58` (wet) / `-10.43 .. +1.70` (damp) |
| `yaw_rate` | rad/s | `-1.42 .. +1.43` (wet) / `-1.60 .. +3.07` (damp) |
| `g_lat` | g | `-2.44 .. +2.40` (wet) / `-3.27 .. +3.32` (damp) |
| `g_long` | g | `-3.70 .. +1.79` (wet) / `-4.67 .. +1.96` (damp) |
| `ers_soc_mj` | MJ | `0.03 .. 3.00` (`soc_window_mj = 4.0`, `recharge_mj = 8.5`) |
| `fuel_kg` | kg | `9.00 -> ~4.18` over 4 laps (~1.2 kg/lap) |
| `gear`, `rpm` | integer `1..8`, rev/min | `0 .. 15,000` |

## Track Geometry Source

Loaded from `data/imports/comp-racing-history-20261117/sources/agx-racing-kit/data/racing-<track>.mapdata.json` (`parkland`, `highland`, `crossover`):
- Uniform arc-length centerline (`~2.0 m` spacing) with `s_m`, `x_m`, `y_m`, `heading_rad`, `curvature_1pm`, `grade_dz_ds`, `half_width_left_m`, `half_width_right_m`.
- `pit` (`entry_s`, `exit_s`, `speed_limit_kph`) and `start_grid` (22 grid slots).
