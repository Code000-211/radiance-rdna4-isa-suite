# Offline kernel plan

Target: `gfx1201`. No GPU executed. No correctness certification.

| Workload | Suggested candidate | Basis | Decision |
|---|---|---|---|
| dense-M1-N2560-K640 | w4a8-prefill-tile27 | uncalibrated analytic scenario | NEEDS_MEASUREMENT |
| dense-M5-N2560-K640 | w4a8-prefill-tile27 | uncalibrated analytic scenario | NEEDS_MEASUREMENT |
| dense-M64-N2560-K640 | w4a8-prefill-tile13 | uncalibrated analytic scenario | NEEDS_MEASUREMENT |
| dense-M128-N2560-K640 | w4a8-prefill-tile10 | uncalibrated analytic scenario | NEEDS_MEASUREMENT |
| dense-M129-N2560-K640 | w4a8-prefill-tile27 | uncalibrated analytic scenario | NEEDS_MEASUREMENT |
| dense-M256-N2560-K640 | w4a8-prefill-tile0 | uncalibrated analytic scenario | NEEDS_MEASUREMENT |
| dense-M1024-N2560-K640 | w4a8-prefill-tile0 | uncalibrated analytic scenario | NEEDS_MEASUREMENT |

## dense-M1-N2560-K640

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| w4a8-prefill-tile27 | [128, 256, 64] | 8.4938 | 0.0078 | 10 |
| w4a8-prefill-tile28 | [128, 256, 64] | 8.4938 | 0.0078 | 10 |
| w4a8-prefill-tile29 | [128, 256, 64] | 8.4938 | 0.0078 | 10 |
| w4a8-prefill-tile30 | [128, 256, 64] | 8.4938 | 0.0078 | 10 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: w4a8-prefill-tile27, w4a8-prefill-tile28, w4a8-prefill-tile29, w4a8-prefill-tile30

Rejected:
- `w4a8-prefill-tile0`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile2`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile3`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile5`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile6`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile8`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile9`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile10`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile13`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile14`: source-known scalar Ascale tail read: candidate excluded for this M

## dense-M5-N2560-K640

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| w4a8-prefill-tile27 | [128, 256, 64] | 8.4938 | 0.0391 | 10 |
| w4a8-prefill-tile28 | [128, 256, 64] | 8.4938 | 0.0391 | 10 |
| w4a8-prefill-tile29 | [128, 256, 64] | 8.4938 | 0.0391 | 10 |
| w4a8-prefill-tile30 | [128, 256, 64] | 8.4938 | 0.0391 | 10 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: w4a8-prefill-tile27, w4a8-prefill-tile28, w4a8-prefill-tile29, w4a8-prefill-tile30

Rejected:
- `w4a8-prefill-tile0`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile2`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile3`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile5`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile6`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile8`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile9`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile10`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile13`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile14`: source-known scalar Ascale tail read: candidate excluded for this M

## dense-M64-N2560-K640

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| w4a8-prefill-tile13 | [64, 256, 64] | 5.9161 | 1.0000 | 10 |
| w4a8-prefill-tile8 | [64, 256, 64] | 5.9161 | 1.0000 | 10 |
| w4a8-prefill-tile27 | [128, 256, 64] | 8.4938 | 0.5000 | 10 |
| w4a8-prefill-tile28 | [128, 256, 64] | 8.4938 | 0.5000 | 10 |
| w4a8-prefill-tile29 | [128, 256, 64] | 8.4938 | 0.5000 | 10 |
| w4a8-prefill-tile30 | [128, 256, 64] | 8.4938 | 0.5000 | 10 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: w4a8-prefill-tile13, w4a8-prefill-tile8, w4a8-prefill-tile27, w4a8-prefill-tile28, w4a8-prefill-tile29, w4a8-prefill-tile30

Rejected:
- `w4a8-prefill-tile0`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile2`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile3`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile5`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile6`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile9`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile10`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile14`: source-known scalar Ascale tail read: candidate excluded for this M

## dense-M128-N2560-K640

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| w4a8-prefill-tile10 | [128, 128, 64] | 7.4066 | 1.0000 | 20 |
| w4a8-prefill-tile13 | [64, 256, 64] | 7.4180 | 1.0000 | 20 |
| w4a8-prefill-tile8 | [64, 256, 64] | 7.4180 | 1.0000 | 20 |
| w4a8-prefill-tile0 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile2 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile27 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile28 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile29 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile3 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile30 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile5 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |
| w4a8-prefill-tile6 | [128, 256, 64] | 8.4938 | 1.0000 | 10 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: w4a8-prefill-tile10, w4a8-prefill-tile13, w4a8-prefill-tile8, w4a8-prefill-tile0, w4a8-prefill-tile2, w4a8-prefill-tile27, w4a8-prefill-tile28, w4a8-prefill-tile29, w4a8-prefill-tile3, w4a8-prefill-tile30, w4a8-prefill-tile5, w4a8-prefill-tile6

Rejected:
- `w4a8-prefill-tile9`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile14`: source-known scalar Ascale tail read: candidate excluded for this M

## dense-M129-N2560-K640

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| w4a8-prefill-tile27 | [128, 256, 64] | 8.5962 | 0.5039 | 20 |
| w4a8-prefill-tile28 | [128, 256, 64] | 8.5962 | 0.5039 | 20 |
| w4a8-prefill-tile29 | [128, 256, 64] | 8.5962 | 0.5039 | 20 |
| w4a8-prefill-tile30 | [128, 256, 64] | 8.5962 | 0.5039 | 20 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: w4a8-prefill-tile27, w4a8-prefill-tile28, w4a8-prefill-tile29, w4a8-prefill-tile30

Rejected:
- `w4a8-prefill-tile0`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile2`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile3`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile5`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile6`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile8`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile9`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile10`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile13`: source-known scalar Ascale tail read: candidate excluded for this M
- `w4a8-prefill-tile14`: source-known scalar Ascale tail read: candidate excluded for this M

## dense-M256-N2560-K640

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| w4a8-prefill-tile0 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile2 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile27 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile28 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile29 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile3 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile30 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile5 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile6 | [128, 256, 64] | 9.4789 | 1.0000 | 20 |
| w4a8-prefill-tile14 | [256, 128, 64] | 9.9274 | 1.0000 | 20 |
| w4a8-prefill-tile10 | [128, 128, 64] | 10.3989 | 1.0000 | 40 |
| w4a8-prefill-tile13 | [64, 256, 64] | 10.4217 | 1.0000 | 40 |
| w4a8-prefill-tile8 | [64, 256, 64] | 10.4217 | 1.0000 | 40 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: w4a8-prefill-tile0, w4a8-prefill-tile2, w4a8-prefill-tile27, w4a8-prefill-tile28, w4a8-prefill-tile29, w4a8-prefill-tile3, w4a8-prefill-tile30, w4a8-prefill-tile5, w4a8-prefill-tile6, w4a8-prefill-tile14, w4a8-prefill-tile10, w4a8-prefill-tile13, w4a8-prefill-tile8

Rejected:
- `w4a8-prefill-tile9`: source-known scalar Ascale tail read: candidate excluded for this M

## dense-M1024-N2560-K640

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| w4a8-prefill-tile0 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile2 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile27 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile28 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile29 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile3 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile30 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile5 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile6 | [128, 256, 64] | 24.6726 | 1.0000 | 80 |
| w4a8-prefill-tile13 | [64, 256, 64] | 28.4440 | 1.0000 | 160 |
| w4a8-prefill-tile8 | [64, 256, 64] | 28.4440 | 1.0000 | 160 |
| w4a8-prefill-tile14 | [256, 128, 64] | 28.5590 | 1.0000 | 80 |
| w4a8-prefill-tile10 | [128, 128, 64] | 30.5251 | 1.0000 | 160 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: w4a8-prefill-tile0, w4a8-prefill-tile2, w4a8-prefill-tile27, w4a8-prefill-tile28, w4a8-prefill-tile29, w4a8-prefill-tile3, w4a8-prefill-tile30, w4a8-prefill-tile5, w4a8-prefill-tile6, w4a8-prefill-tile13, w4a8-prefill-tile8, w4a8-prefill-tile14, w4a8-prefill-tile10

Rejected:
- `w4a8-prefill-tile9`: source-known scalar Ascale tail read: candidate excluded for this M
