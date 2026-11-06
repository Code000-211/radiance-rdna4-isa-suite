# Offline kernel plan

Target: `host`. No GPU executed. No correctness certification.

| Workload | Suggested candidate | Basis | Decision |
|---|---|---|---|
| cpu-M5-N16-K96 | libref | calibrated upper envelope | NEEDS_MEASUREMENT |
| cpu-M5-N96-K64 | libref | calibrated upper envelope | NEEDS_MEASUREMENT |
| cpu-M31-N64-K128 | libref | calibrated upper envelope | CALIBRATED_SEPARATION |
| cpu-M32-N64-K32 | libref | calibrated upper envelope | CALIBRATED_SEPARATION |
| cpu-M8-N32-K32 | libref | calibrated upper envelope | NEEDS_MEASUREMENT |
| cpu-M1-N64-K96 | panel16 | calibrated upper envelope | NEEDS_MEASUREMENT |
| cpu-M16-N16-K32 | libref | calibrated upper envelope | NEEDS_MEASUREMENT |
| cpu-M33-N32-K32 | libref | calibrated upper envelope | NEEDS_MEASUREMENT |
| cpu-M2-N64-K128 | libref | calibrated upper envelope | NEEDS_MEASUREMENT |
| cpu-M5-N32-K128 | libref | calibrated upper envelope | NEEDS_MEASUREMENT |

## cpu-M5-N16-K96

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 1.9178 | 1.0000 | 80 |
| panel16 | [1, 16, 1] | 2.5263 | 1.0000 | 5 |
| panel1 | [1, 1, 1] | 4.4649 | 1.0000 | 80 |
| panel4 | [1, 4, 1] | 6.8412 | 1.0000 | 20 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: libref, panel16

## cpu-M5-N96-K64

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 6.2618 | 1.0000 | 480 |
| panel16 | [1, 16, 1] | 10.0274 | 1.0000 | 30 |
| panel1 | [1, 1, 1] | 17.8344 | 1.0000 | 480 |
| panel4 | [1, 4, 1] | 27.3648 | 1.0000 | 120 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: libref, panel16

## cpu-M31-N64-K128

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 27.3066 | 1.0000 | 1984 |
| panel16 | [1, 16, 1] | 81.6105 | 1.0000 | 124 |
| panel1 | [1, 1, 1] | 147.3698 | 1.0000 | 1984 |
| panel4 | [1, 4, 1] | 226.2160 | 1.0000 | 496 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: none for ranking; numerical validation still required

## cpu-M32-N64-K32

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 12.2171 | 1.0000 | 2048 |
| panel16 | [1, 16, 1] | 21.8068 | 1.0000 | 128 |
| panel1 | [1, 1, 1] | 38.0372 | 1.0000 | 2048 |
| panel4 | [1, 4, 1] | 58.3783 | 1.0000 | 512 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: none for ranking; numerical validation still required

## cpu-M8-N32-K32

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 2.2413 | 1.0000 | 256 |
| panel16 | [1, 16, 1] | 2.7701 | 1.0000 | 16 |
| panel1 | [1, 1, 1] | 4.7620 | 1.0000 | 256 |
| panel4 | [1, 4, 1] | 7.2973 | 1.0000 | 64 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: libref, panel16

## cpu-M1-N64-K96

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| panel16 | [1, 16, 1] | 2.0312 | 1.0000 | 4 |
| libref | [1, 1, 1] | 3.2978 | 1.0000 | 64 |
| panel1 | [1, 1, 1] | 3.5736 | 1.0000 | 64 |
| panel4 | [1, 4, 1] | 5.4730 | 1.0000 | 16 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: panel16, libref

## cpu-M16-N16-K32

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 2.1487 | 1.0000 | 256 |
| panel16 | [1, 16, 1] | 2.7701 | 1.0000 | 16 |
| panel1 | [1, 1, 1] | 4.7620 | 1.0000 | 256 |
| panel4 | [1, 4, 1] | 7.2973 | 1.0000 | 64 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: libref, panel16

## cpu-M33-N32-K32

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 6.6948 | 1.0000 | 1056 |
| panel16 | [1, 16, 1] | 11.2686 | 1.0000 | 66 |
| panel1 | [1, 1, 1] | 19.6170 | 1.0000 | 1056 |
| panel4 | [1, 4, 1] | 30.1013 | 1.0000 | 264 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: libref, panel16

## cpu-M2-N64-K128

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 4.9519 | 1.0000 | 128 |
| panel16 | [1, 16, 1] | 5.3125 | 1.0000 | 8 |
| panel1 | [1, 1, 1] | 9.5156 | 1.0000 | 128 |
| panel4 | [1, 4, 1] | 14.5946 | 1.0000 | 32 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: libref, panel16

## cpu-M5-N32-K128

| Candidate | Tile | Score (us) | Padding efficiency | Groups |
|---|---|---:|---:|---:|
| libref | [1, 1, 1] | 3.9708 | 1.0000 | 160 |
| panel16 | [1, 16, 1] | 6.6280 | 1.0000 | 10 |
| panel1 | [1, 1, 1] | 11.8924 | 1.0000 | 160 |
| panel4 | [1, 4, 1] | 18.2432 | 1.0000 | 40 |

Score is a scenario unless explicitly calibrated. No inferred sparse boost; storage type is not compute type.

Measurements requested: libref, panel16
