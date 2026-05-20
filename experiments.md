# CNN ablation experiments

## Baseline (volume + moving average)

Config:
- INCLUDE_VOL = true
- INCLUDE_MA = true

Results:
- accuracy = 0.4972
- f1 = 0.5676
- auc = 0.5209
- brier = 0.2800

---

## No volume

Config:
- INCLUDE_VOL = false
- INCLUDE_MA = true

Results:
- accuracy = 0.5482
- f1 = 0.6888
- auc = 0.5470
- brier = 0.3191

---

## No moving average

Config:
- INCLUDE_VOL = true
- INCLUDE_MA = false

Results:
- accuracy = 0.4858
- f1 = 0.3552
- auc = 0.5167
- brier = 0.3092