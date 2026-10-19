# Results so far and review points

All values are test-set metrics in physical units unless stated. Split seed 0.

## 1. Plain MLP, one model per dataset (both targets in one model, single seed, 70/10/20)

| Dataset | Target | R² | RMSE | Max rel. error |
| --- | --- | --- | --- | --- |
| cone | Nu | 0.999998 | 4.53e-4 | 0.81% |
| cone | Sh | 0.999991 | 6.47e-4 | 0.53% |
| plate | Nu | 0.999997 | 4.44e-4 | 0.85% |
| plate | Sh | 0.999992 | 6.02e-4 | 0.81% |
| wedge | Nu | 0.999996 | 4.52e-4 | 0.85% |
| wedge | Sh | 0.999991 | 6.04e-4 | 0.80% |

Final losses: train about 1.4e-6, validation 5.3e-6 to 6.4e-6 (standardised units).

## 2. User's tuned run in Claude Code (as reported by the user)

| Model | Target | R² % | "Accuracy %" | MAPE % | "RMSE %" | "MAE %" | Max rel. err % | Within 0.1% | Within 0.5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cone | Nu | 99.9999 | 99.959 | 0.041 | 0.060 | 0.037 | 0.38 | 91.9% | 100% |
| cone | Sh | 99.9997 | 99.986 | 0.014 | 0.021 | 0.013 | 0.45 | 99.0% | 100% |
| plate | Nu | 99.9998 | 99.954 | 0.047 | 0.078 | 0.045 | 0.42 | 88.7% | 100% |
| plate | Sh | 99.9997 | 99.987 | 0.014 | 0.024 | 0.013 | 0.23 | 99.0% | 100% |
| wedge | Nu | 99.9998 | 99.952 | 0.048 | 0.083 | 0.047 | 0.45 | 88.1% | 100% |
| wedge | Sh | 99.9996 | 99.986 | 0.014 | 0.025 | 0.014 | 0.24 | 98.8% | 100% |

### Review of this table (from the chat)

1. **Was the tuning judged on validation or test error? This is the most important point.** If settings were compared using test metrics, the test set has been used for model selection and every number above is optimistic. Fix: keep the tuned settings, retrain with `--split-seed 1` (a fresh test set) and report that run.
2. **"Accuracy %" equals 100 − MAPE in every row** (100 − 0.041 = 99.959). It is not a regression metric and duplicates MAPE. Remove it.
3. **"RMSE %" and "MAE %" are undefined.** They look like RMSE/mean(y) × 100 but could also be divided by the range. Report absolute RMSE and MAE (same units as Nu, Sh), or state the formula in the caption.
4. **R² should be a decimal** (0.999999), not a percentage. Differences in the 5th–6th decimal carry no information; RMSE, max error and the "within" columns separate the models.
5. **The "within 0.1% / 0.5%" columns are the most informative ones; keep them.** Nu falls short of Sh at 0.1% because Nu goes down to 0.010, where a tiny absolute error is a large relative one. Check by sorting `predictions_test.csv` by relative error.
6. **Record what the tuning changed** (width, depth, learning rate, steps, seeds, activation, split). The methods section must state the final settings.
7. Compared with the PASR-Net results below, this tuned run had **lower Sh max relative error** (0.23–0.45% vs 0.29–0.61%) and **higher Nu max relative error** (0.38–0.45% vs 0.24–0.27%). Combining the tuned settings with the quadratic anchor for Nu should give the best of both. This is the main open experiment.

## 3. Quadratic-anchored residual design (PASR-Net), one model per dataset per target, 3 seeds

| Dataset | Target | Design | Plain MLP test RMSE (mean ± sd) | New design test RMSE (mean ± sd) | 3-seed ensemble RMSE | Ensemble R² | Ensemble max rel. error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cone | Nu | quad anchor | 3.61e-4 ± 2.3e-5 | 1.47e-4 ± 4.9e-6 | 1.11e-4 | 0.9999999 | 0.24% |
| cone | Sh | dual encoding | 8.82e-4 ± 8.8e-5 | 5.09e-4 ± 5.6e-6 | 4.25e-4 | 0.9999963 | 0.61% |
| plate | Nu | quad anchor | 3.73e-4 ± 2.9e-5 | 1.21e-4 ± 6.7e-6 | 9.39e-5 | 0.9999998 | 0.25% |
| plate | Sh | dual encoding | 5.71e-4 ± 4.5e-5 | 4.24e-4 ± 1.3e-5 | 3.39e-4 | 0.9999973 | 0.41% |
| wedge | Nu | quad anchor | 3.86e-4 ± 2.0e-5 | 1.22e-4 ± 6.9e-6 | 9.38e-5 | 0.9999998 | 0.27% |
| wedge | Sh | dual encoding | 5.83e-4 ± 5.9e-6 | 3.98e-4 ± 2.2e-5 | 3.06e-4 | 0.9999978 | 0.29% |

Validation-error improvement (what the choice was based on): Nu 2.1–2.8×, Sh 1.4–1.5×.

Observation: a single-target Sh model (plain MLP) was weaker than the joint Nu+Sh model of §1 (e.g. cone Sh 8.8e-4 vs 6.5e-4). Sh appears to benefit from being learned together with Nu. Open check: run `--auto` without `--per-target` and compare Sh validation error.

## 4. Recurrent models (cone, single seed, same split and training)

| Model | Parameters | Nu RMSE | Sh RMSE |
| --- | --- | --- | --- |
| MLP 4×128 SiLU | 51,330 | 4.5e-4 | 6.5e-4 |
| GRU, parameters as a sequence | 50,562 | 1.05e-3 | 1.59e-3 |
| tanh RNN (as in the fin paper), file order | 17,026 | 1.27e-3 | 2.32e-3 |
| same, shuffled order | 17,026 | 1.37e-3 | 1.78e-3 |
| tanh RNN, all inputs at once | 18,306 | 1.55e-3 | 2.49e-3 |

## 5. Shared model (one network for all geometries, 80/20 split, for reference)

Random split: mean R² 0.999998, RMSE 2.8e-4 – 4.7e-4. Corners split: mean R² 0.999988, worst relative error 1.1%.

## 6. Open items, in priority order

1. Find out whether the user's tuning used validation or test error; if test, rerun with `--split-seed 1`.
2. Apply the quadratic anchor to Nu on top of the user's tuned settings; keep the better Sh setup (tuned joint model vs dual encoding) by validation error.
3. Error bars: repeat the final configuration over split seeds 0–4 and report mean ± sd.
4. Paper table: R² (decimal), RMSE, MAE, MAPE, max relative error, within 0.1%, within 0.5%. No "accuracy %".
5. Confirm with the user whether the wedge angle should be an input.
