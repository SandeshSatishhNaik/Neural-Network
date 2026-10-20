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

## 2. User's tuned run in Claude Code (as reported by the user) — not reproducible

**Status (19 Oct 2026):** the tuned code is lost and the changes were never recorded, so these numbers cannot be reproduced or described in a methods section. Do not cite this table. The paper numbers are in §7.

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

1. ~~Find out whether the user's tuning used validation or test error.~~ Closed: the tuned code is lost (§2).
2. ~~Apply the quadratic anchor on top of the user's tuned settings.~~ Dropped with item 1.
3. ~~Error bars over split seeds 0–4.~~ Done, §7.
4. ~~Paper table.~~ Done, §7 and `results_final/paper_table.csv`.
5. Confirm with the user whether the wedge angle should be an input.
6. ~~Sh near small Nb: test Nt/Nb as a feature.~~ Done, §8: adopted for Sh (general improvement of about 1.1–1.3×), but it does not remove the small-Nb difficulty. Still open: Sh in the Nt/Nb > 4 corner is 6–7× harder than elsewhere. Options: more solver points near Nb = 0.2 (best evidence would come from the data), or a loss weighted toward that corner (must be judged on validation).
7. Sh, quad anchor vs dual encoding: indistinguishable on validation at single-seed resolution (§7.1). If it matters for the paper, compare them with `--seeds 3` on validation.
8. Check 1 (new definition, 20 Oct 2026) cannot fail on split seeds 0–4, which define its band (§8.4). Suggested: band from the reference model's seeds 0–4, applied to new runs only, plus R² ≥ 0.9999. Awaiting the user's decision.

**Two claims from the chat sessions did not hold** (§7.1, §8): (1) "Sh does better in the joint Nu+Sh model": per-target won Sh on all three geometries once the PASR designs were available. (2) "Dual encoding is best for Sh": it beat the plain MLP but not the quadratic anchor. Neither should appear in the paper.

## 7. Final configuration (Claude Code session, 19 Oct 2026)

### 7.1 Design choice (split seed 0, validation only)

`python train_separate.py data/*.xlsx --per-target --auto --seeds 3 --out results_pasr`, then `analysis/select_design.py` keeps joint (Nu+Sh) or per-target per geometry and target, whichever has the lower validation RMSE (`results_final/final_config.csv`).

| Geometry | Target | Chosen | Val RMSE | Joint model design | Joint val RMSE | Joint / chosen |
| --- | --- | --- | --- | --- | --- | --- |
| cone | Nu | per-target, quad anchor | 1.51e-4 | dual encoding | 2.92e-4 | 1.93 |
| cone | Sh | per-target, quad anchor | 5.23e-4 | dual encoding | 5.49e-4 | 1.05 |
| plate | Nu | per-target, quad anchor | 1.21e-4 | quad anchor | 1.39e-4 | 1.14 |
| plate | Sh | per-target, quad anchor | 3.62e-4 | quad anchor | 4.12e-4 | 1.14 |
| wedge | Nu | per-target, quad anchor | 1.29e-4 | quad anchor | 1.47e-4 | 1.14 |
| wedge | Sh | per-target, quad anchor | 3.64e-4 | quad anchor | 4.20e-4 | 1.15 |

- Per-target wins everywhere, but apart from cone Nu every margin is below the ~20% single-run noise level. The earlier observation "Sh does better in the joint model" (§3) was not reproduced; with quad anchor or dual encoding the joint advantage disappears.
- Per-target Sh `--auto` picked the quad anchor on all three geometries (validation NRMSE vs dual encoding: cone 2.45e-3 vs 2.99e-3, plate 2.01e-3 vs 2.34e-3, wedge 2.16e-3 vs 2.24e-3, single seed). The earlier "dual encoding is best for Sh" was measured against a plain MLP, not against the quad anchor.
- The joint cone model picked dual encoding with all four candidates within 15% of each other: a noise-level choice.

### 7.2 Paper table (test set, 3-seed ensemble, per-target quad-anchor design)

Split seed 0, and mean ± sd over split seeds 0–4 (design fixed by the split-0 validation choice; seeds 1–4 are test sets no decision has touched). RMSE and MAE are absolute, in units of Nu/Sh. Source: `results_final/paper_table.csv`, `results_final/paper_table.md`.

| Geometry | Target | Split | R² | RMSE | MAE | MAPE % | Max rel. err. % | Within 0.1 % | Within 0.5 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cone | Nu | seed 0 | 0.9999999 | 1.12e-04 | 7.24e-05 | 0.016 | 0.273 | 99.2 | 100.0 |
| | | mean ± sd | 0.9999998 ± 0.0000000 | 1.14e-04 ± 1.42e-05 | 7.10e-05 ± 6.71e-06 | 0.015 ± 0.001 | 0.251 ± 0.069 | 99.3 ± 0.3 | 100.0 ± 0.0 |
| cone | Sh | seed 0 | 0.9999956 | 4.64e-04 | 2.15e-04 | 0.014 | 0.795 | 98.2 | 99.9 |
| | | mean ± sd | 0.9999947 ± 0.0000022 | 4.97e-04 ± 9.82e-05 | 2.30e-04 ± 1.77e-05 | 0.015 ± 0.001 | 0.735 ± 0.460 | 98.0 ± 0.4 | 99.9 ± 0.1 |
| plate | Nu | seed 0 | 0.9999998 | 9.99e-05 | 5.58e-05 | 0.016 | 0.313 | 98.8 | 100.0 |
| | | mean ± sd | 0.9999998 ± 0.0000000 | 9.70e-05 ± 1.03e-05 | 5.41e-05 ± 3.76e-06 | 0.015 ± 0.001 | 0.265 ± 0.113 | 99.3 ± 0.4 | 100.0 ± 0.0 |
| plate | Sh | seed 0 | 0.9999970 | 3.61e-04 | 1.51e-04 | 0.010 | 0.500 | 98.9 | 100.0 |
| | | mean ± sd | 0.9999957 ± 0.0000038 | 4.04e-04 ± 1.63e-04 | 1.56e-04 ± 9.34e-06 | 0.010 ± 0.001 | 0.463 ± 0.212 | 99.3 ± 0.2 | 100.0 ± 0.0 |
| wedge | Nu | seed 0 | 0.9999998 | 1.03e-04 | 5.62e-05 | 0.016 | 0.306 | 99.2 | 100.0 |
| | | mean ± sd | 0.9999998 ± 0.0000000 | 9.74e-05 ± 1.12e-05 | 5.42e-05 ± 3.73e-06 | 0.015 ± 0.001 | 0.259 ± 0.104 | 99.3 ± 0.3 | 100.0 ± 0.0 |
| wedge | Sh | seed 0 | 0.9999970 | 3.57e-04 | 1.59e-04 | 0.011 | 0.445 | 98.5 | 100.0 |
| | | mean ± sd | 0.9999954 ± 0.0000047 | 4.10e-04 ± 1.90e-04 | 1.57e-04 ± 9.91e-06 | 0.010 ± 0.001 | 0.448 ± 0.249 | 99.0 ± 0.3 | 100.0 ± 0.0 |

The split-0 numbers sit inside the seed 0–4 spread, so there is no sign that split 0 was over-used. Nu RMSE matches §3; Sh RMSE (quad anchor) is 1.07–1.17× the §3 dual-encoding test values, within the split-to-split sd. This difference is on test data and must not be used to switch designs.

### 7.3 Checks (CLAUDE.md), 30 models = 6 targets × 5 split seeds (`results_final/checks.csv`)

| Check | Result |
| --- | --- |
| 1. Test R² ≥ 0.9999, RMSE ≤ 2× reference (§3) | R² passes everywhere (min 0.99999). RMSE **fails twice**: plate Sh split 3 (2.04×), wedge Sh split 3 (2.44×). |
| 2. Validation loss falls during L-BFGS | Passes for every seed. Final val/train ratio is 3.9–9.2× for Nu and 7–38× for Sh, above the 3–5× "normal" range, but validation loss decreases through L-BFGS, so this is not overfitting; the Sh gap comes from the few hard rows in §7.4. |
| 3. Error histogram centred (|mean| < 0.1 sd) | Passes everywhere (max 0.063). |
| 4. Max relative error < 1% | **Fails once**: cone Sh split 3, 1.51% (row 2603: Nt 1.25, Nb 0.21, Sh 0.9465 predicted 0.9608). |
| 5. Loss drops after Adam → L-BFGS | Passes everywhere. |

All three failures are Sh on split 3 and come from the same few rows (§7.4). Without its 5 worst rows, split-3 Sh RMSE is 2.3–3.8e-4, in line with the other splits.

### 7.4 Where Sh fails: small Nb, large Nt/Nb

- The worst Sh test rows have Nb ≈ 0.20–0.21 (the lower bound of Nb) and Nt/Nb ≈ 5–7, for every geometry and split. The same rows (e.g. 1096, 1660, 2603) top the list across geometries.
- Rows with Nt/Nb > 4 are about 4% of the test set but have 6–9× the mean relative Sh error (e.g. cone split 0: 0.102% vs 0.011%). On split 3, 7–8 of the 10 worst Sh rows have Nt/Nb > 4.
- Nu is not affected (0–2 of the 10 worst rows).
- The worst 5 rows carry 42–90% of the squared Sh error, which is why Sh RMSE varies so much between splits (sd up to 47% of the mean).
- Probable cause (not verified against the governing equations): the concentration equation of a Buongiorno-type nanofluid model contains (Nt/Nb)·θ″, so Sh changes steeply as Nb → its lower bound. A network on min–max-scaled inputs sees this only as a sharp corner.
- Corner location: these rows sit within 0.4% of the Nb face of the input cube (median distance of all test rows to the nearest face is 3%), consistent with the earlier corners-split finding that Sh errors at the edges are systematic.

### 7.5 Figures per final model

`results_pasr/data_<geometry>__<target>/loss_curve.png`, `error_histogram.png`, `parity.png` (split seed 0); the same files for split seeds 1–4 in `results_split<s>/<target>/data_<geometry>/`.

## 8. Nt/Nb as a feature (`--ratio-features`, Claude Code, 20 Oct 2026)

(b) = final design (per-target quad anchor, 3 seeds) + log(Nt/Nb), standardised on training rows, as an extra network input and an extra linear anchor term; (a) = final design.

### 8.1 Validation, split seeds 0–4 (decision)

Rule fixed before any (b) result: (b) wins for a target if its validation RMSE is lower in ≥ 12 of 15 (geometry × split) pairs. Source: `results_ratio/val_comparison.md`, `val_comparison_paired.csv`. (a) validation predictions rebuilt from saved weights (`analysis/rebuild_predictions.py`, all 30 reproduce their saved test predictions exactly).

| Target | (b) lower val RMSE | Decision | Median (a)/(b) val RMSE | Median (a)/(b), Nt/Nb > 4 rows |
| --- | --- | --- | --- | --- |
| Nu | 8 / 15 | no change | 1.02 | 1.04 |
| Sh | 14 / 15 | **adopt (b)** | 1.28 | 1.32 |

Validation, mean ± sd over split seeds 0–4:

| Geometry | Target | Design | Val RMSE | Val RMSE, Nt/Nb > 4 | Val RMSE, Nt/Nb ≤ 4 | Median rel. err. % | 95th pct rel. err. % | Max rel. err. % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cone | Nu | (a) | 1.13e-04 ± 2.14e-05 | 2.56e-04 ± 3.92e-05 | 1.04e-04 ± 2.04e-05 | 0.009 ± 0.000 | 0.043 ± 0.003 | 0.195 ± 0.067 |
| cone | Nu | (b) | 1.15e-04 ± 1.44e-05 | 2.38e-04 ± 4.79e-05 | 1.07e-04 ± 1.35e-05 | 0.010 ± 0.000 | 0.046 ± 0.003 | 0.179 ± 0.079 |
| cone | Sh | (a) | 4.94e-04 ± 1.30e-04 | 1.91e-03 ± 6.99e-04 | 3.21e-04 ± 3.54e-05 | 0.007 ± 0.001 | 0.045 ± 0.008 | 0.664 ± 0.463 |
| cone | Sh | (b) | 3.63e-04 ± 8.19e-05 | 1.23e-03 ± 4.51e-04 | 2.69e-04 ± 3.27e-05 | 0.006 ± 0.001 | 0.037 ± 0.006 | 0.350 ± 0.277 |
| plate | Nu | (a) | 9.33e-05 ± 1.80e-05 | 1.58e-04 ± 3.98e-05 | 9.01e-05 ± 1.85e-05 | 0.009 ± 0.001 | 0.044 ± 0.003 | 0.206 ± 0.093 |
| plate | Nu | (b) | 8.71e-05 ± 1.03e-05 | 1.45e-04 ± 4.18e-05 | 8.37e-05 ± 8.61e-06 | 0.010 ± 0.001 | 0.045 ± 0.006 | 0.199 ± 0.059 |
| plate | Sh | (a) | 2.98e-04 ± 6.53e-05 | 1.03e-03 ± 2.87e-04 | 2.23e-04 ± 3.60e-05 | 0.005 ± 0.000 | 0.034 ± 0.007 | 0.199 ± 0.066 |
| plate | Sh | (b) | 2.59e-04 ± 8.59e-05 | 8.83e-04 ± 4.68e-04 | 1.88e-04 ± 3.07e-05 | 0.005 ± 0.000 | 0.026 ± 0.004 | 0.168 ± 0.089 |
| wedge | Nu | (a) | 9.64e-05 ± 2.03e-05 | 1.61e-04 ± 5.00e-05 | 9.31e-05 ± 2.16e-05 | 0.010 ± 0.000 | 0.046 ± 0.003 | 0.210 ± 0.105 |
| wedge | Nu | (b) | 9.30e-05 ± 1.13e-05 | 1.57e-04 ± 4.60e-05 | 8.95e-05 ± 8.66e-06 | 0.010 ± 0.001 | 0.047 ± 0.004 | 0.193 ± 0.058 |
| wedge | Sh | (a) | 3.13e-04 ± 8.03e-05 | 1.11e-03 ± 3.81e-04 | 2.25e-04 ± 3.84e-05 | 0.005 ± 0.000 | 0.031 ± 0.005 | 0.220 ± 0.124 |
| wedge | Sh | (b) | 2.59e-04 ± 8.23e-05 | 9.42e-04 ± 4.61e-04 | 1.77e-04 ± 2.33e-05 | 0.005 ± 0.000 | 0.025 ± 0.003 | 0.186 ± 0.105 |

### 8.2 Test confirmation, Sh, split seeds 5–9 (test used once; no decision taken after it)

(a) and (b) both trained on seeds 5–9 so the confirmation has a comparison. Source: `results_confirm/report_b/paper_table.md` (final Sh), `results_confirm/report_a/paper_table.md` (comparison).

| Geometry | Target | Design | Split | R² | RMSE | MAE | MAPE % | Median rel. err. % | 95th pct rel. err. % | Max rel. err. % | Within 0.1 % | Within 0.5 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cone | Sh | quad anchor + log(Nt/Nb) | seed 5 | 0.9999981 | 2.97e-04 | 1.70e-04 | 0.010 | 0.006 | 0.031 | 0.351 | 99.6 | 100.0 |
| | | | mean ± sd (seeds 5–9) | 0.9999962 ± 0.0000027 | 4.10e-04 ± 1.38e-04 | 1.87e-04 ± 1.50e-05 | 0.012 ± 0.001 | 0.006 ± 0.000 | 0.036 ± 0.004 | 0.514 ± 0.183 | 99.0 ± 0.3 | 99.9 ± 0.0 |
| plate | Sh | quad anchor + log(Nt/Nb) | seed 5 | 0.9999984 | 2.61e-04 | 1.16e-04 | 0.008 | 0.004 | 0.023 | 0.415 | 99.6 | 100.0 |
| | | | mean ± sd (seeds 5–9) | 0.9999972 ± 0.0000022 | 3.26e-04 ± 1.25e-04 | 1.28e-04 ± 1.92e-05 | 0.008 ± 0.001 | 0.004 ± 0.000 | 0.026 ± 0.003 | 0.373 ± 0.117 | 99.5 ± 0.2 | 100.0 ± 0.0 |
| wedge | Sh | quad anchor + log(Nt/Nb) | seed 5 | 0.9999984 | 2.61e-04 | 1.17e-04 | 0.008 | 0.005 | 0.023 | 0.370 | 99.6 | 100.0 |
| | | | mean ± sd (seeds 5–9) | 0.9999971 ± 0.0000022 | 3.36e-04 ± 1.17e-04 | 1.30e-04 ± 1.77e-05 | 0.008 ± 0.001 | 0.004 ± 0.000 | 0.025 ± 0.003 | 0.367 ± 0.107 | 99.4 ± 0.2 | 100.0 ± 0.0 |
| cone | Sh | quad anchor (a) | mean ± sd (seeds 5–9) | 0.9999952 ± 0.0000017 | 4.74e-04 ± 8.04e-05 | 2.21e-04 ± 2.23e-05 | 0.014 ± 0.001 | 0.007 ± 0.000 | 0.046 ± 0.007 | 0.549 ± 0.012 | 98.3 ± 0.7 | 99.9 ± 0.0 |
| plate | Sh | quad anchor (a) | mean ± sd (seeds 5–9) | 0.9999953 ± 0.0000054 | 4.04e-04 ± 2.20e-04 | 1.58e-04 ± 2.00e-05 | 0.010 ± 0.001 | 0.005 ± 0.000 | 0.033 ± 0.003 | 0.397 ± 0.280 | 99.1 ± 0.4 | 100.0 ± 0.0 |
| wedge | Sh | quad anchor (a) | mean ± sd (seeds 5–9) | 0.9999950 ± 0.0000059 | 4.15e-04 ± 2.30e-04 | 1.56e-04 ± 1.93e-05 | 0.010 ± 0.001 | 0.005 ± 0.000 | 0.033 ± 0.003 | 0.430 ± 0.287 | 99.1 ± 0.3 | 100.0 ± 0.0 |

- (b) has lower test RMSE in 14 of 15 (geometry × split) pairs, median (a)/(b) = 1.12 (validation: 1.28; a smaller gain on fresh data is expected).
- (b) has a lower 95th-percentile relative error in 15 of 15 pairs, but a lower **maximum** relative error in only 6 of 15: the feature does not improve the worst case.
- Nu was not confirmed on test (it did not win on validation) and keeps design (a): paper numbers for Nu remain §7.2.

### 8.3 What the feature does and does not do

- It does **not** fix the small-Nb corner. Test RMSE on rows with Nt/Nb > 4 falls 14% (1.65e-3 → 1.42e-3), on the other rows 19% (2.64e-4 → 2.13e-4); the corner stays 6–7× harder, and on average 5.9 of the 10 worst rows still have Nt/Nb > 4 under both designs.
- It is a general improvement of Sh by about 1.1–1.3×, plausibly because log(Nt/Nb) gives the quadratic anchor a term it cannot build from x, x² and cross terms. The paper should describe it this way, not as a fix for the corner.

### 8.4 Checks, seeds 5–9 (`results_confirm/report_{a,b}/checks.csv`)

| Check | (a) | (b) |
| --- | --- | --- |
| 1. RMSE ≤ mean + 2 sd of the same model's seeds 0–4 | fails 2/15: plate, wedge split 6 | fails 3/15: cone, plate, wedge split 6 |
| 2. Validation loss falls in L-BFGS | passes (val/train 4.7–17×) | passes |
| 3. Histogram centred | passes | passes |
| 4. Max rel. error < 1% | passes (max 0.91%) | passes (max 0.79%) |
| 5. Loss drops after the switch | passes | passes |

- Split 6 is a hard split for Sh in both designs (its test set holds more of the hard rows); (b) is still better than (a) on it for plate and wedge (1.46×, 1.52×) and slightly worse for cone (0.95×). (b) fails check 1 more often because its own seed 0–4 band is narrower, not because it is worse.
- R² ≥ 0.99998 on every run (no longer part of check 1).
- The new check 1 does fire on new seeds, but on seeds 0–4 it passed all 30 runs (§7.3 failures included), because a failing run widens its own band.

### 8.5 Final design for the paper

| Target | Design | Command | Paper numbers |
| --- | --- | --- | --- |
| Nu | per-target, quad anchor, 3 seeds | `--targets Nu --anchor quad --seeds 3` | §7.2 (split seeds 0–4) |
| Sh | per-target, quad anchor + log(Nt/Nb), 3 seeds | `--targets Sh --anchor quad --ratio-features --seeds 3` | §8.2 (split seeds 5–9) |

Figures for the final Sh models: `results_confirm/b/split<s>/Sh/data_<geometry>/` (`loss_curve.png`, `error_histogram.png`, `parity.png`).

