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
6. **Sh near small Nb (new, §7.4).** Sh errors concentrate where Nt/Nb > 4. Candidate fix: add Nt/Nb as an input feature (method change, needs the user's agreement). Because this was found by looking at test rows, judge it on validation and confirm on split seeds that have not been used yet (5–9), not on 0–4.
7. Sh, quad anchor vs dual encoding: indistinguishable on validation at single-seed resolution (§7.1). If it matters for the paper, compare them with `--seeds 3` on validation.

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
