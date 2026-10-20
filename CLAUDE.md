# Nu–Sh neural surrogate (cone, plate, wedge)

Research project by Sandy (B.Tech AIML, GM University). Goal: neural-network surrogates that predict the Nusselt number (Nu) and Sherwood number (Sh) of a boundary-layer heat and mass transfer problem for three geometries, as accurately and as honestly as possible, for a journal paper.

Read these before doing anything:
- `docs/PROJECT_HISTORY.md`: everything tried so far and why decisions were made.
- `docs/RESULTS_AND_REVIEW.md`: every accuracy number obtained so far (including the user's own tuned run) and the open review points.

All training code is in `train_separate.py`. Do not rewrite it. Run it, read its outputs, and change it only when the user asks or a check below fails. Say what you changed and why.

## Data

| File (put in `data/`) | Rows | Inputs (sheet `Input`) | Targets (sheets) |
| --- | --- | --- | --- |
| `data_cone.xlsx` | 5,500 | β₁ (beta1), β₂ (beta2), α (alpha), Gr, δ (delta), Rd, Bi, Nt, Nb, Cr, P | `Nu`, `Sh` |
| `data_plate.xlsx` | 5,500 | same 11, identical rows across the three files | `Nu`, `Sh` |
| `data_wedge.xlsx` | 5,500 | same | `Nu`, `Sh` |

Facts already verified (do not re-derive unless the data changes):
- Inputs form an exact Latin hypercube: every one of 5,500 equal slices of each input's range holds exactly one row. Largest |correlation| between inputs is 0.03. A random train/test split is therefore a fair test.
- Ranges: β₁ 1–2, β₂ 0.1–0.5, α 0–0.5, Gr 1–2, δ 1–2, Rd 0.5–2.5, Bi 0.5–1, Nt 0.2–1.5, Nb 0.2–1.5, Cr 0.5–1.5, P 0.01–0.05.
- Nu spans 0.010–1.42 (about 100×), so it is modelled as log Nu. Sh spans 0.83–2.58.
- Plate and wedge outputs correlate at 0.9999. Whether a wedge angle should be an input is an open question for the user.
- Permutation importance: Nb, Rd, Nt drive Nu; Cr drives Sh; β₁, β₂, α, Gr, δ, P each score < 0.1 on every target.
- There is no Cf (skin friction) in these files.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU is enough (float64 is faster on CPU at this size). Training is about 3–5 minutes per network on 2 cores; run long jobs in the background and report progress.

## Commands

| Purpose | Command |
| --- | --- |
| Smoke test (seconds per network) | `python train_separate.py data/*.xlsx --quick --out results_quick` |
| Plain MLP, one model per dataset | `python train_separate.py data/*.xlsx --out results_mlp` |
| **Recommended final run** | `python train_separate.py data/*.xlsx --per-target --auto --seeds 3 --out results_pasr` |
| Fixed design instead of auto | add `--anchor quad` (chosen on validation for every geometry and target on 19 Oct 2026, see `docs/RESULTS_AND_REVIEW.md` §7) |
| Final configuration on a new split | `python analysis/split_commands.py results_final/final_config.csv <split seeds>`, then `python analysis/final_report.py results_final/final_config.csv` |

`--auto` trains four designs (MLP, quadratic anchor, dual encoding, both) and keeps the one with the lowest **validation** error; it writes `design_selection.csv`. It is about 4× slower than a single design.

## Method (do not change without the user's agreement)

- **Split:** 70% train / 10% validation / 20% test, random, `--split-seed 0` → 3,850 / 550 / 1,100 rows. The test set is never used for training, scaling or choosing anything.
- **Inputs:** min–max to [−1, 1] using training rows only.
- **Targets:** log for names starting with `Nu`; then standardised with training mean and sd. Metrics are always computed after converting back to physical units.
- **Network:** fully connected, 4 hidden layers × 128, SiLU, linear output.
- **Training:** full batch, float64; Adam (lr 3e-3, cosine decay, 2,500 steps) then L-BFGS (≤1,500 iterations, history 50, strong-Wolfe line search); loss = MSE on the standardised (or rescaled residual) target.
- **Quadratic-anchored residual design** (`--anchor quad`, called PASR-Net in earlier notes): fit a least-squares quadratic response surface p(x) (78 terms) on training rows; the network learns r = t − p(x) divided by its training sd s; prediction t̂ = p(x) + s·NN(x). Best for Nu.
- **Dual encoding** (`--dual-encoding`): network input = [scaled x, standardised log(x + 0.01·range)], 22 inputs. Beats the plain MLP for Sh, but not the quadratic anchor (19 Oct 2026, single seed, within noise).
- **Ensemble:** `--seeds 3` averages three independently initialised networks in physical units.
- **Ratio feature** (`--ratio-features`, default off, under evaluation since 20 Oct 2026): appends log(Nt/Nb), standardised on training rows, to the network inputs and as one extra linear term of the quadratic anchor (79 terms). Motivated by Sh errors concentrating where Nt/Nb > 4. With the flag off, results are bit-identical to earlier runs.

## Metrics written (`metrics.csv`, per target, for train / val / test)

R² (report as a decimal, e.g. 0.999998, never as a percentage), RMSE and MAE (absolute, same units as Nu/Sh), MaxAE, MAPE (%), MedRelErr and P95RelErr (median and 95th-percentile relative error, %), MaxRelErr (%), Within_0p1pct and Within_0p5pct (% of rows with relative error ≤ 0.1% / ≤ 0.5%).

Every report gives the median and 95th-percentile relative error next to the maximum: for Sh a handful of rows dominate the squared error and the maximum. (`metrics.csv` files written before 20 Oct 2026 lack these two columns; `analysis/final_report.py` computes them from `predictions_test.csv`.)

Do not add an "accuracy %" column: for regression it is just 100 − MAPE and reviewers object to it. If "RMSE %" or "MAE %" are wanted, state the formula (e.g. RMSE / mean(y) × 100) in the table caption.

## Outputs per model folder

`metrics.csv`, `loss_curve.png` (train + validation, Adam → L-BFGS marked), `error_histogram.png`, `parity.png`, `predictions_test.csv`, `predictions_val.csv` (since 20 Oct 2026; for older runs rebuild it with `analysis/rebuild_predictions.py`), `loss_history.csv`, `model_seed*.pt`, `scaler.json`, and `design_selection.csv` with `--auto`. Across models: `summary_metrics.csv`, `summary_r2_rmse.png`.

## Checks after every training run (report any failure to the user)

1. Test RMSE within mean + 2 sd of split seeds 0–4 for that model (`analysis/final_report.py`, `--band-template` for the folders of the seed 0–4 runs).
2. Validation loss keeps falling during L-BFGS. With noise-free solver data a validation/train loss ratio of 3–5× is normal; validation loss rising while train loss falls is overfitting.
3. Error histogram centred on zero (mean error < 10% of its sd).
4. MaxRelErr below about 1% for the final design; if not, list the worst rows from `predictions_test.csv`. For Nu they are expected to be the smallest Nu values.
5. Loss drops after the Adam → L-BFGS switch.

## Rules

- Choose designs, settings and seeds by **validation** error only. Report test error once, at the end.
- Keep `--split-seed 0` for comparability. To check for test-set overuse, rerun the final settings once with a new seed (e.g. `--split-seed 1`) and report both.
- Never delete or overwrite earlier `results*/` folders; use a new `--out` for each run.
- Do not claim accuracy outside the input ranges listed above.
- Differences under about 20% between single runs are within run-to-run noise. Use `--seeds 3` before claiming one design beats another.
- Already tested and rejected (do not re-add without new evidence): bigger network 6×256, power-law correlation backbone, cubic anchor, a second residual stage, log-transforming Sh, RNN/GRU, Padé activation, residual blocks, gradient boosting.
- If you change the method, update this file and `docs/PROJECT_HISTORY.md` in the same commit.
